"""Training entrypoint.

Usage:
    python -m sim1.train --name ppo_reach --override ppo.total_steps=200000 env.num_envs=64
    python -m sim1.train --resume runs/<run_id>

The config *framework* is deferred: defaults live in `sim1.config` dataclasses, with light dotted
overrides (`section.field=value`). Each run gets an organized, resumable directory with a frozen
config, TensorBoard + JSONL metrics, a tqdm progress bar, and periodic + best + final checkpoints.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from sim1.algos.ppo import PPOTrainer
from sim1.config import TrainConfig
from sim1.envs.engine_vecenv import make_vecenv
from sim1.envs.task_env import TaskEnv
from sim1.tasks.reach import ReachTask
from sim1.tasks.stand import StandTask
from sim1.tasks.getup import GetUpTask
from sim1.tasks.walk import WalkTask
from sim1.utils.checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint, step_number
from sim1.utils.logging import MetricLogger
from sim1.utils.run_dir import create_run_dir
from sim1.utils.seeding import seed_everything


def resolve_device(name: str) -> str:
    if name == "cuda" and torch.cuda.is_available():
        return "cuda"
    if name == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_task(cfg: TrainConfig, vecenv):
    name = cfg.task.name
    if name == "reach":
        return ReachTask(
            ndof=vecenv.act_dim,
            pos_weight=cfg.task.pos_weight,
            vel_weight=cfg.task.vel_weight,
            action_weight=cfg.task.action_weight,
            target_scale=cfg.env.target_scale,
        )
    if name in ("stand", "getup"):
        # torque mode: map unit policy output onto the torque range; PD-target mode: onto radians.
        action_scale = cfg.env.max_torque if cfg.env.action_mode == "torque" else cfg.task.pd_action_scale
        cls = StandTask if name == "stand" else GetUpTask
        return cls(
            ndof=vecenv.ndof,
            nbody=vecenv.nbody,
            act_dim=vecenv.act_dim,
            action_scale=action_scale,
            upright_weight=cfg.task.upright_weight,
            height_weight=cfg.task.height_weight,
            alive_bonus=cfg.task.alive_bonus,
            action_weight=cfg.task.action_weight,
            fall_height_frac=cfg.task.fall_height_frac,
            upright_fall=cfg.task.upright_fall,
        )
    if name == "walk":
        action_scale = cfg.env.max_torque if cfg.env.action_mode == "torque" else cfg.task.pd_action_scale
        return WalkTask(
            ndof=vecenv.ndof,
            nbody=vecenv.nbody,
            act_dim=vecenv.act_dim,
            action_scale=action_scale,
            target_speed_range=(cfg.task.target_speed_min, cfg.task.target_speed_max),
            upright_weight=cfg.task.upright_weight,
            height_weight=cfg.task.height_weight,
            alive_bonus=cfg.task.alive_bonus,
            action_weight=cfg.task.action_weight,
            command_weight=cfg.task.command_weight,
            fall_height_frac=cfg.task.fall_height_frac,
            upright_fall=cfg.task.upright_fall,
        )
    raise ValueError(f"unknown task {name!r} (expected 'reach', 'stand', 'getup', or 'walk')")


def build_env(cfg: TrainConfig) -> TaskEnv:
    vecenv = make_vecenv(cfg.env, seed=cfg.run.seed)
    task = build_task(cfg, vecenv)
    return TaskEnv(vecenv, task, episode_len=cfg.env.episode_len, seed=cfg.run.seed)


def _newest_checkpoint(ckpt_dir: Path) -> Path | None:
    final = ckpt_dir / "final.pt"
    if final.exists():
        return final
    steps = sorted(ckpt_dir.glob("step_*.pt"), key=step_number)  # numeric, not lexicographic
    return steps[-1] if steps else None


def _warm_start(trainer: PPOTrainer, ckpt_path: str, device: str) -> None:
    """Initialize a FRESH run from a prior policy's weights (transfer / curriculum): load only the
    network + obs-normalizer, keeping a fresh optimizer/step/rng. Shape-tolerant — tensors that don't
    match (e.g. the input layer when a new task adds goal channels, or the output layer on a rig
    change) are skipped and left at their fresh init, so the shared trunk still transfers."""
    s = load_checkpoint(ckpt_path, map_location=device)
    src, dst = s["model"], trainer.model.state_dict()
    loaded, skipped = [], []
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    trainer.model.load_state_dict(dst)
    if trainer.obs_rms is not None and s.get("obs_rms") is not None and \
       s["obs_rms"]["mean"].shape == trainer.obs_rms.mean.shape:
        trainer.obs_rms.load_state_dict(s["obs_rms"])
    print(f"warm-start from {ckpt_path}: loaded {len(loaded)} tensors"
          + (f", skipped {len(skipped)} (shape mismatch → fresh init): {skipped}" if skipped else ""))


def run_training(cfg: TrainConfig, resume: str | None = None, init_from: str | None = None) -> dict:
    device = resolve_device(cfg.run.device)
    seed_everything(cfg.run.seed)

    env = build_env(cfg)
    trainer = PPOTrainer(env, cfg.ppo, device=device, seed=cfg.run.seed)

    if resume:
        run_dir = Path(resume)
        ckpt = _newest_checkpoint(run_dir / "checkpoints")
        if ckpt is not None:
            trainer.load_state_dict(load_checkpoint(ckpt, map_location=device))
    else:
        run_dir = create_run_dir(cfg.run.runs_root, cfg.run.name, cfg.to_dict(), cfg.run.seed)
        if init_from:  # warm-start a fresh run from a prior policy (transfer / curriculum)
            _warm_start(trainer, init_from, device)

    logger = MetricLogger(run_dir)
    history: list[dict] = []
    best_return = -float("inf")
    base_lr = cfg.ppo.lr
    total = cfg.ppo.total_steps
    start = time.time()

    pbar = tqdm(total=total, initial=trainer.global_step, desc=cfg.run.name, unit="step")
    while trainer.global_step < total:
        if cfg.ppo.anneal_lr:
            frac = max(0.0, 1.0 - trainer.iteration / trainer.num_iterations)
            for g in trainer.opt.param_groups:
                g["lr"] = frac * base_lr

        batch, roll_metrics = trainer.collect()
        upd_metrics = trainer.update(batch)

        sps = trainer.global_step / max(1e-9, time.time() - start)
        metrics = {
            **roll_metrics,
            **upd_metrics,
            "charts/learning_rate": trainer.opt.param_groups[0]["lr"],
            "charts/sps": sps,
        }
        logger.log(metrics, trainer.global_step)
        history.append({"global_step": trainer.global_step, "iteration": trainer.iteration, **metrics})

        ret = roll_metrics["charts/ep_return_mean"]
        pbar.set_postfix(ret=(f"{ret:.2f}" if ret == ret else "n/a"), sps=f"{sps:.0f}")
        pbar.update(trainer.global_step - pbar.n)

        # best checkpoint (by episode return once available)
        if ret == ret and ret > best_return:
            best_return = ret
            save_checkpoint(run_dir / "checkpoints" / "best.pt", trainer.state_dict())
        # periodic checkpoint
        if trainer.iteration % cfg.run.checkpoint_interval == 0:
            save_checkpoint(run_dir / "checkpoints" / f"step_{trainer.global_step}.pt", trainer.state_dict())
            prune_checkpoints(run_dir / "checkpoints", cfg.run.keep_last)

    save_checkpoint(run_dir / "checkpoints" / "final.pt", trainer.state_dict())
    pbar.close()
    logger.close()
    return {"run_dir": str(run_dir), "history": history, "best_return": best_return}


def main() -> None:
    ap = argparse.ArgumentParser(description="sim1 PPO training (P0)")
    ap.add_argument("--name", type=str, default=None, help="run name")
    ap.add_argument("--resume", type=str, default=None, help="path to an existing run dir to resume")
    ap.add_argument("--init-from", type=str, default=None,
                    help="warm-start a FRESH run from a prior checkpoint's weights (transfer/curriculum)")
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda | mps")
    ap.add_argument("--override", "-o", action="append", default=[], help="config override, e.g. ppo.lr=1e-3")
    args = ap.parse_args()

    if args.resume:
        cfg = TrainConfig.from_dict(json.loads((Path(args.resume) / "config.json").read_text()))
    else:
        cfg = TrainConfig()
    cfg.apply_overrides(args.override)
    if args.name:
        cfg.run.name = args.name
    if args.device:
        cfg.run.device = args.device

    result = run_training(cfg, resume=args.resume, init_from=args.init_from)
    print(f"done. run_dir={result['run_dir']} best_return={result['best_return']:.3f}")


if __name__ == "__main__":
    main()
