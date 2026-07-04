"""Evaluation: load a checkpoint and run deterministic rollouts, reporting mean return.

Usage: python -m sim1.eval --run runs/<run_id> [--episodes 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sim1.algos.ppo import PPOTrainer
from sim1.config import TrainConfig
from sim1.train import build_env, resolve_device
from sim1.utils.checkpoint import load_checkpoint


def evaluate(run: str, checkpoint: str = "best.pt", episodes: int = 20) -> float:
    run_dir = Path(run)
    cfg = TrainConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    device = resolve_device(cfg.run.device)

    env = build_env(cfg)
    trainer = PPOTrainer(env, cfg.ppo, device=device, seed=cfg.run.seed)
    ckpt = run_dir / "checkpoints" / checkpoint
    if not ckpt.exists():
        ckpt = run_dir / "checkpoints" / "final.pt"
    trainer.load_state_dict(load_checkpoint(ckpt, map_location=device))

    steps = episodes * cfg.env.episode_len
    returns: list[float] = []
    ep_ret = np.zeros(env.num_envs)
    obs = trainer._process_obs(env.reset(), update=False)
    for _ in range(steps):
        with torch.no_grad():
            mean = trainer.model.actor_mean(obs)  # deterministic (mean) action
        obs_np, reward, done, _ = env.step(mean.cpu().numpy())
        ep_ret += reward
        for i in np.nonzero(done)[0]:
            returns.append(float(ep_ret[i]))
            ep_ret[i] = 0.0
        obs = trainer._process_obs(obs_np, update=False)

    mean_ret = float(np.mean(returns)) if returns else float("nan")
    print(f"eval: {len(returns)} episodes, mean_return={mean_ret:.3f}")
    return mean_ret


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    evaluate(args.run, args.checkpoint, args.episodes)


if __name__ == "__main__":
    main()
