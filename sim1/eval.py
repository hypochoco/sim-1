"""Evaluation & verification: load a checkpoint, run deterministic rollouts, and report whether the
humanoid is actually standing (not just scoring high reward).

For the `stand` task it computes physical diagnostics an agent can threshold — survival rate, mean
episode length vs the time limit, mean uprightness (world-up · torso-up), and mean root height vs
the standing height — and prints a STANDING ✓/✗ verdict. Returns a metrics dict for programmatic use.

Usage:
    python -m sim1.eval --run runs/<run_id> [--checkpoint best.pt] [--episodes 40]

Note: this is *numeric* verification (headless). True 3D visual playback needs the full (non-training)
engine build with the renderer — out of scope for the headless training binding.
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

# Verdict thresholds for the stand task.
_SURVIVAL_OK = 0.8   # fraction of episodes reaching the time limit without falling
_UPRIGHT_OK = 0.8    # mean uprightness (1.0 = perfectly vertical)


def evaluate(run: str, checkpoint: str = "best.pt", episodes: int = 40, device: str | None = None) -> dict:
    run_dir = Path(run)
    cfg = TrainConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    if device:
        cfg.run.device = device
    dev = resolve_device(cfg.run.device)

    env = build_env(cfg)
    trainer = PPOTrainer(env, cfg.ppo, device=dev, seed=cfg.run.seed)
    ckpt = run_dir / "checkpoints" / checkpoint
    if not ckpt.exists():
        ckpt = run_dir / "checkpoints" / "final.pt"
    trainer.load_state_dict(load_checkpoint(ckpt, map_location=dev))

    is_stand = cfg.task.name == "stand"
    is_track = cfg.task.name == "track"
    n = env.num_envs

    ep_returns: list[float] = []
    ep_lens: list[int] = []
    ep_survived: list[bool] = []
    ep_ret = np.zeros(n)
    ep_len = np.zeros(n, dtype=np.int64)
    up_sum = h_sum = 0.0
    step_count = 0
    term_sums: dict[str, float] = {}   # track: per-term reward accumulation (eval-side diagnostic)
    term_steps = 0

    obs = trainer._process_obs(env.reset(), update=False)
    target_h = float(np.mean(env.task._target_h)) if is_stand else float("nan")

    while len(ep_returns) < episodes:
        with torch.no_grad():
            action = trainer.model.actor_mean(obs)  # deterministic (mean) action
        obs_np, reward, done, info = env.step(action.cpu().numpy())
        ep_ret += reward
        ep_len += 1

        if is_stand:
            term = info["terminal_obs"]  # [height, quat_wxyz(4), ...]
            h = term[:, 0]
            qx, qz = term[:, 2], term[:, 4]
            up = 1.0 - 2.0 * (qx * qx + qz * qz)
            up_sum += float(np.sum(up))
            h_sum += float(np.sum(h))
            step_count += n

        if is_track:   # accumulate the reward variant's per-term means (computed in task.reward)
            for k, v in env.task.reward_info.items():
                term_sums[k] = term_sums.get(k, 0.0) + v
            term_steps += 1

        for i in np.nonzero(done)[0]:
            ep_returns.append(float(ep_ret[i]))
            ep_lens.append(int(ep_len[i]))
            ep_survived.append(bool(info["truncated"][i] and not info["fail"][i]))
            ep_ret[i] = 0.0
            ep_len[i] = 0
        obs = trainer._process_obs(obs_np, update=False)

    result = {
        "episodes": len(ep_returns),
        "mean_return": float(np.mean(ep_returns)),
        "mean_ep_len": float(np.mean(ep_lens)),
        "episode_len": env.episode_len,
    }
    print(f"eval [{cfg.task.name}, {cfg.env.model if cfg.env.kind == 'engine' else 'mock'}]: "
          f"{result['episodes']} episodes")
    print(f"  mean return         {result['mean_return']:8.2f}")
    print(f"  mean episode length {result['mean_ep_len']:8.1f} / {env.episode_len}")

    if is_stand:
        survival = float(np.mean(ep_survived))
        mean_up = up_sum / max(1, step_count)
        mean_h = h_sum / max(1, step_count)
        result.update({"survival_rate": survival, "mean_uprightness": mean_up,
                       "mean_root_height": mean_h, "target_height": target_h})
        standing = survival >= _SURVIVAL_OK and mean_up >= _UPRIGHT_OK
        result["standing"] = bool(standing)
        print(f"  survival rate       {survival * 100:7.1f}%   (reached the time limit without falling)")
        print(f"  mean uprightness    {mean_up:8.3f}   (1.0 = vertical)")
        print(f"  mean root height    {mean_h:8.3f} m (standing ≈ {target_h:.3f} m)")
        print(f"  VERDICT: {'STANDING ✓' if standing else 'NOT STANDING ✗'} "
              f"(needs survival ≥ {_SURVIVAL_OK:.0%} and uprightness ≥ {_UPRIGHT_OK})")

    if is_track and term_steps:
        breakdown = {k: v / term_steps for k, v in term_sums.items()}
        result["reward_variant"] = cfg.task.track_reward
        result["reward_terms"] = breakdown
        survival = float(np.mean(ep_survived))
        result["survival_rate"] = survival
        print(f"  reward variant      {cfg.task.track_reward}")
        print(f"  survival rate       {survival * 100:7.1f}%   (reached the time limit without diverging)")
        print("  per-term reward (mean over rollout):")
        for k, v in breakdown.items():
            print(f"      {k:9s} {v:7.3f}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate/verify a trained sim1 policy.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(args.run, args.checkpoint, args.episodes, args.device)


if __name__ == "__main__":
    main()
