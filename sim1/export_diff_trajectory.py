"""Roll out a trained policy on the **diff-cpu** backend and dump its per-frame body trajectory as a
`SIM1_MOTION_V2` file that `sim1_viz` replays kinematically — Option B for visualizing diff-ABA
policies (the visualizer runs only the reduced/realtime PhysicsWorld, so we can't step the diff env
live yet; we record a faithful diff-ABA rollout and replay it).

The diff-cpu backend runs the SAME physics as the CUDA training backend, so this is a faithful
recording. Per-body world transforms come from the diff env's `linkWorld` readout (validated to match
the reduced backend's authored pose to ~1e-9).

Usage:
    python -m sim1.export_diff_trajectory --run runs/<id> [--checkpoint best.pt] [--frames 300] [--out traj.motion.txt]
    ./build-viz/csrc/viz/sim1_viz <traj.motion.txt>     # first arg is a motion file → replay mode
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


def export_diff_trajectory(run: str, checkpoint: str = "best.pt", frames: int = 300,
                           out: str | None = None, device: str = "cpu", seed: int = 0) -> Path:
    run_dir = Path(run)
    cfg = TrainConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    # Replay locally on the CPU diff backend (same physics as a cuda run; no CUDA needed here).
    if cfg.env.kind in ("cuda", "diff-cpu"):
        cfg.env.kind = "diff-cpu"
    else:
        raise SystemExit(f"run trained on kind={cfg.env.kind!r}; this dumper is for diff/cuda backends "
                         "(reduced-backend policies already visualize via sim1_viz <policy.txt>).")
    cfg.env.num_envs = 1
    cfg.run.device = device
    dev = resolve_device(device)

    env = build_env(cfg)
    ve = env.env
    trainer = PPOTrainer(env, cfg.ppo, device=dev, seed=seed)
    ckpt = run_dir / "checkpoints" / checkpoint
    if not ckpt.exists():
        ckpt = run_dir / "checkpoints" / "final.pt"
    trainer.load_state_dict(load_checkpoint(ckpt, map_location=dev))

    obs = trainer._process_obs(env.reset(), update=False)
    B = ve.nbody
    P, Q, L, A = [], [], [], []
    for _ in range(frames):
        # record the current diff-ABA state (reset pose on frame 0; then post-step states)
        P.append(np.asarray(ve.body_pos)[0].copy())
        Q.append(np.asarray(ve.body_quat)[0].copy())
        L.append(np.asarray(ve.body_linvel)[0].copy())
        A.append(np.asarray(ve.body_angvel)[0].copy())
        with torch.no_grad():
            a = trainer.model.actor_mean(obs)
        obs_np, _, _, _ = env.step(a.cpu().numpy())
        obs = trainer._process_obs(obs_np, update=False)

    fps = 1.0 / float(cfg.env.control_dt)
    out_path = Path(out) if out else run_dir / "trajectory.motion.txt"
    lines = ["SIM1_MOTION_V2", f"fps {fps:.10g} num_frames {frames} num_bodies {B}"]
    for f in range(frames):
        for b in range(B):
            p, q, l, a = P[f][b], Q[f][b], L[f][b], A[f][b]
            lines.append(" ".join(f"{x:.8e}" for x in (*p, *q, *l, *a)))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"exported diff-cpu trajectory → {out_path}  ({frames} frames, {B} bodies, fps {fps:.3g}, "
          f"model={cfg.env.model}, task={cfg.task.name})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump a diff-cpu policy rollout as a sim1_viz-replayable trajectory.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    export_diff_trajectory(args.run, args.checkpoint, args.frames, args.out, args.device)


if __name__ == "__main__":
    main()
