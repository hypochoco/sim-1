"""Export a trained policy to a portable, dependency-free text file a C++ visual runner can load.

Closes the loop for visualization: the headless trainer produces a checkpoint; this writes the
*deterministic* policy (actor mean) + the observation normalizer + the sim knobs needed to reproduce
the training dynamics, in a flat whitespace-delimited format that C++ parses with `ifstream >>`
(no JSON/torch dependency on the engine side).

Usage:
    python -m sim1.export_policy --run runs/<run_id> [--checkpoint best.pt] [--out <path>]

File format (SIM1_POLICY_V8) — a sequence of `label value` tokens then arrays, read positionally:
    SIM1_POLICY_V8
    model <str> backend <str> action_mode <str>
    substeps <int> control_dt <float> kp <float> kd <float> max_torque <float> ground_friction <float>
    episode_len <int> fall_height_frac <float> upright_fall <float>
    ndof <int> nbody <int> obs_dim <int> act_dim <int> action_scale <float> norm_eps <float>
    command_type <str> command_dim <int> rotation <str> frame <str> body_obs <int> terminate_on_fall <int> motion_duration <float>
    norm_mean <obs_dim floats>   # V2 cmd, V3 rot, V4 frame, V5 body_obs, V6 term_on_fall, V7 ground_friction, V8 motion_duration
    norm_var  <obs_dim floats>   # command_type "phase" (tracking) → the command channels are a (sin,cos) phase clock, period motion_duration
    n_layers <L>
    (per layer, in forward order; activation is tanh on all but the last)
    layer <out> <in>
      <out*in weights, row-major [out][in]>
      <out biases>
The C++ forward is: x = (obs - norm_mean) / sqrt(norm_var + norm_eps); for each layer y = W x + b,
tanh(y) except the final layer; action = y_final * action_scale; env.setAction(action); env.step().
Compose `obs` exactly as StandTask.observe: from Environment::packDefaultObs, keep index 1 (root
height) and indices 3.. (drop root x,z) → [height, quat4, linvel3, angvel3, q[ndof], qd[ndof], contacts].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _linear_layers(model_sd: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract (weight, bias) for each Linear in the actor_mean Sequential, in forward order."""
    idxs = sorted({int(k.split(".")[1]) for k in model_sd if k.startswith("actor_mean.") and k.endswith(".weight")})
    layers = []
    for i in idxs:
        w = model_sd[f"actor_mean.{i}.weight"].cpu().numpy().astype(np.float64)  # [out, in]
        b = model_sd[f"actor_mean.{i}.bias"].cpu().numpy().astype(np.float64)    # [out]
        layers.append((w, b))
    return layers


def export(run: str, checkpoint: str = "best.pt", out: str | None = None) -> Path:
    run_dir = Path(run)
    cfg = json.loads((run_dir / "config.json").read_text())
    env, task = cfg["env"], cfg["task"]

    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "final.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    layers = _linear_layers(ckpt["model"])
    obs_dim = layers[0][0].shape[1]
    act_dim = layers[-1][0].shape[0]

    rms = ckpt.get("obs_rms")
    if rms is not None:
        mean = rms["mean"].cpu().numpy().astype(np.float64)
        var = rms["var"].cpu().numpy().astype(np.float64)
    else:  # obs normalization was disabled → identity
        mean = np.zeros(obs_dim)
        var = np.ones(obs_dim)

    action_scale = float(env["max_torque"]) if env["action_mode"] == "torque" else float(task["pd_action_scale"])

    # nbody is rig-defined; command_dim is whatever the obs carries beyond the proprioception block
    # (which itself depends on the rotation encoding: quat=4, sixd=6).
    nbody = {"amp": 15, "humanoid": 14}.get(env["model"], obs_dim - (1 + 4 + 3 + 3 + 2 * act_dim))
    rotation = task.get("rotation") or "quat"   # older frozen configs may store null / omit these
    rot_dim = 6 if rotation == "sixd" else 4
    proprio = 1 + rot_dim + 3 + 3 + 2 * act_dim + nbody
    # per-body 6D block (SuperTrack, rot=sixd): nbody·(pos3 + 6D6 + linvel3 + angvel3 + height1) + up3
    body_obs = bool(task.get("body_obs", False))
    per_body = (nbody * (3 + 6 + 3 + 3 + 1) + 3) if body_obs else 0
    command_dim = obs_dim - proprio - per_body
    command_type = {"walk": "heading_speed", "track": "phase"}.get(task["name"], "none")
    frame = task.get("frame") or "world"
    # Whether the task ends an episode on a fall. `getup` uses no fall termination (never_terminate) —
    # it recovers instead — so the visualizer must NOT auto-reset it on a fall (mirror training).
    terminate_on_fall = 0 if task["name"] == "getup" else 1
    # Tracking policies read a phase clock; the visualizer needs the clip duration to advance it.
    motion_duration = 0.0
    if command_type == "phase":
        from sim1.motion.motion_lib import default_motion_path, load_reference_motion
        motion_duration = load_reference_motion(
            default_motion_path(task.get("motion_clip", "amp_humanoid_walk"))).duration

    def fmt(a: np.ndarray) -> str:
        return " ".join(f"{x:.8e}" for x in np.asarray(a).ravel())

    lines = [
        "SIM1_POLICY_V8",
        f"model {env['model']} backend {env.get('backend', 'reduced')} action_mode {env['action_mode']}",
        f"substeps {env['substeps']} control_dt {env['control_dt']:.10g} kp {env['kp']:.10g} "
        f"kd {env['kd']:.10g} max_torque {env['max_torque']:.10g} "
        f"ground_friction {env.get('ground_friction', 0.9):.10g}",
        f"episode_len {env['episode_len']} fall_height_frac {task['fall_height_frac']:.10g} "
        f"upright_fall {task['upright_fall']:.10g}",
        f"ndof {act_dim} nbody {nbody} "
        f"obs_dim {obs_dim} act_dim {act_dim} action_scale {action_scale:.10g} norm_eps 1e-8",
        f"command_type {command_type} command_dim {command_dim} rotation {rotation} frame {frame} "
        f"body_obs {int(body_obs)} terminate_on_fall {terminate_on_fall} motion_duration {motion_duration:.10g}",
        f"norm_mean {fmt(mean)}",
        f"norm_var {fmt(var)}",
        f"n_layers {len(layers)}",
    ]
    for w, b in layers:
        lines.append(f"layer {w.shape[0]} {w.shape[1]}")
        lines.append(fmt(w))   # row-major [out][in]
        lines.append(fmt(b))

    out_path = Path(out) if out else run_dir / "policy.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"exported policy → {out_path}")
    print(f"  model={env['model']} action_mode={env['action_mode']} obs_dim={obs_dim} act_dim={act_dim} "
          f"layers={[w.shape[0] for w, _ in layers]} action_scale={action_scale:g}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a trained sim1 policy for the C++ visual runner.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    export(args.run, args.checkpoint, args.out)


if __name__ == "__main__":
    main()
