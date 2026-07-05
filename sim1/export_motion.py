"""Export a reference motion (ASE clip, retargeted to our rig frame) to a flat file `sim1_viz` can
replay kinematically — step 2 of the mocap plan. Same spirit as `export_policy`: a dependency-free
whitespace format, no torch/json needed by the C++ reader.

Format (SIM1_MOTION_V2) — per-body state in OUR rig frame, wxyz quats, body order == makeAMPHumanoid:
    SIM1_MOTION_V2
    fps <float> num_frames <F> num_bodies <B>
    <F*B rows of: px py pz  qw qx qy qz  lvx lvy lvz  avx avy avz>   # frame-major, then body-major
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sim1.motion.motion_lib import (
    ReferenceMotion,
    default_motion_path,
    load_reference_motion,
    retarget_to_rig,
)


def _rig_rest_body_pos():
    """Our rig's authored body-center positions (our frame + body order) from the engine at reset."""
    from sim1.config import EnvConfig
    from sim1.envs.engine_vecenv import EngineVecEnv
    import numpy as np

    env = EngineVecEnv(EnvConfig(kind="engine", model="amp", num_envs=1), seed=0)
    env.reset(0)
    return np.asarray(env.body_pos)[0].copy()


def export_motion(motion: ReferenceMotion, out: str | Path) -> Path:
    out = Path(out)
    F, B = motion.num_frames, motion.num_bodies
    lines = ["SIM1_MOTION_V2", f"fps {motion.fps:.10g} num_frames {F} num_bodies {B}"]
    pos, quat = motion.body_pos, motion.body_quat            # (F,B,3), (F,B,4 wxyz)
    lin, ang = motion.body_linvel, motion.body_angvel        # (F,B,3), (F,B,3)
    for f in range(F):
        for b in range(B):
            p, q, l, a = pos[f, b], quat[f, b], lin[f, b], ang[f, b]
            lines.append(f"{p[0]:.8e} {p[1]:.8e} {p[2]:.8e} {q[0]:.8e} {q[1]:.8e} {q[2]:.8e} {q[3]:.8e} "
                         f"{l[0]:.8e} {l[1]:.8e} {l[2]:.8e} {a[0]:.8e} {a[1]:.8e} {a[2]:.8e}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export an ASE reference motion for sim1_viz replay.")
    ap.add_argument("--name", default="amp_humanoid_walk", help="clip name under the ASE motions dir")
    ap.add_argument("--path", default=None, help="explicit .npy path (overrides --name)")
    ap.add_argument("--out", default=None, help="output file (default: <name>.motion.txt)")
    ap.add_argument("--raw", action="store_true",
                    help="skip the rotation-based retarget (export ASE joint positions as-is)")
    args = ap.parse_args()

    path = Path(args.path) if args.path else default_motion_path(args.name)
    motion = load_reference_motion(path)
    if not args.raw:
        motion = retarget_to_rig(motion, _rig_rest_body_pos())   # onto our rig's proportions
    out = export_motion(motion, args.out or f"{args.name}.motion.txt")
    print(f"exported motion → {out}  ({'raw ASE joints' if args.raw else 'retargeted to our rig'})")
    print(f"  fps={motion.fps} frames={motion.num_frames} bodies={motion.num_bodies} "
          f"duration={motion.duration:.2f}s")


if __name__ == "__main__":
    main()
