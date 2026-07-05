"""Tracking-reward registry — a library of named, self-contained imitation-reward variants.

We iterate a lot on the tracking reward, so each variant is a full recipe (weights + terms +
termination) living here, selectable by name (`config.task.track_reward`, recorded in the run's
config.json). Shared physics/error math is factored out (`track_errors`, helpers) so variants stay
DRY, but each variant's **composition** is written out in full for easy side-by-side comparison.

Contract: `make_reward(name, nbody, ...)` → a `TrackingReward` whose `evaluate(sim, ref, actions)`
returns `RewardOut(total, info, failed)`:
  - `total`  (N,)   per-env reward for this step,
  - `info`   dict   per-term means (for eval-side diagnostics; NOT logged during training),
  - `failed` (N,)   early-termination mask.
`sim` and `ref` are anything with `.body_pos/.body_quat/.body_linvel/.body_angvel` of shape (N,B,·)
(the sim `BodyState` and the reference `ReferenceState` both qualify), all in the world/our frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim1.tasks.proprio import _quat_mul, _yaw_from_quat

HANDS_AND_FEET = (5, 8, 13, 14)   # end-effectors in our feet-last body order
FEET = (13, 14)


# --- shared helpers ---------------------------------------------------------------------------
def _quat_geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) between batched wxyz quats a,b (..., 4) → (...)."""
    d = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(d, 0.0, 1.0))


def heading_frame(bp: np.ndarray, bq: np.ndarray, bl: np.ndarray, ba: np.ndarray):
    """Per-body state in the ROOT's heading frame (global yaw removed) → direction-invariant.
    Positions become root-relative and the whole set is rotated by −(root yaw) about +Y; body
    orientations are left-multiplied by the inverse root-heading. Returns (rel_pos, quat, linvel,
    angvel), all yaw-canceled, so a global heading change leaves them unchanged."""
    bp = np.asarray(bp, np.float32); bq = np.asarray(bq, np.float32)
    bl = np.asarray(bl, np.float32); ba = np.asarray(ba, np.float32)
    n, b = bp.shape[0], bp.shape[1]
    yaw = _yaw_from_quat(bq[:, 0])
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]

    def rot_xz(v):
        x, z = v[..., 0], v[..., 2]
        return np.stack([c * x - s * z, v[..., 1], s * x + c * z], axis=-1).astype(np.float32)

    rel = rot_xz(bp - bp[:, :1, :])
    lv, av = rot_xz(bl), rot_xz(ba)
    half = 0.5 * yaw
    qhi = np.stack([np.cos(half), np.zeros_like(half), -np.sin(half), np.zeros_like(half)], axis=1).astype(np.float32)
    q = _quat_mul(np.repeat(qhi, b, axis=0), bq.reshape(n * b, 4)).reshape(n, b, 4)
    return rel, q, lv, av


def foot_slip(sim, feet, height_thresh: float = 0.08) -> np.ndarray:
    """Sum of squared horizontal foot speed while a foot is near the ground (a grounded foot should
    not translate) → penalizes the shuffle/slide hack. (N,)."""
    fy = np.asarray(sim.body_pos)[:, feet, 1]
    v = np.asarray(sim.body_linvel)[:, feet]
    horiz2 = v[..., 0] ** 2 + v[..., 2] ** 2
    contact = (fy < height_thresh).astype(np.float32)
    return np.sum(contact * horiz2, axis=1)


@dataclass
class BodyState:
    """Per-body world state (N, B, ·); mirrors ReferenceState's fields so both feed the reward math."""
    body_pos: np.ndarray
    body_quat: np.ndarray
    body_linvel: np.ndarray
    body_angvel: np.ndarray


@dataclass
class TrackErrors:
    ori: np.ndarray          # (N,) Σ per-(non-root)-body orientation geodesic²
    pos: np.ndarray          # (N,) Σ root-relative body position error²
    vel: np.ndarray          # (N,) Σ body linear + 0.1·angular velocity error²
    ee: np.ndarray           # (N,) Σ end-effector root-relative position error²
    root: np.ndarray         # (N,) root height² + tilt² + 0.1·root velocity²
    rms: np.ndarray          # (N,) sqrt(pos / nbody) — for termination
    root_h_sim: np.ndarray   # (N,)
    root_h_ref: np.ndarray   # (N,)
    s_root_vel: np.ndarray   # (N,3) sim root heading-frame linear velocity
    r_root_vel: np.ndarray   # (N,3) ref root heading-frame linear velocity


def track_errors(sim, ref, ee: np.ndarray, nbody: int) -> TrackErrors:
    """All the heading-frame tracking errors, computed once (shared by every variant)."""
    s_rel, s_q, s_lv, s_av = heading_frame(sim.body_pos, sim.body_quat, sim.body_linvel, sim.body_angvel)
    r_rel, r_q, r_lv, r_av = heading_frame(ref.body_pos, ref.body_quat, ref.body_linvel, ref.body_angvel)
    root_h_sim = np.asarray(sim.body_pos)[:, 0, 1]
    root_h_ref = np.asarray(ref.body_pos)[:, 0, 1]
    pos = np.sum((s_rel - r_rel) ** 2, axis=(1, 2))
    return TrackErrors(
        ori=np.sum(_quat_geodesic(s_q[:, 1:], r_q[:, 1:]) ** 2, axis=1),
        pos=pos,
        vel=np.sum((s_lv - r_lv) ** 2, axis=(1, 2)) + 0.1 * np.sum((s_av - r_av) ** 2, axis=(1, 2)),
        ee=np.sum((s_rel[:, ee] - r_rel[:, ee]) ** 2, axis=(1, 2)),
        root=((root_h_sim - root_h_ref) ** 2 + _quat_geodesic(s_q[:, 0], r_q[:, 0]) ** 2
              + 0.1 * np.sum((s_lv[:, 0] - r_lv[:, 0]) ** 2, axis=1)),
        rms=np.sqrt(pos / nbody),
        root_h_sim=root_h_sim, root_h_ref=root_h_ref,
        s_root_vel=s_lv[:, 0], r_root_vel=r_lv[:, 0],
    )


@dataclass
class RewardOut:
    total: np.ndarray
    info: dict
    failed: np.ndarray


# --- variants ---------------------------------------------------------------------------------
class TrackingReward:
    """Base: holds the common config; subclasses implement `evaluate`."""
    name = "base"

    def __init__(self, nbody: int, *, ee=HANDS_AND_FEET, fall_height_frac: float = 0.5,
                 term_pos_err: float = 0.5, **_ignored):
        self.nbody = int(nbody)
        self.ee = np.array(ee, dtype=np.int64)
        self.fall_height_frac = float(fall_height_frac)
        self.term_pos_err = float(term_pos_err)
        # error sharpness (DeepMimic-inspired) — shared defaults
        self.a_pose, self.a_vel, self.a_ee, self.a_root = 2.0, 0.1, 40.0, 10.0

    def _failed(self, e: TrackErrors) -> np.ndarray:
        return (e.rms > self.term_pos_err) | (e.root_h_sim < self.fall_height_frac * e.root_h_ref)

    def evaluate(self, sim, ref, actions) -> RewardOut:
        raise NotImplementedError


class RewardV1(TrackingReward):
    """Baseline (the reward the first track_walk run used): heading-invariant pose/vel/ee/root +
    alive − small control penalty; terminate on RMS body error or fall."""
    name = "v1"

    def __init__(self, nbody, *, pose_weight=0.5, vel_weight=0.05, ee_weight=0.2, root_weight=0.15,
                 alive_bonus=0.1, action_weight=0.002, **kw):
        super().__init__(nbody, **kw)
        self.w_pose, self.w_vel, self.w_ee, self.w_root = pose_weight, vel_weight, ee_weight, root_weight
        self.alive, self.action_weight = float(alive_bonus), float(action_weight)

    def evaluate(self, sim, ref, actions) -> RewardOut:
        e = track_errors(sim, ref, self.ee, self.nbody)
        r_pose = np.exp(-self.a_pose * e.ori)
        r_vel = np.exp(-self.a_vel * e.vel)
        r_ee = np.exp(-self.a_ee * e.ee)
        r_root = np.exp(-self.a_root * e.root)
        ctrl = self.action_weight * np.mean(np.asarray(actions) ** 2, axis=1)
        total = self.w_pose * r_pose + self.w_vel * r_vel + self.w_ee * r_ee + self.w_root * r_root + self.alive - ctrl
        info = {"pose": float(r_pose.mean()), "vel": float(r_vel.mean()), "ee": float(r_ee.mean()),
                "root": float(r_root.mean()), "ctrl": float(ctrl.mean())}
        return RewardOut(total.astype(np.float32), info, self._failed(e))


class RewardV2(TrackingReward):
    """V1 + (2) foot-slip term, (3) forward-progress term (match the reference's heading-frame root
    planar velocity → forces locomotion), (4) tighter early termination. All terms are bounded
    exp(−α·error) rewards so the total stays positive. Self-contained so the recipe diffs against V1."""
    name = "v2"

    def __init__(self, nbody, *, pose_weight=0.5, vel_weight=0.05, ee_weight=0.2, root_weight=0.15,
                 alive_bonus=0.1, action_weight=0.002, progress_weight=0.2, slip_weight=0.1, **kw):
        kw.pop("term_pos_err", None)                       # (4) V2 bakes a tighter threshold; ignore the config value
        super().__init__(nbody, term_pos_err=0.3, **kw)
        self.w_pose, self.w_vel, self.w_ee, self.w_root = pose_weight, vel_weight, ee_weight, root_weight
        self.alive, self.action_weight = float(alive_bonus), float(action_weight)
        self.w_progress, self.w_slip = float(progress_weight), float(slip_weight)
        self.a_progress, self.a_slip = 5.0, 2.0

    def evaluate(self, sim, ref, actions) -> RewardOut:
        e = track_errors(sim, ref, self.ee, self.nbody)
        r_pose = np.exp(-self.a_pose * e.ori)
        r_vel = np.exp(-self.a_vel * e.vel)
        r_ee = np.exp(-self.a_ee * e.ee)
        r_root = np.exp(-self.a_root * e.root)
        # (3) forward progress: match the reference root's planar (x,z) velocity in the heading frame
        prog_err = np.sum((e.s_root_vel[:, [0, 2]] - e.r_root_vel[:, [0, 2]]) ** 2, axis=1)
        r_prog = np.exp(-self.a_progress * prog_err)
        # (2) foot-slip: 1 when grounded feet are still, → 0 when a grounded foot slides (bounded)
        r_slip = np.exp(-self.a_slip * foot_slip(sim, FEET))
        ctrl = self.action_weight * np.mean(np.asarray(actions) ** 2, axis=1)
        total = (self.w_pose * r_pose + self.w_vel * r_vel + self.w_ee * r_ee + self.w_root * r_root
                 + self.w_progress * r_prog + self.w_slip * r_slip + self.alive - ctrl)
        info = {"pose": float(r_pose.mean()), "vel": float(r_vel.mean()), "ee": float(r_ee.mean()),
                "root": float(r_root.mean()), "progress": float(r_prog.mean()),
                "slip": float(r_slip.mean()), "ctrl": float(ctrl.mean())}
        return RewardOut(total.astype(np.float32), info, self._failed(e))


REGISTRY: dict[str, type[TrackingReward]] = {"v1": RewardV1, "v2": RewardV2}


def make_reward(name: str, nbody: int, **kwargs) -> TrackingReward:
    if name not in REGISTRY:
        raise ValueError(f"unknown track reward {name!r} (have: {sorted(REGISTRY)})")
    return REGISTRY[name](nbody, **kwargs)
