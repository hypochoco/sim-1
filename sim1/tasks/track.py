"""TrackingTask — DeepMimic-style motion imitation on the composable base.

Observation = proprioception + a **phase clock** (sin/cos of the clip phase); the policy learns *where
in the cycle* it is and the **imitation reward** drives it toward the reference. Reward = weighted
sum of exp(−α·error) over pose (per-body orientation), velocity, end-effectors, and the root — all
**root-relative / velocity-based** so a translating locomotion clip loops seamlessly (no absolute
x,z tracking). Episodes start from **reference-state-init** (a random reference frame set into the
sim via `env.set_articulation_state`) and **terminate early** when tracking diverges (RMS body error)
or the character falls. Reference is retargeted to our rig (see `motion_lib.retarget_to_rig`).

Phase note: `reward()` advances the per-env reference time by `control_dt` (it is called exactly once
per env step, right after `env.step()`), keeping the phase obs and the reward's reference in sync.
"""

from __future__ import annotations

import numpy as np

from sim1.envs.vecenv import VecEnv
from sim1.motion.motion_lib import AmpMotionLib, ReferenceMotion
from sim1.tasks.composite import CompositeTask
from sim1.tasks.proprio import _quat_mul, _yaw_from_quat


def _quat_geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) between batched wxyz quats a,b (..., 4) → (...)."""
    d = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(d, 0.0, 1.0))


def _heading_frame(bp: np.ndarray, bq: np.ndarray, bl: np.ndarray, ba: np.ndarray):
    """Express per-body state in the ROOT's heading frame (global yaw removed) → direction-invariant.
    Positions are root-relative; the whole set is rotated by −(root yaw) about +Y and body orientations
    are left-multiplied by the inverse root-heading. Returns (rel_pos, quat, linvel, angvel), all
    yaw-canceled, so a global heading change leaves them unchanged."""
    bp = np.asarray(bp, np.float32); bq = np.asarray(bq, np.float32)
    bl = np.asarray(bl, np.float32); ba = np.asarray(ba, np.float32)
    n, b = bp.shape[0], bp.shape[1]
    yaw = _yaw_from_quat(bq[:, 0])                         # (n,) root heading from projected forward axis
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]

    def rot_xz(v):                                        # rotate (x, z) by −yaw about +Y, keep y
        x, z = v[..., 0], v[..., 2]
        return np.stack([c * x - s * z, v[..., 1], s * x + c * z], axis=-1).astype(np.float32)

    rel = rot_xz(bp - bp[:, :1, :])
    lv, av = rot_xz(bl), rot_xz(ba)
    half = 0.5 * yaw
    qhi = np.stack([np.cos(half), np.zeros_like(half), -np.sin(half), np.zeros_like(half)], axis=1).astype(np.float32)
    q = _quat_mul(np.repeat(qhi, b, axis=0), bq.reshape(n * b, 4)).reshape(n, b, 4)
    return rel, q, lv, av


class PhaseObs:
    """ObsComponent: (sin, cos) of the clip phase ∈ [0, 2π) — the DeepMimic phase clock."""

    dim = 2

    def observe(self, env: VecEnv, task) -> np.ndarray:
        ph = (task._ref_time / task.motion.duration) * (2.0 * np.pi)
        return np.stack([np.sin(ph), np.cos(ph)], axis=1).astype(np.float32)


class TrackingTask(CompositeTask):
    # end-effectors in our (feet-last) body order: hands + feet
    EE = (5, 8, 13, 14)

    def __init__(
        self,
        ndof: int,
        nbody: int,
        act_dim: int,
        action_scale: float,
        motion: ReferenceMotion,
        control_dt: float,
        *,
        pose_weight: float = 0.5,
        vel_weight: float = 0.05,
        ee_weight: float = 0.2,
        root_weight: float = 0.15,
        alive_bonus: float = 0.1,
        action_weight: float = 0.002,   # small control penalty (discourage jitter/high torque)
        term_pos_err: float = 0.5,   # RMS root-relative body position error (m) that ends an episode
        fall_height_frac: float = 0.5,
        rsi: bool = True,
        rot_repr: str = "quat",
        frame: str = "world",
    ):
        super().__init__(
            ndof, nbody, act_dim, action_scale,
            reward_terms=[], command_weight=0.0,
            extra_obs=[PhaseObs()], rot_repr=rot_repr, frame=frame,
        )
        self._lib = AmpMotionLib().add(motion)
        self.motion = motion
        self.control_dt = float(control_dt)
        self.ee = np.array(self.EE, dtype=np.int64)
        self.w_pose, self.w_vel, self.w_ee, self.w_root = pose_weight, vel_weight, ee_weight, root_weight
        self.alive = float(alive_bonus)
        self.action_weight = float(action_weight)
        # error sharpness (DeepMimic-inspired)
        self.a_pose, self.a_vel, self.a_ee, self.a_root = 2.0, 0.1, 40.0, 10.0
        self.term_pos_err = float(term_pos_err)
        self.fall_height_frac = float(fall_height_frac)
        self.rsi = bool(rsi)
        self._ref_time: np.ndarray | None = None
        self._start_time: np.ndarray | None = None
        self._failed: np.ndarray | None = None

    # --- RSI ---
    def _apply_rsi(self, env: VecEnv, mask: np.ndarray) -> None:
        idx = np.where(np.asarray(mask, dtype=bool))[0]
        if idx.size == 0:
            return
        ref = self._lib.state_at(self._ref_time[idx])
        pos = np.asarray(env.body_pos).copy()
        quat = np.asarray(env.body_quat).copy()
        lin = np.asarray(env.body_linvel).copy()
        ang = np.asarray(env.body_angvel).copy()
        pos[idx], quat[idx], lin[idx], ang[idx] = ref.body_pos, ref.body_quat, ref.body_linvel, ref.body_angvel
        env.set_articulation_state(pos, quat, lin, ang)

    def _sample_times(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 1)
        return rng.uniform(0.0, self.motion.duration, size=n) if self.rsi else np.zeros(n)

    def reset(self, env: VecEnv, seed: int) -> None:
        super().reset(env, seed)
        n = env.num_envs
        self._start_time = self._sample_times(n, seed)
        self._ref_time = self._start_time.copy()
        self._failed = np.zeros(n, dtype=bool)
        self._apply_rsi(env, np.ones(n, dtype=bool))

    def reset_masked(self, env: VecEnv, mask: np.ndarray, seed: int) -> None:
        super().reset_masked(env, mask, seed)
        m = np.asarray(mask, dtype=bool)
        k = int(m.sum())
        if k:
            self._start_time[m] = self._sample_times(k, seed)
            self._ref_time[m] = self._start_time[m]
            self._failed[m] = False
            self._apply_rsi(env, m)

    # --- reward (advances the phase) + termination ---
    def reward(self, env: VecEnv, actions: np.ndarray) -> np.ndarray:
        self._ref_time = np.mod(self._ref_time + self.control_dt, self.motion.duration)
        ref = self._lib.state_at(self._ref_time)

        # Direction-invariant: compare sim vs reference in each pose's OWN heading frame (global yaw
        # removed), so the character is rewarded for the motion regardless of which way it faces.
        s_rel, s_q, s_lv, s_av = _heading_frame(env.body_pos, env.body_quat, env.body_linvel, env.body_angvel)
        r_rel, r_q, r_lv, r_av = _heading_frame(ref.body_pos, ref.body_quat, ref.body_linvel, ref.body_angvel)
        root_h_sim = np.asarray(env.body_pos)[:, 0, 1]
        root_h_ref = ref.body_pos[:, 0, 1]

        ori_err = np.sum(_quat_geodesic(s_q[:, 1:], r_q[:, 1:]) ** 2, axis=1)   # non-root bodies (root → root term)
        pos_err = np.sum((s_rel - r_rel) ** 2, axis=(1, 2))
        vel_err = np.sum((s_lv - r_lv) ** 2, axis=(1, 2)) + 0.1 * np.sum((s_av - r_av) ** 2, axis=(1, 2))
        ee_err = np.sum((s_rel[:, self.ee] - r_rel[:, self.ee]) ** 2, axis=(1, 2))
        root_err = ((root_h_sim - root_h_ref) ** 2                             # height (yaw-invariant)
                    + _quat_geodesic(s_q[:, 0], r_q[:, 0]) ** 2                 # tilt (yaw removed)
                    + 0.1 * np.sum((s_lv[:, 0] - r_lv[:, 0]) ** 2, axis=1))     # speed in heading frame

        r_pose = np.exp(-self.a_pose * ori_err)
        r_vel = np.exp(-self.a_vel * vel_err)
        r_ee = np.exp(-self.a_ee * ee_err)
        r_root = np.exp(-self.a_root * root_err)
        ctrl = self.action_weight * np.mean(np.asarray(actions) ** 2, axis=1)
        total = (self.w_pose * r_pose + self.w_vel * r_vel + self.w_ee * r_ee
                 + self.w_root * r_root + self.alive - ctrl)
        self.reward_info = {"pose": float(r_pose.mean()), "vel": float(r_vel.mean()),
                            "ee": float(r_ee.mean()), "root": float(r_root.mean()),
                            "ctrl": float(ctrl.mean())}

        rms = np.sqrt(pos_err / self.nbody)
        fell = root_h_sim < self.fall_height_frac * root_h_ref
        self._failed = (rms > self.term_pos_err) | fell
        return total.astype(np.float32)

    def done(self, env: VecEnv, ep_step: np.ndarray) -> np.ndarray:
        return self._failed if self._failed is not None else np.zeros(env.num_envs, dtype=bool)
