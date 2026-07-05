"""TrackingTask — DeepMimic-style motion imitation on the composable base.

Observation = proprioception + a **phase clock** (sin/cos of the clip phase); the policy learns *where
in the cycle* it is and the **imitation reward** drives it toward the reference. Episodes start from
**reference-state-init** (a random reference frame set into the sim via `env.set_articulation_state`).
The reward itself is a pluggable, named variant from `track_rewards.py` (`reward_name`, recorded in
the run config) — so we can iterate on / A-B different reward recipes without touching this task.

Phase note: `reward()` advances the per-env reference time by `control_dt` (it is called exactly once
per env step, right after `env.step()`), keeping the phase obs and the reward's reference in sync.
"""

from __future__ import annotations

import numpy as np

from sim1.envs.vecenv import VecEnv
from sim1.motion.motion_lib import AmpMotionLib, ReferenceMotion
from sim1.tasks.composite import CompositeTask
from sim1.tasks.track_rewards import BodyState, HANDS_AND_FEET, make_reward


class PhaseObs:
    """ObsComponent: (sin, cos) of the clip phase ∈ [0, 2π) — the DeepMimic phase clock."""

    dim = 2

    def observe(self, env: VecEnv, task) -> np.ndarray:
        ph = (task._ref_time / task.motion.duration) * (2.0 * np.pi)
        return np.stack([np.sin(ph), np.cos(ph)], axis=1).astype(np.float32)


class TrackingTask(CompositeTask):
    def __init__(
        self,
        ndof: int,
        nbody: int,
        act_dim: int,
        action_scale: float,
        motion: ReferenceMotion,
        control_dt: float,
        *,
        reward_name: str = "v2",
        pose_weight: float = 0.5,
        vel_weight: float = 0.05,
        ee_weight: float = 0.2,
        root_weight: float = 0.15,
        alive_bonus: float = 0.1,
        action_weight: float = 0.002,
        term_pos_err: float = 0.5,
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
        self.rsi = bool(rsi)
        # The reward recipe (weights / extra terms / termination) lives in the selected variant.
        self.reward_name = reward_name
        self._reward = make_reward(
            reward_name, nbody, ee=HANDS_AND_FEET,
            pose_weight=pose_weight, vel_weight=vel_weight, ee_weight=ee_weight, root_weight=root_weight,
            alive_bonus=alive_bonus, action_weight=action_weight,
            term_pos_err=term_pos_err, fall_height_frac=fall_height_frac,
        )
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

    # --- reward (advances the phase) + termination — delegated to the selected variant ---
    def reward(self, env: VecEnv, actions: np.ndarray) -> np.ndarray:
        self._ref_time = np.mod(self._ref_time + self.control_dt, self.motion.duration)
        ref = self._lib.state_at(self._ref_time)
        sim = BodyState(np.asarray(env.body_pos), np.asarray(env.body_quat),
                        np.asarray(env.body_linvel), np.asarray(env.body_angvel))
        out = self._reward.evaluate(sim, ref, np.asarray(actions))
        self.reward_info = out.info      # per-term means (consumed by the eval-side breakdown)
        self._failed = out.failed
        return out.total

    def done(self, env: VecEnv, ep_step: np.ndarray) -> np.ndarray:
        return self._failed if self._failed is not None else np.zeros(env.num_envs, dtype=bool)
