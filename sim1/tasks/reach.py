"""ReachTask — a simple, demonstrably-learnable task for the mock env (P0).

Each env has a fixed per-episode target in configuration space. Observation is the target-relative
position and the velocity; reward drives the position to the target with a small action penalty.
This validates the whole pipeline (rollout → GAE → PPO update → logging → checkpoint) end to end.
"""

from __future__ import annotations

import numpy as np

from sim1.envs.vecenv import VecEnv


class ReachTask:
    def __init__(
        self,
        ndof: int,
        pos_weight: float = 1.0,
        vel_weight: float = 0.0,
        action_weight: float = 0.01,
        target_scale: float = 1.0,
    ):
        self.ndof = int(ndof)
        self.act_dim = int(ndof)
        self.obs_dim = 2 * int(ndof)  # (q - target) ++ qd
        self.pos_weight = float(pos_weight)
        self.vel_weight = float(vel_weight)
        self.action_weight = float(action_weight)
        self.target_scale = float(target_scale)
        self._targets: np.ndarray | None = None

    def reset(self, env: VecEnv, seed: int) -> None:
        self._targets = np.zeros((env.num_envs, self.ndof), dtype=np.float32)
        self._sample(np.arange(env.num_envs), seed)

    def reset_masked(self, env: VecEnv, mask: np.ndarray, seed: int) -> None:
        if self._targets is None:
            self.reset(env, seed)
            return
        idx = np.nonzero(np.asarray(mask).astype(bool))[0]
        self._sample(idx, seed)

    def _sample(self, idx: np.ndarray, seed: int) -> None:
        for i in idx:
            rng = np.random.default_rng(seed + int(i))
            self._targets[i] = rng.standard_normal(self.ndof).astype(np.float32) * self.target_scale

    def observe(self, env: VecEnv) -> np.ndarray:
        return np.concatenate([env.joint_q - self._targets, env.joint_qd], axis=1).astype(np.float32)

    def reward(self, env: VecEnv, actions: np.ndarray) -> np.ndarray:
        pos_err = np.sum((env.joint_q - self._targets) ** 2, axis=1)
        vel_pen = np.sum(env.joint_qd ** 2, axis=1)
        act_pen = np.sum(actions ** 2, axis=1)
        r = -(self.pos_weight * pos_err + self.vel_weight * vel_pen + self.action_weight * act_pen)
        return r.astype(np.float32)

    def done(self, env: VecEnv, ep_step: np.ndarray) -> np.ndarray:
        return np.zeros((env.num_envs,), dtype=bool)  # no failure state; time-limit only
