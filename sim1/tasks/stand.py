"""StandTask — keep the physically-simulated humanoid upright (P1).

Observation (translation-invariant): root height, root orientation quat, root linear + angular
velocity, per-DOF joint q/qd, and per-body contact flags. Reward = alive bonus + torso uprightness
+ staying near the standing height − a small control penalty. The episode terminates on a fall
(root drops below a fraction of standing height, or the torso tips past an uprightness threshold),
so the alive bonus drives the policy to stay standing as long as possible.

Actuation: the policy outputs ~unit-scale values; `action_scale = max_torque` maps them onto the
engine's per-DOF torque range (the engine then clamps to ±max_torque). The standing reference
height is captured from the rig's authored pose at reset (rig-agnostic: works for 21-DOF or AMP).
"""

from __future__ import annotations

import numpy as np

from sim1.envs.vecenv import VecEnv


class StandTask:
    def __init__(
        self,
        ndof: int,
        nbody: int,
        act_dim: int,
        action_scale: float,
        upright_weight: float = 1.0,
        height_weight: float = 1.0,
        alive_bonus: float = 1.0,
        action_weight: float = 0.01,
        fall_height_frac: float = 0.5,
        upright_fall: float = 0.3,
    ):
        self.ndof = int(ndof)
        self.nbody = int(nbody)
        self.act_dim = int(act_dim)
        # obs: height(1) + quat(4) + linvel(3) + angvel(3) + q[ndof] + qd[ndof] + contacts[nbody]
        self.obs_dim = 1 + 4 + 3 + 3 + 2 * self.ndof + self.nbody
        # policy output (~unit) → env action. Torque mode: ≈max_torque. PD-target mode: ≈radians.
        self.action_scale = float(action_scale)

        self.upright_weight = float(upright_weight)
        self.height_weight = float(height_weight)
        self.alive_bonus = float(alive_bonus)
        self.action_weight = float(action_weight)
        self.fall_height_frac = float(fall_height_frac)
        self.upright_fall = float(upright_fall)
        self._target_h: np.ndarray | None = None  # standing height, captured at reset

    def reset(self, env: VecEnv, seed: int) -> None:
        self._target_h = env.root_pose[:, 1].copy()  # authored standing height per env

    def reset_masked(self, env: VecEnv, mask: np.ndarray, seed: int) -> None:
        if self._target_h is None:
            self.reset(env, seed)
        # reset is in-place to the same standing pose, so the target height is unchanged.

    @staticmethod
    def _uprightness(env: VecEnv) -> np.ndarray:
        # world up-component of the root's local +Y axis for quat (w, x, y, z): 1 - 2(x^2 + z^2).
        x = env.root_pose[:, 4]
        z = env.root_pose[:, 6]
        return 1.0 - 2.0 * (x * x + z * z)

    def observe(self, env: VecEnv) -> np.ndarray:
        return np.concatenate(
            [
                env.root_pose[:, 1:2],   # height
                env.root_pose[:, 3:7],   # quat wxyz
                env.root_twist[:, 0:3],  # linear velocity
                env.root_twist[:, 3:6],  # angular velocity
                env.joint_q,
                env.joint_qd,
                env.contact_flags,
            ],
            axis=1,
        ).astype(np.float32)

    def reward(self, env: VecEnv, actions: np.ndarray) -> np.ndarray:
        up = self._uprightness(env)
        h = env.root_pose[:, 1]
        height_term = np.exp(-10.0 * (h - self._target_h) ** 2)
        ctrl = np.mean(actions ** 2, axis=1)
        r = (
            self.alive_bonus
            + self.upright_weight * up
            + self.height_weight * height_term
            - self.action_weight * ctrl
        )
        return r.astype(np.float32)

    def done(self, env: VecEnv, ep_step: np.ndarray) -> np.ndarray:
        up = self._uprightness(env)
        h = env.root_pose[:, 1]
        fell = (h < self.fall_height_frac * self._target_h) | (up < self.upright_fall)
        return fell.astype(bool)
