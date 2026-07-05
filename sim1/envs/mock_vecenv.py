"""MockVecEnv — a contract-compatible stand-in for the engine binding (P0).

N independent point-masses in R^ndof. Honors the `VecEnv` contract exactly (write `actions`, call
`step()`, read raw state), so the Task/PPO stack developed here runs unchanged against the real
`engine_py.VecEnv` in P1. Dynamics are a damped double-integrator:

    qd += (clip(action, -1, 1) * action_scale - damping * qd) * dt
    q  += qd * dt

Deterministic given a seed: `reset(seed)` zeroes state (targets are owned by the Task, not the env).
The raw-state accessors present the same shapes the humanoid binding will (root pose/twist/contacts
are trivial here). `ndof` maps onto `joint_q`/`joint_qd`; `act_dim == ndof`.
"""

from __future__ import annotations

import numpy as np


class MockVecEnv:
    def __init__(
        self,
        num_envs: int,
        ndof: int = 2,
        dt: float = 0.05,
        damping: float = 0.1,
        action_scale: float = 1.0,
        seed: int = 0,
    ):
        self.num_envs = int(num_envs)
        self.ndof = int(ndof)
        self.act_dim = int(ndof)
        self.dt = float(dt)
        self.damping = float(damping)
        self.action_scale = float(action_scale)
        self._nbody = 1

        self._q = np.zeros((self.num_envs, self.ndof), dtype=np.float32)
        self._qd = np.zeros((self.num_envs, self.ndof), dtype=np.float32)
        self._actions = np.zeros((self.num_envs, self.ndof), dtype=np.float32)
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self._q[:] = 0.0
        self._qd[:] = 0.0
        self._actions[:] = 0.0

    def reset_masked(self, mask: np.ndarray, seed: int = 0) -> None:
        m = np.asarray(mask).astype(bool)
        self._q[m] = 0.0
        self._qd[m] = 0.0
        self._actions[m] = 0.0

    def step(self) -> None:
        a = np.clip(self._actions, -1.0, 1.0) * self.action_scale
        self._qd += (a - self.damping * self._qd) * self.dt
        self._q += self._qd * self.dt

    def set_articulation_state(self, pos: np.ndarray, quat: np.ndarray,
                               lin: np.ndarray, ang: np.ndarray) -> None:
        # The mock is a reduced-order toy (root only); RSI/tracking needs the full engine rig.
        raise NotImplementedError("set_articulation_state (RSI/tracking) requires the engine backend")

    # --- contract: writable action buffer ---
    @property
    def actions(self) -> np.ndarray:
        return self._actions

    # --- contract: raw batched state ---
    @property
    def joint_q(self) -> np.ndarray:
        return self._q

    @property
    def joint_qd(self) -> np.ndarray:
        return self._qd

    @property
    def root_pose(self) -> np.ndarray:
        rp = np.zeros((self.num_envs, 7), dtype=np.float32)
        rp[:, 3] = 1.0  # identity quaternion (w, x, y, z)
        return rp

    @property
    def root_twist(self) -> np.ndarray:
        return np.zeros((self.num_envs, 6), dtype=np.float32)

    @property
    def contact_flags(self) -> np.ndarray:
        return np.zeros((self.num_envs, self._nbody), dtype=np.float32)

    # --- per-body world-space state (mock: one body at the root, identity orientation, at rest) ---
    @property
    def body_pos(self) -> np.ndarray:
        return np.zeros((self.num_envs, self._nbody, 3), dtype=np.float32)

    @property
    def body_quat(self) -> np.ndarray:
        q = np.zeros((self.num_envs, self._nbody, 4), dtype=np.float32)
        q[:, :, 0] = 1.0  # identity (w, x, y, z)
        return q

    @property
    def body_linvel(self) -> np.ndarray:
        return np.zeros((self.num_envs, self._nbody, 3), dtype=np.float32)

    @property
    def body_angvel(self) -> np.ndarray:
        return np.zeros((self.num_envs, self._nbody, 3), dtype=np.float32)

    # --- observation composition (Python oracle; the engine backend uses the C++ binding instead) ---
    def compose_proprio(self, rotation: str, frame: str) -> np.ndarray:
        from sim1.tasks.proprio import proprio_obs
        return proprio_obs(self, rotation, frame)

    def compose_body(self) -> np.ndarray:
        from sim1.tasks.proprio import per_body_obs
        return per_body_obs(self, "sixd")
