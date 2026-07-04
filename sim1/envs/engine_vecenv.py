"""EngineVecEnv — the real backend: adapts `engine_py.VecEnv` to the `VecEnv` contract (P1).

Wraps the nanobind binding (built from `csrc/`) and presents the exact same surface as `MockVecEnv`
(`sim1/envs/vecenv.py`), so the Task/PPO stack runs unchanged. The binding hands us two zero-copy
NumPy views — a writable `(N, act_dim)` action buffer and a read-only `(N, obs_dim)` observation
buffer packed as [pos3 | quat_wxyz4 | linvel3 | angvel3 | q[ndof] | qd[ndof] | contacts[nbody]] — and
we slice the contract's named fields out of the obs view here (obs composition is the trainer's job).

The Python `EnvConfig` maps onto the engine's centralized `SimConfig` (the configs we added), keeping
one source of truth for substeps / control_dt / gains / torque clamp / friction / backend / gravity.
"""

from __future__ import annotations

import numpy as np


def _build_sim_config(cfg):
    """Map sim1.config.EnvConfig → engine_py.SimConfig (the engine's centralized tuning config)."""
    from sim1 import engine_py

    sim = engine_py.SimConfig()
    sim.substeps = int(cfg.substeps)
    sim.control_dt = float(cfg.control_dt)
    sim.max_torque = float(cfg.max_torque)
    sim.kp = float(cfg.kp)
    sim.kd = float(cfg.kd)
    sim.ground_friction = float(cfg.ground_friction)
    sim.action_mode = (engine_py.ActionMode.PDTarget if cfg.action_mode == "pd_target"
                       else engine_py.ActionMode.Torque)
    sim.backend = (engine_py.Backend.Reduced if getattr(cfg, "backend", "reduced") == "reduced"
                   else engine_py.Backend.Realtime)
    return sim


def _build_articulation(cfg):
    from sim1 import engine_py
    model = getattr(cfg, "model", "humanoid")
    return engine_py.make_amp_humanoid() if model == "amp" else engine_py.make_humanoid()


class EngineVecEnv:
    """Contract-compatible wrapper around the C++ `engine_py.VecEnv` (drop-in for MockVecEnv)."""

    def __init__(self, cfg, seed: int = 0):
        from sim1 import engine_py

        env_config = engine_py.EnvConfig(_build_articulation(cfg), _build_sim_config(cfg))
        self._env = engine_py.VecEnv(int(cfg.num_envs), env_config, threads=int(getattr(cfg, "threads", 0)))
        self.num_envs = self._env.num_envs
        self.act_dim = self._env.act_dim
        self.ndof = self._env.ndof
        self.nbody = self._env.nbody
        self._env.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self._env.reset(seed)

    def reset_masked(self, mask: np.ndarray, seed: int = 0) -> None:
        self._env.reset_masked(np.asarray(mask).astype(np.uint8), seed)

    def step(self) -> None:
        self._env.step()

    # --- writable action buffer (zero-copy view into the C++ batch) ---
    @property
    def actions(self) -> np.ndarray:
        return self._env.actions()

    # --- raw batched state, sliced from the zero-copy obs view (layout above) ---
    @property
    def _obs(self) -> np.ndarray:
        return self._env.observations()

    @property
    def root_pose(self) -> np.ndarray:      # (N, 7) pos(3) + quat wxyz(4)
        return self._obs[:, 0:7]

    @property
    def root_twist(self) -> np.ndarray:     # (N, 6) linvel(3) + angvel(3)
        return self._obs[:, 7:13]

    @property
    def joint_q(self) -> np.ndarray:        # (N, ndof)
        return self._obs[:, 13:13 + self.ndof]

    @property
    def joint_qd(self) -> np.ndarray:       # (N, ndof)
        o = 13 + self.ndof
        return self._obs[:, o:o + self.ndof]

    @property
    def contact_flags(self) -> np.ndarray:  # (N, nbody)
        return self._obs[:, 13 + 2 * self.ndof:]


def make_vecenv(cfg, seed: int = 0):
    """Factory: return the mock or the real engine-backed VecEnv per `cfg.kind`."""
    if cfg.kind == "engine":
        return EngineVecEnv(cfg, seed=seed)
    from sim1.envs.mock_vecenv import MockVecEnv
    return MockVecEnv(num_envs=cfg.num_envs, ndof=cfg.ndof, dt=cfg.dt,
                      damping=cfg.damping, action_scale=cfg.action_scale, seed=seed)
