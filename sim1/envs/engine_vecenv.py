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
    """Contract-compatible wrapper around the C++ engine envs (drop-in for MockVecEnv).

    `cfg.kind` selects the C++ backend:
      • "engine"   → engine_py.VecEnv     (reduced/realtime PGS PhysicsWorld; full per-body + RSI)
      • "diff-cpu" → engine_py.DiffVecEnv  (CPU diff ABA + smoothed contact; flat obs only)
      • "cuda"     → engine_py.CudaVecEnv  (GPU diff ABA, same physics; flat obs only)
    The diff/cuda backends are walk-scoped: they expose the flat obs + proprio (all walk needs) but
    NOT per-body world state / RSI (deferred with getup/track) — those raise NotImplementedError.
    """

    _ENGINE_CLASS = {"engine": "VecEnv", "diff-cpu": "DiffVecEnv", "cuda": "CudaVecEnv"}

    def __init__(self, cfg, seed: int = 0):
        from sim1 import engine_py

        kind = getattr(cfg, "kind", "engine")
        cls_name = self._ENGINE_CLASS.get(kind, "VecEnv")
        if cls_name == "CudaVecEnv" and not getattr(engine_py, "HAS_CUDA", False):
            raise RuntimeError(
                "env.kind='cuda' but engine_py was built without CUDA. Rebuild on an NVIDIA box:\n"
                "  CUDA_HOME=... CMAKE_PREFIX_PATH=.deps pip install -e . --no-build-isolation "
                "--config-settings=cmake.define.SIM1_CUDA=ON")
        self._flat_only = kind in ("diff-cpu", "cuda")   # no per-body world state / RSI (walk-scoped)
        EnvCls = getattr(engine_py, cls_name)

        env_config = engine_py.EnvConfig(_build_articulation(cfg), _build_sim_config(cfg))
        self._env = EnvCls(int(cfg.num_envs), env_config, threads=int(getattr(cfg, "threads", 0)))
        self.num_envs = self._env.num_envs
        self.act_dim = self._env.act_dim
        self.ndof = self._env.ndof
        self.nbody = self._env.nbody
        self._env.reset(seed)

    def _require_full(self, what: str) -> None:
        if self._flat_only:
            raise NotImplementedError(
                f"{what} is not available on the diff/cuda backend (walk-scoped): it needs per-body "
                "world state / RSI, deferred with getup/track. Use kind='engine' for those tasks.")

    def _body_readout(self, what: str) -> np.ndarray:
        """Per-body world state. Available on `engine` (PGS world) and `diff-cpu` (via linkWorld FK,
        rendering readout). NOT on `cuda` (state stays on-device — deferred to viz Option A)."""
        fn = getattr(self._env, what, None)
        if fn is None:
            raise NotImplementedError(
                f"{what}: no per-body world readout on this backend (cuda keeps state on-device — "
                "deferred to viz Option A). Use kind='engine' or 'diff-cpu'.")
        return fn()

    def reset(self, seed: int = 0) -> None:
        self._env.reset(seed)

    def reset_masked(self, mask: np.ndarray, seed: int = 0) -> None:
        self._env.reset_masked(np.asarray(mask).astype(np.uint8), seed)

    def set_articulation_state(self, pos: np.ndarray, quat: np.ndarray,
                               lin: np.ndarray, ang: np.ndarray) -> None:
        """RSI: set every env's articulation from per-body world states (N, nbody, ·), quats wxyz."""
        self._require_full("set_articulation_state (RSI)")
        self._env.set_articulation_state(
            np.ascontiguousarray(pos, dtype=np.float32), np.ascontiguousarray(quat, dtype=np.float32),
            np.ascontiguousarray(lin, dtype=np.float32), np.ascontiguousarray(ang, dtype=np.float32))

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

    # --- per-body world-space state (zero-copy 3D views; refreshed on reset/step) ---
    @property
    def body_pos(self) -> np.ndarray:       # (N, nbody, 3)
        return self._body_readout("body_pos")

    @property
    def body_quat(self) -> np.ndarray:      # (N, nbody, 4) wxyz
        return self._body_readout("body_quat")

    @property
    def body_linvel(self) -> np.ndarray:    # (N, nbody, 3)
        return self._body_readout("body_linvel")

    @property
    def body_angvel(self) -> np.ndarray:    # (N, nbody, 3)
        return self._body_readout("body_angvel")

    # --- observation composition (single C++ source via the binding) ---
    def compose_proprio(self, rotation: str, frame: str) -> np.ndarray:
        return self._env.proprio(rotation, frame)

    def compose_body(self) -> np.ndarray:
        self._require_full("compose_body (per-body 6D block)")
        return self._env.body_block()


def make_vecenv(cfg, seed: int = 0):
    """Factory: return the mock or an engine-backed VecEnv per `cfg.kind`.

    kinds: "mock" (P0) | "engine" (reduced/realtime PGS) | "diff-cpu" (CPU diff ABA) |
    "cuda" (GPU diff ABA). The diff/cuda backends need substeps >= 32 for float/stiff-PD stability
    (the reduced backend's default 8 diverges) — warn if under-set.
    """
    if cfg.kind in ("engine", "diff-cpu", "cuda"):
        if cfg.kind in ("diff-cpu", "cuda") and int(getattr(cfg, "substeps", 0)) < 32:
            import warnings
            warnings.warn(
                f"env.kind='{cfg.kind}' with substeps={cfg.substeps} < 32: the diff ABA + smoothed "
                "contact under stiff PD is unstable below ~32 substeps (NaNs). Set env.substeps>=32.",
                stacklevel=2)
        return EngineVecEnv(cfg, seed=seed)
    from sim1.envs.mock_vecenv import MockVecEnv
    return MockVecEnv(num_envs=cfg.num_envs, ndof=cfg.ndof, dt=cfg.dt,
                      damping=cfg.damping, action_scale=cfg.action_scale, seed=seed)
