"""The VecEnv binding contract.

This mirrors the C++ `engine::physics_env::VecEnv` surface (see the training-integration review).
The engine binding (`engine_py.VecEnv`) and the P0 `MockVecEnv` both satisfy this Protocol, so the
task/algorithm code is written once and runs against either. Key idea from the C++ design: actions
are a *writable buffer* — you write into `actions`, then call `step()`; state is read back from the
raw-state accessors (SoA, zero-copy in the real binding).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VecEnv(Protocol):
    num_envs: int
    act_dim: int

    def reset(self, seed: int) -> None: ...
    def reset_masked(self, mask: np.ndarray, seed: int) -> None: ...
    def step(self) -> None: ...

    # Writable action buffer, shape (num_envs, act_dim).
    @property
    def actions(self) -> np.ndarray: ...

    # --- raw batched state (compose observations from these) ---
    @property
    def joint_q(self) -> np.ndarray: ...        # (num_envs, ndof)
    @property
    def joint_qd(self) -> np.ndarray: ...       # (num_envs, ndof)
    @property
    def root_pose(self) -> np.ndarray: ...      # (num_envs, 7)  pos(3) + quat wxyz(4)
    @property
    def root_twist(self) -> np.ndarray: ...     # (num_envs, 6)  linvel(3) + angvel(3)
    @property
    def contact_flags(self) -> np.ndarray: ...  # (num_envs, nbody)
