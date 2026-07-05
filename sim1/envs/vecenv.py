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

    # Reference-state-init: set the whole articulation from per-body world states (num_envs, nbody, ·),
    # quats (w, x, y, z). Mirrors engine_py.VecEnv.set_articulation_state (reduced backend reconstructs
    # its generalized coords). Used to reset episodes to sampled reference frames (RSI).
    def set_articulation_state(self, pos: np.ndarray, quat: np.ndarray,
                               lin: np.ndarray, ang: np.ndarray) -> None: ...

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

    # --- per-body world-space state (for the SuperTrack per-body 6D representation) ---
    # Present on the engine backend + mock. Shapes are (num_envs, nbody, ·); quats are (w, x, y, z).
    @property
    def body_pos(self) -> np.ndarray: ...       # (num_envs, nbody, 3)  world position
    @property
    def body_quat(self) -> np.ndarray: ...      # (num_envs, nbody, 4)  world orientation (wxyz)
    @property
    def body_linvel(self) -> np.ndarray: ...    # (num_envs, nbody, 3)  world linear velocity
    @property
    def body_angvel(self) -> np.ndarray: ...    # (num_envs, nbody, 3)  world angular velocity

    # --- observation composition (single source; engine = C++ via binding, mock = Python oracle) ---
    # Compose the transfer-stable proprioception block [height | root_rot | linvel | angvel | q | qd |
    # contacts] with the given rotation ("quat"|"sixd") + frame ("world"|"local"). The caller appends
    # command channels; compose_body() appends the per-body 6D block.
    def compose_proprio(self, rotation: str, frame: str) -> np.ndarray: ...   # (num_envs, proprio_dim)
    def compose_body(self) -> np.ndarray: ...                                 # (num_envs, per_body_dim)
