"""Task interface.

A Task turns raw VecEnv state into the RL problem: it composes the observation, computes the
reward, decides failure-termination, and samples per-env task parameters on reset (the only
task-specific randomization seam). Time-limit truncation is handled by the `TaskEnv` adapter, not
here. This is exactly the layer the C++ engine deliberately omits.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from sim1.envs.vecenv import VecEnv


class Task(Protocol):
    obs_dim: int
    act_dim: int

    def reset(self, env: VecEnv, seed: int) -> None:
        """Sample per-env task parameters (e.g. targets) for all envs."""

    def reset_masked(self, env: VecEnv, mask: np.ndarray, seed: int) -> None:
        """Resample task parameters for the masked (done) envs only."""

    def observe(self, env: VecEnv) -> np.ndarray:
        """Compose the observation batch, shape (num_envs, obs_dim)."""

    def reward(self, env: VecEnv, actions: np.ndarray) -> np.ndarray:
        """Per-env reward from the post-step state, shape (num_envs,)."""

    def done(self, env: VecEnv, ep_step: np.ndarray) -> np.ndarray:
        """Per-env failure termination (bool), shape (num_envs,). Time limit is separate."""
