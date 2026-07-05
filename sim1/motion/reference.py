"""Motion / mocap seam — interfaces for the imitation phase (NOT yet implemented).

The composable task system already has the two integration points imitation needs, so adding mocap
does **not** change the trainer, the obs plumbing, or the eval/export path — it's new components:

  1. **Reference observation** — an `ObsComponent` (see `composite.py`) that appends reference-pose
     and/or phase features to the observation (for tracking policies that condition on "where the
     reference is now/next"). Appends after the command channels, preserving transfer.
  2. **Imitation reward** — a `RewardTerm` scoring similarity to the reference: either explicit
     **tracking error** (DeepMimic / SuperTrack — pose/vel/rot gaps) or an **AMP discriminator**
     `-log(1 - D(s, s'))` (adversarial style prior). Slots into `reward_terms` like any other term.

Main work item when we get here: a **MotionLib** that loads the ASE clips
(`humanoid-motion/ASE/ase/data/motions/*.npy`, SkeletonMotion format) and **retargets** them to our
engine's AMP rig (joint order / DOF / frame conventions), exposing per-frame reference states we can
compare against the sim. Retargeting is the bulk of the effort; the RL wiring is small given the
seams above.

This module currently defines only the Protocols (the contract), so callers/design can reference
them; the implementations land with the imitation milestone.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from sim1.envs.vecenv import VecEnv


class MotionLib(Protocol):
    """Loads + serves reference motion clips retargeted to our rig."""

    def load(self, path: str) -> None: ...
    def num_clips(self) -> int: ...
    def sample_frames(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n (clip, phase) frames → reference states for RSI / tracking targets."""
    def state_at(self, clip: np.ndarray, phase: np.ndarray) -> dict:
        """Reference root pose/twist + joint q/qd at (clip, phase), aligned to our rig layout."""


class ReferenceObs(Protocol):
    """An `ObsComponent`: appends reference/phase features to the observation."""

    dim: int
    def observe(self, env: VecEnv, task) -> np.ndarray: ...


class ImitationReward(Protocol):
    """A reward source scoring similarity to the reference (tracking error or AMP discriminator).
    Used to build a `RewardTerm`."""

    def reward(self, env: VecEnv, task) -> np.ndarray: ...
