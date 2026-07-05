"""Commands — the goal / control-input abstraction that makes tasks goal-conditioned and, at
inference, user-controllable.

A `Command` owns a per-env goal vector. The SAME object serves three roles behind one interface:
  1. **training** — `sample(n, rng)` draws random goals (goal-conditioned RL over the goal space);
  2. **observation** — `to_obs(goal)` are the channels appended to the proprioception block, so the
     policy is conditioned on the goal;
  3. **reward** — `reward(env, goal)` scores goal achievement.
At inference, drive the same channels from user input instead of `sample` → the policy is directly
**user-controllable** (e.g. arrow keys → walk direction). Adding a new control input later (combat
target, stance, weapon) = a new `Command` (or composing several) — no trainer/obs-plumbing changes.

Composition: `CompositeCommand([...])` concatenates several commands (goals, obs channels, rewards),
which is how a combat controller layers e.g. "move here" + "face target" + "attack id".
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from sim1.envs.vecenv import VecEnv
from sim1.tasks.proprio import planar_velocity, rotate_to_heading, root_yaw


class Command(Protocol):
    dim: int  # number of observation channels this command contributes

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n goals, shape (n, goal_dim). goal_dim need not equal `dim`."""

    def to_obs(self, env: VecEnv, goal: np.ndarray) -> np.ndarray:
        """Goal → observation channels (may depend on env state, e.g. root-relative), shape (n, dim)."""

    def reward(self, env: VecEnv, goal: np.ndarray) -> np.ndarray:
        """Goal-achievement reward in [0, 1]-ish, shape (n,)."""


class NoCommand:
    """No goal (stand / getup). Contributes zero obs channels and zero reward."""

    dim = 0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.zeros((n, 0), dtype=np.float32)

    def to_obs(self, env: VecEnv, goal: np.ndarray) -> np.ndarray:
        return np.zeros((goal.shape[0], 0), dtype=np.float32)

    def reward(self, env: VecEnv, goal: np.ndarray) -> np.ndarray:
        return np.zeros(goal.shape[0], dtype=np.float32)


class HeadingSpeedCommand:
    """Walk/steer: a target planar velocity. The goal is stored world-frame (vx, vz); the two obs
    channels are that target expressed in the character's **root-local (heading) frame** by default
    (`frame="local"`) — so the policy learns one egocentric "walk in my commanded direction" skill
    that generalizes over world heading, and user control maps naturally (forward/strafe). Reward is
    frame-agnostic (velocity-gap magnitude), so it stays in world frame. Set `frame="world"` for the
    simpler absolute-direction variant."""

    dim = 2

    def __init__(self, speed_range: tuple[float, float] = (0.5, 1.5), track_scale: float = 1.0,
                 frame: str = "local"):
        self.speed_range = (float(speed_range[0]), float(speed_range[1]))
        self.track_scale = float(track_scale)
        self.frame = frame

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        theta = rng.uniform(-np.pi, np.pi, size=n)
        speed = rng.uniform(self.speed_range[0], self.speed_range[1], size=n)
        return np.stack([np.cos(theta) * speed, np.sin(theta) * speed], axis=1).astype(np.float32)

    def to_obs(self, env: VecEnv, goal: np.ndarray) -> np.ndarray:
        if self.frame == "world":
            return goal.astype(np.float32)
        # express the world target (vx, vz) in the character's heading frame (shared convention)
        yaw = root_yaw(env)
        lx, lz = rotate_to_heading(goal[:, 0], goal[:, 1], yaw)
        return np.stack([lx, lz], axis=1).astype(np.float32)

    def reward(self, env: VecEnv, goal: np.ndarray) -> np.ndarray:
        err = np.sum((planar_velocity(env) - goal) ** 2, axis=1)
        return np.exp(-self.track_scale * err).astype(np.float32)


class CompositeCommand:
    """Concatenate several commands into one (goals, obs channels, and rewards are stacked/summed).
    Lets a controller layer multiple control inputs (the combat case)."""

    def __init__(self, commands: Sequence[Command]):
        self.commands = list(commands)
        self.dim = sum(c.dim for c in self.commands)
        self._goal_splits = np.cumsum([c.sample(1, np.random.default_rng(0)).shape[1] for c in self.commands])[:-1]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.concatenate([c.sample(n, rng) for c in self.commands], axis=1)

    def _split(self, goal: np.ndarray) -> list[np.ndarray]:
        return np.split(goal, self._goal_splits, axis=1) if self.commands else []

    def to_obs(self, env: VecEnv, goal: np.ndarray) -> np.ndarray:
        parts = self._split(goal)
        return np.concatenate([c.to_obs(env, g) for c, g in zip(self.commands, parts)], axis=1)

    def reward(self, env: VecEnv, goal: np.ndarray) -> np.ndarray:
        parts = self._split(goal)
        return np.sum([c.reward(env, g) for c, g in zip(self.commands, parts)], axis=0)
