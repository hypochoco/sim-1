"""Composable reward terms + termination factories.

A reward is a **weighted sum of named terms** rather than one monolithic expression. This scales to
tasks with many objectives (combat: stay-up + track-target + hit + energy + style) and lets us log
each term's contribution to diagnose what the policy is actually optimizing. Each term is
`fn(env, actions, task) -> (num_envs,)`; the task multiplies by the term's weight and sums.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sim1.envs.vecenv import VecEnv
from sim1.tasks.proprio import root_height, uprightness

TermFn = Callable[[VecEnv, np.ndarray, "object"], np.ndarray]


@dataclass
class RewardTerm:
    name: str
    weight: float
    fn: TermFn


# --- standard humanoid terms (raw signals; the weight is applied by CompositeTask) ---
def term_alive(env: VecEnv, actions: np.ndarray, task) -> np.ndarray:
    return np.ones(env.root_pose.shape[0], dtype=np.float32)


def term_upright(env: VecEnv, actions: np.ndarray, task) -> np.ndarray:
    return uprightness(env).astype(np.float32)


def term_height(env: VecEnv, actions: np.ndarray, task) -> np.ndarray:
    # Gaussian on the gap to the authored standing height (task._target_h, captured at reset).
    return np.exp(-10.0 * (root_height(env) - task._target_h) ** 2).astype(np.float32)


def term_ctrl_penalty(env: VecEnv, actions: np.ndarray, task) -> np.ndarray:
    return np.mean(actions ** 2, axis=1).astype(np.float32)


def position_keep_term(scale: float = 1.0) -> TermFn:
    """Factory: `exp(-scale · ‖root_xz − start_xz‖²)` — 1.0 at the start position, decaying with
    horizontal drift. Rewards holding a spot (stand/getup); needs `task._start_xz` (captured at
    reset). Not for locomotion tasks (walk *should* move)."""
    def fn(env: VecEnv, actions: np.ndarray, task) -> np.ndarray:
        xz = np.stack([env.root_pose[:, 0], env.root_pose[:, 2]], axis=1)
        d2 = np.sum((xz - task._start_xz) ** 2, axis=1)
        return np.exp(-scale * d2).astype(np.float32)
    return fn


# --- termination factories: fn(env, task) -> bool (num_envs,) ---
def fall_termination(fall_height_frac: float, upright_fall: float) -> Callable:
    def fn(env: VecEnv, task) -> np.ndarray:
        fell = (root_height(env) < fall_height_frac * task._target_h) | (uprightness(env) < upright_fall)
        return fell.astype(bool)
    return fn


def never_terminate(env: VecEnv, task) -> np.ndarray:
    return np.zeros(env.root_pose.shape[0], dtype=bool)
