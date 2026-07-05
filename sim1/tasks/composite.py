"""CompositeTask — the extensible task base.

A task is assembled from parts rather than hand-rolled, so new scenarios are declarative and the
observation stays transfer-stable:

    observation = [ proprioception | command channels | extra obs components... ]   (canonical order)
    reward      = Σ weightᵢ · termᵢ(env, actions, task)  +  command_weight · command.reward
    terminate   = terminate_fn(env, task)   (time-limit truncation is handled by TaskEnv)

Why this shape:
- **Proprioception first, always identical** → the policy's input trunk aligns across tasks, so
  `--init-from` warm-starts transfer cleanly (stand → getup → walk → combat).
- **Command channels appended** → goal-conditioning + user control drop in without touching obs
  plumbing; adding a control input = adding a `Command`.
- **Extra obs components** → the seam for reference/phase features when mocap/tracking arrives.
- **Reward as terms** → many-objective tasks (combat) stay legible and per-term loggable
  (`task.reward_info`).

Implements the `Task` Protocol (`base.py`), so `TaskEnv`/PPO/eval/export use it unchanged.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from sim1.envs.vecenv import VecEnv
from sim1.tasks.command import Command, NoCommand
from sim1.tasks.proprio import proprio_dim
from sim1.tasks.rewards import RewardTerm, never_terminate


class ObsComponent(Protocol):
    dim: int
    def observe(self, env: VecEnv, task) -> np.ndarray: ...  # (num_envs, dim)


class CompositeTask:
    def __init__(
        self,
        ndof: int,
        nbody: int,
        act_dim: int,
        action_scale: float,
        *,
        command: Command | None = None,
        reward_terms: Sequence[RewardTerm] = (),
        terminate_fn: Callable = never_terminate,
        command_weight: float = 1.0,
        extra_obs: Sequence[ObsComponent] = (),
        rot_repr: str = "quat",
        frame: str = "world",
    ):
        self.ndof = int(ndof)
        self.nbody = int(nbody)
        self.act_dim = int(act_dim)
        self.action_scale = float(action_scale)
        self.rot_repr = rot_repr   # root-orientation obs encoding: "quat" | "sixd"
        self.frame = frame         # proprioceptive frame: "world" | "local" (heading-relative)

        self.command: Command = command if command is not None else NoCommand()
        self.reward_terms = list(reward_terms)
        self.terminate_fn = terminate_fn
        self.command_weight = float(command_weight)
        self.extra_obs = list(extra_obs)

        self.obs_dim = proprio_dim(ndof, nbody, rot_repr) + self.command.dim + sum(c.dim for c in self.extra_obs)
        self._target_h: np.ndarray | None = None   # authored standing height, captured at reset
        self._start_xz: np.ndarray | None = None    # start (x, z) position, captured at reset
        self._goals: np.ndarray | None = None        # per-env command goal
        self.reward_info: dict[str, float] = {}      # last per-term reward means (for logging/debug)

    # --- Task protocol ---
    def reset(self, env: VecEnv, seed: int) -> None:
        self._target_h = env.root_pose[:, 1].copy()
        self._start_xz = env.root_pose[:, [0, 2]].copy()
        n = env.root_pose.shape[0]
        self._goals = self.command.sample(n, np.random.default_rng(seed))

    def reset_masked(self, env: VecEnv, mask: np.ndarray, seed: int) -> None:
        if self._goals is None or self._target_h is None:
            self.reset(env, seed)
            return
        m = np.asarray(mask, dtype=bool)
        if m.any():
            self._goals[m] = self.command.sample(int(m.sum()), np.random.default_rng(seed))
            self._start_xz[m] = env.root_pose[m][:, [0, 2]]   # re-anchor reset envs to their new spot

    def observe(self, env: VecEnv) -> np.ndarray:
        parts = [env.compose_proprio(self.rot_repr, self.frame)]   # single obs source (C++ engine / Python mock)
        if self.command.dim:
            parts.append(self.command.to_obs(env, self._goals))
        for c in self.extra_obs:
            parts.append(c.observe(env, self))
        return np.concatenate(parts, axis=1).astype(np.float32) if len(parts) > 1 else parts[0]

    def reward(self, env: VecEnv, actions: np.ndarray) -> np.ndarray:
        total = np.zeros(env.root_pose.shape[0], dtype=np.float32)
        info: dict[str, float] = {}
        for t in self.reward_terms:
            v = t.weight * t.fn(env, actions, self)
            total += v
            info[t.name] = float(v.mean())
        if self.command.dim or self.command_weight:
            cr = self.command_weight * self.command.reward(env, self._goals)
            total += cr
            info["command"] = float(cr.mean())
        self.reward_info = info
        return total.astype(np.float32)

    def done(self, env: VecEnv, ep_step: np.ndarray) -> np.ndarray:
        return self.terminate_fn(env, self)

    # --- inference: set the goal from user input instead of sampling (user-controllable) ---
    def set_goal(self, goal: np.ndarray) -> None:
        self._goals = np.asarray(goal, dtype=np.float32)
