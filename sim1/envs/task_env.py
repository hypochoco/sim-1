"""TaskEnv — adapts a (VecEnv + Task) pair into the standard vectorized RL interface.

`reset() -> obs` and `step(actions) -> (obs, reward, done, info)`, with fixed-horizon time-limit
truncation and auto-reset of finished envs. On any env that finishes, `info["terminal_obs"]` holds
the observation at the terminal state (before reset) and `info["truncated"]` marks time-limit ends
(vs `info["fail"]`), so the algorithm can bootstrap the value correctly.
"""

from __future__ import annotations

import numpy as np

from sim1.envs.vecenv import VecEnv
from sim1.tasks.base import Task


class TaskEnv:
    def __init__(self, env: VecEnv, task: Task, episode_len: int, seed: int = 0):
        self.env = env
        self.task = task
        self.episode_len = int(episode_len)
        self.num_envs = env.num_envs
        self.obs_dim = task.obs_dim
        self.act_dim = task.act_dim
        self._seed = int(seed)
        self._reset_counter = 0
        self._ep_step = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self) -> np.ndarray:
        self.env.reset(self._seed)
        self.task.reset(self.env, self._seed)
        self._ep_step[:] = 0
        return self.task.observe(self.env)

    def step(self, actions: np.ndarray):
        self.env.actions[:] = actions * self.task.action_scale
        self.env.step()
        self._ep_step += 1

        reward = self.task.reward(self.env, actions)
        fail = self.task.done(self.env, self._ep_step)
        truncated = self._ep_step >= self.episode_len

        terminal_obs = self.task.observe(self.env)  # obs at terminal state (before reset)

        # Divergence guard: a physics blow-up (non-finite obs/reward) is treated as a failed
        # episode — sanitize the values and force-reset that env, so one bad world can't NaN-poison
        # the whole batch or the policy update.
        nonfinite = ~np.isfinite(terminal_obs).all(axis=1)
        if nonfinite.any():
            fail = fail | nonfinite
            terminal_obs = np.nan_to_num(terminal_obs, nan=0.0, posinf=0.0, neginf=0.0)
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        done = np.logical_or(fail, truncated)
        info = {"terminal_obs": terminal_obs, "truncated": truncated.copy(), "fail": fail.copy()}

        obs = terminal_obs
        if done.any():
            self._reset_counter += 1
            seed = self._seed + 7919 * self._reset_counter
            self.task.reset_masked(self.env, done, seed)
            self.env.reset_masked(done, seed)
            self._ep_step[done] = 0
            obs = self.task.observe(self.env)  # fresh obs for the continuing rollout

        return obs, reward, done, info
