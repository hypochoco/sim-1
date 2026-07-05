"""Tests for the composable task infrastructure (proprio / command / composite / walk).

The most important assertion is the **transfer invariant**: the canonical proprioception block is
byte-identical to what `StandTask.observe` produces, so a stand/getup policy warm-starts walk (and
future tasks) with an aligned input trunk.
"""

from __future__ import annotations

import numpy as np

from sim1.envs.mock_vecenv import MockVecEnv
from sim1.tasks.command import HeadingSpeedCommand, NoCommand
from sim1.tasks.composite import CompositeTask
from sim1.tasks.proprio import proprio_dim, proprio_obs
from sim1.tasks.rewards import RewardTerm, fall_termination, term_alive, term_upright
from sim1.tasks.stand import StandTask
from sim1.tasks.walk import WalkTask


def _env(n=8, ndof=28):
    return MockVecEnv(num_envs=n, ndof=ndof, seed=0)


def test_proprio_matches_stand_obs():
    # Transfer invariant: the canonical proprio block == StandTask's hand-rolled observation.
    env = _env()
    stand = StandTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0)
    stand.reset(env, 0)
    assert np.array_equal(proprio_obs(env), stand.observe(env))
    assert proprio_dim(env.ndof, 1) == stand.obs_dim


def test_nocommand_composite_is_proprio_only():
    env = _env()
    t = CompositeTask(env.ndof, 1, env.act_dim, 1.0, command=NoCommand())
    t.reset(env, 0)
    assert t.obs_dim == proprio_dim(env.ndof, 1)
    assert t.observe(env).shape == (env.num_envs, t.obs_dim)


def test_heading_speed_command():
    env = _env()
    cmd = HeadingSpeedCommand(speed_range=(0.5, 1.5))
    rng = np.random.default_rng(0)
    g = cmd.sample(env.num_envs, rng)
    assert g.shape == (env.num_envs, 2)
    speeds = np.linalg.norm(g, axis=1)
    assert np.all(speeds >= 0.5 - 1e-5) and np.all(speeds <= 1.5 + 1e-5)
    assert cmd.to_obs(env, g).shape == (env.num_envs, 2)
    # mock env has identity orientation (yaw 0), so root-local == world here
    assert np.allclose(cmd.to_obs(env, g), g)
    r = cmd.reward(env, g)
    assert r.shape == (env.num_envs,) and np.all((r >= 0) & (r <= 1))


def test_stand_behavior_matches_closed_form():
    # Weight-compatibility guard: the migrated (composable) StandTask must reproduce the original
    # observation + reward + termination exactly, so weights from prior runs stay usable.
    from sim1.tasks.proprio import root_height, uprightness
    env = _env()
    uw, hw, ab, aw = 1.3, 0.7, 1.0, 0.02
    t = StandTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0,
                  upright_weight=uw, height_weight=hw, alive_bonus=ab, action_weight=aw,
                  fall_height_frac=0.5, upright_fall=0.3)
    t.reset(env, 0)
    env.step()
    a = np.random.default_rng(1).standard_normal((env.num_envs, env.act_dim)).astype(np.float32)
    up = uprightness(env); h = root_height(env)
    expected = (ab + uw * up + hw * np.exp(-10.0 * (h - t._target_h) ** 2) - aw * np.mean(a ** 2, axis=1))
    assert np.allclose(t.reward(env, a), expected, atol=1e-6)
    assert np.array_equal(proprio_obs(env), t.observe(env))          # obs unchanged
    assert not t.done(env, np.zeros(env.num_envs)).any()             # upright mock → no fall


def test_walk_task_obs_and_reward():
    env = _env()
    t = WalkTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0)
    # obs = proprio + 2 command channels
    assert t.obs_dim == proprio_dim(env.ndof, 1) + 2
    t.reset(env, 0)
    obs = t.observe(env)
    assert obs.shape == (env.num_envs, t.obs_dim)
    r = t.reward(env, np.zeros((env.num_envs, env.act_dim), dtype=np.float32))
    assert r.shape == (env.num_envs,) and np.all(np.isfinite(r))
    # per-term logging populated (alive/upright/height/ctrl/command)
    assert {"alive", "upright", "height", "ctrl", "command"} <= set(t.reward_info)


def test_composite_reset_masked_resamples_subset():
    env = _env()
    t = WalkTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0)
    t.reset(env, 0)
    g0 = t._goals.copy()
    mask = np.zeros(env.num_envs, dtype=bool)
    mask[: env.num_envs // 2] = True
    t.reset_masked(env, mask, seed=1)
    assert not np.array_equal(t._goals[mask], g0[mask])          # masked envs resampled
    assert np.array_equal(t._goals[~mask], g0[~mask])            # others unchanged


def test_set_goal_for_user_control():
    env = _env()
    t = WalkTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0)
    t.reset(env, 0)
    t.set_goal(np.tile(np.array([1.0, 0.0], dtype=np.float32), (env.num_envs, 1)))
    assert np.array_equal(t.observe(env)[:, -2:], np.tile([1.0, 0.0], (env.num_envs, 1)))
