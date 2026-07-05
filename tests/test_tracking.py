"""TrackingTask (DeepMimic-style imitation) — RSI, imitation reward, phase clock, early termination.
Skipped when the engine binding or the ASE walk clip isn't available."""

from __future__ import annotations

import numpy as np
import pytest

engine_py = pytest.importorskip("sim1.engine_py", reason="engine binding not built")

from sim1.config import EnvConfig
from sim1.envs.engine_vecenv import EngineVecEnv
from sim1.motion.motion_lib import default_motion_path, load_reference_motion, retarget_to_rig
from sim1.tasks.proprio import proprio_dim
from sim1.tasks.track import TrackingTask

_WALK = default_motion_path("amp_humanoid_walk")
pytestmark = pytest.mark.skipif(not _WALK.exists(), reason=f"ASE walk clip not found at {_WALK}")


def _make(n=4):
    env = EngineVecEnv(EnvConfig(kind="engine", model="amp", num_envs=n, action_mode="pd_target"), seed=0)
    env.reset(0)
    motion = retarget_to_rig(load_reference_motion(_WALK), np.asarray(env.body_pos)[0])
    task = TrackingTask(env.ndof, env.nbody, env.act_dim, action_scale=1.0, motion=motion, control_dt=1.0 / 60.0)
    return env, task, motion


def _zero_act(env):
    return np.zeros((env.num_envs, env.act_dim), dtype=np.float32)


def test_obs_is_proprio_plus_phase():
    env, task, _ = _make()
    assert task.obs_dim == proprio_dim(env.ndof, env.nbody, "quat") + 2
    task.reset(env, seed=0)
    obs = task.observe(env)
    assert obs.shape == (env.num_envs, task.obs_dim) and np.isfinite(obs).all()
    assert np.abs(obs[:, -2:]).max() <= 1.0 + 1e-6         # phase = (sin, cos)


def test_rsi_sets_sim_to_reference():
    env, task, _ = _make()
    task.reset(env, seed=0)
    ref = task._lib.state_at(task._ref_time)
    assert np.abs(np.asarray(env.body_pos) - ref.body_pos).max() < 1e-3       # position: exact
    # velocities are re-derived from the reference angular velocities projected onto valid joint DOFs
    # (revolute joints drop off-axis components) + base twist — physically consistent, so just bounded
    assert np.isfinite(np.asarray(env.body_linvel)).all()
    assert np.isfinite(np.asarray(env.body_angvel)).all()
    assert np.abs(np.asarray(env.body_linvel)).max() < 20.0


def test_reward_peaks_at_reference():
    env, task, motion = _make()
    task.reset(env, seed=0)                       # sim == reference@start
    r_match = task.reward(env, _zero_act(env)).mean()
    assert r_match > 0.7                          # near max (only a fraction of a ref-frame of drift)
    # sim at rest, phase mid-stride → clear mismatch → lower reward
    env.reset(0)
    task._ref_time[:] = 0.4 * motion.duration
    r_off = task.reward(env, _zero_act(env)).mean()
    assert r_off < r_match - 0.2


def test_phase_advances_and_wraps():
    env, task, motion = _make()
    task.reset(env, seed=0)
    t0 = task._ref_time.copy()
    task.reward(env, _zero_act(env))
    assert np.allclose(task._ref_time, np.mod(t0 + 1.0 / 60.0, motion.duration))


def test_early_termination_on_divergence():
    env, task, motion = _make()
    task.term_pos_err = 0.05                      # tighten so a rest-vs-stride mismatch trips it
    task.reset(env, seed=0)
    env.reset(0)                                  # diverge the sim (rest) from a mid-stride phase
    task._ref_time[:] = 0.4 * motion.duration
    task.reward(env, _zero_act(env))
    assert task.done(env, None).any()


def test_heading_frame_is_direction_invariant():
    # The reward's comparison features must be invariant to a global heading (yaw) rotation.
    from sim1.tasks.proprio import _quat_mul
    from sim1.tasks.track import _heading_frame

    rng = np.random.default_rng(0)
    n, b = 3, 15
    bp = rng.standard_normal((n, b, 3)).astype(np.float32)
    bq = rng.standard_normal((n, b, 4)).astype(np.float32)
    bq /= np.linalg.norm(bq, axis=-1, keepdims=True)
    bl = rng.standard_normal((n, b, 3)).astype(np.float32)
    ba = rng.standard_normal((n, b, 3)).astype(np.float32)
    f0 = _heading_frame(bp, bq, bl, ba)

    yaw = 1.2                                              # apply an arbitrary global yaw about +Y
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    qy = np.tile(np.array([np.cos(yaw / 2), 0, np.sin(yaw / 2), 0], np.float32), (n * b, 1))
    bp2 = bp @ Ry.T; bl2 = bl @ Ry.T; ba2 = ba @ Ry.T
    bq2 = _quat_mul(qy, bq.reshape(n * b, 4)).reshape(n, b, 4)
    f1 = _heading_frame(bp2, bq2, bl2, ba2)

    for a, c in zip(f0, f1):                               # rel_pos, quat, linvel, angvel all unchanged
        assert np.abs(np.asarray(a) - np.asarray(c)).max() < 1e-4
