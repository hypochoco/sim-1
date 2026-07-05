"""setArticulationState (RSI primitive) — the reduced backend must reconstruct its generalized state
from per-body world poses/velocities exactly, i.e. be the inverse of the body_pos/quat/linvel/angvel
readers. Guards the engine change from 2026-07-05-rsi-state-set-engine-plan.md. Skipped when the
engine binding isn't built."""

from __future__ import annotations

import numpy as np
import pytest

engine_py = pytest.importorskip("sim1.engine_py", reason="engine binding not built")


def _amp_vecenv(n=1):
    cfg = engine_py.EnvConfig(engine_py.make_amp_humanoid(), engine_py.SimConfig())
    return engine_py.VecEnv(n, cfg, 0)


def _read(env):
    return (np.array(env.body_pos()), np.array(env.body_quat()),
            np.array(env.body_linvel()), np.array(env.body_angvel()))


def _f32(*arrs):
    return tuple(a.astype(np.float32) for a in arrs)


def test_set_articulation_state_roundtrip():
    # env A → a strongly articulated, MOVING pose (collapse under gravity); capture its world state
    A = _amp_vecenv(); A.reset(0)
    A.actions()[:] = 0.0
    for _ in range(18):
        A.step()                       # mid-fall: large displacement from rest + nonzero velocity
    pos, quat, lin, ang = _read(A)
    assert np.isfinite(pos).all()
    R = _amp_vecenv(); R.reset(0)
    assert np.abs(np.array(R.body_pos()) - pos).max() > 0.05   # genuinely away from the rest pose
    assert np.abs(lin).max() > 0.1                             # and genuinely moving (velocity matters)

    # env B: rest → set to A's state → must reproduce it exactly (inverse of the readers)
    B = _amp_vecenv(); B.reset(0)
    B.set_articulation_state(*_f32(pos, quat, lin, ang))
    bpos, bquat, blin, bang = _read(B)
    assert np.abs(bpos - pos).max() < 1e-5
    assert np.abs(blin - lin).max() < 1e-5
    assert np.abs(bang - ang).max() < 1e-5
    # quats equal up to sign + float precision
    assert np.minimum(np.abs(bquat - quat), np.abs(bquat + quat)).max() < 1e-5


def test_state_persists_and_steps_finite():
    # after setting a pose, the dynamics continue from it without blowing up
    A = _amp_vecenv(); A.reset(0)
    act = A.actions(); rng = np.random.default_rng(1)
    for _ in range(5):
        act[:] = rng.uniform(-0.4, 0.4, act.shape); A.step()
    pos, quat, lin, ang = _read(A)

    B = _amp_vecenv(); B.reset(0)
    B.set_articulation_state(*_f32(pos, quat, lin, ang))
    B.actions()[:] = 0.0
    for _ in range(10):
        B.step()
    assert np.isfinite(np.array(B.body_pos())).all()
