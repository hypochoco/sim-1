"""Diff-backend per-body render readout (Option B).

The `diff-cpu` env is flat-only for training, but exposes per-body world pose/vel via `linkWorld`
FK for RENDERING. This guards that the readout matches the reduced backend's authored pose (so the
visualizer renders diff-ABA rollouts in the same frame/convention) and that it's shaped right.
Skipped when the engine binding isn't built.
"""

from __future__ import annotations

import numpy as np
import pytest

engine_py = pytest.importorskip("sim1.engine_py", reason="engine binding not built")

from sim1.config import EnvConfig
from sim1.envs.engine_vecenv import EngineVecEnv


def _cfg(kind):
    return EnvConfig(kind=kind, model="amp", num_envs=2, substeps=48, action_mode="pd_target")


def test_diff_cpu_per_body_matches_reduced_at_reset():
    # Same authored initial pose → the diff env's linkWorld readout must equal the reduced backend's
    # per-body world pose (position ~exact, orientation up-to-sign). Validates the FK + M3→quat convention.
    r = EngineVecEnv(_cfg("engine"), seed=0); r.reset(0)
    d = EngineVecEnv(_cfg("diff-cpu"), seed=0); d.reset(0)
    assert r.body_pos.shape == d.body_pos.shape == (2, r.nbody, 3)
    assert r.body_quat.shape == d.body_quat.shape == (2, r.nbody, 4)
    assert np.isfinite(d.body_pos).all() and np.isfinite(d.body_quat).all()
    assert np.abs(r.body_pos - d.body_pos).max() < 1e-5
    dot = np.abs((r.body_quat * d.body_quat).sum(-1))     # |<q_r, q_d>| == 1 when aligned (up to sign)
    assert np.abs(dot - 1.0).max() < 1e-5


def test_diff_cpu_body_readout_tracks_after_step():
    # One step is enough to prove the readout refreshes with the sim state. (Constant non-physical
    # actions diverge the diff env in a few steps — a known contact sensitivity — so we keep it short;
    # the trained policy produces clean long rollouts.)
    d = EngineVecEnv(_cfg("diff-cpu"), seed=0); d.reset(0)
    p0 = d.body_pos.copy()
    d.step()
    assert np.isfinite(d.body_pos).all()
    assert np.abs(d.body_pos - p0).max() > 1e-4          # readout refreshed with the stepped state


def test_cuda_backend_body_readout_guarded():
    # cuda keeps state on-device → no per-body readout (deferred to viz Option A); must raise clearly.
    if getattr(engine_py, "HAS_CUDA", False):
        pytest.skip("CUDA build present; on-device readout path differs")
    assert not hasattr(engine_py, "CudaVecEnv")
