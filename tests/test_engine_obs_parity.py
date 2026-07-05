"""Parity guard for the single obs source.

Observation composition now lives once in C++ (`csrc/obs/obs.h`, used by the training binding and the
visualizer). `sim1/tasks/proprio.py` is kept as the Python *reference oracle*. This test asserts the
engine's C++-composed obs matches the oracle bit-for-bit (within float tolerance), so the two can't
silently drift. Skipped when the engine binding isn't built (e.g. a pure-Python checkout)."""

from __future__ import annotations

import numpy as np
import pytest

engine_py = pytest.importorskip("sim1.engine_py", reason="engine binding not built")

from sim1.config import EnvConfig
from sim1.envs.engine_vecenv import EngineVecEnv
from sim1.tasks.proprio import per_body_obs, proprio_obs


def _stepped_env(n=8):
    cfg = EnvConfig(kind="engine", model="amp", backend="reduced", num_envs=n,
                    substeps=48, action_mode="pd_target")
    env = EngineVecEnv(cfg, seed=0)
    env.actions[:] = 0.1
    for _ in range(3):
        env.step()
    return env


@pytest.mark.parametrize("rotation", ["quat", "sixd"])
@pytest.mark.parametrize("frame", ["world", "local"])
def test_engine_proprio_matches_python_oracle(rotation, frame):
    env = _stepped_env()
    cpp = env.compose_proprio(rotation, frame)
    py = proprio_obs(env, rotation, frame)
    assert cpp.shape == py.shape
    assert np.max(np.abs(cpp - py)) < 1e-5


def test_engine_body_block_matches_python_oracle():
    env = _stepped_env()
    cpp = env.compose_body()
    py = per_body_obs(env, "sixd")
    assert cpp.shape == py.shape
    assert np.max(np.abs(cpp - py)) < 1e-5
