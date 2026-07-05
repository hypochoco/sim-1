"""Reward-registry unit tests (pure NumPy — no engine/clip needed). Exercises the named tracking
reward variants and the V2 additions (foot-slip, forward-progress, tighter termination)."""

from __future__ import annotations

import numpy as np
import pytest

from sim1.tasks.track_rewards import BodyState, FEET, make_reward

NB, ACT = 15, 28


def _state(seed=0, root_lin=(0.0, 0.0, 0.0)) -> BodyState:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-0.2, 0.2, (1, NB, 3)).astype(np.float32) + np.array([0, 1.0, 0], np.float32)
    pos[0, 0] = [0.0, 1.0, 0.0]                        # root
    for f in FEET:
        pos[0, f, 1] = 0.05                            # feet on the ground
    q = rng.standard_normal((1, NB, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    lin = np.zeros((1, NB, 3), np.float32); lin[0, 0] = root_lin
    ang = np.zeros((1, NB, 3), np.float32)
    return BodyState(pos, q, lin, ang)


_A = np.zeros((1, ACT), np.float32)


def test_registry_and_tighter_termination():
    v1, v2 = make_reward("v1", NB), make_reward("v2", NB)
    assert v1.name == "v1" and v2.name == "v2"
    assert v2.term_pos_err < v1.term_pos_err            # (4) V2 terminates sooner
    with pytest.raises(ValueError):
        make_reward("nope", NB)


def test_info_keys_differ():
    ref = _state()
    i1 = make_reward("v1", NB).evaluate(ref, ref, _A).info
    i2 = make_reward("v2", NB).evaluate(ref, ref, _A).info
    assert set(i1) == {"pose", "vel", "ee", "root", "ctrl"}
    assert {"progress", "slip"} <= set(i2)              # V2 adds the new terms


@pytest.mark.parametrize("name", ["v1", "v2"])
def test_reward_peaks_at_match(name):
    ref = _state(seed=1)
    r = make_reward(name, NB)
    match = r.evaluate(ref, ref, _A).total.mean()
    sim = _state(seed=1); sim.body_pos[:] += 0.4       # displace → worse
    off = r.evaluate(sim, ref, _A).total.mean()
    assert match > 0.7 and match > off


def test_v2_penalizes_foot_slip():
    # sim identical to the reference except a GROUNDED foot slides horizontally.
    ref = _state(seed=2)
    sim = _state(seed=2); sim.body_linvel[0, 13] = [2.0, 0.0, 0.0]
    v1, v2 = make_reward("v1", NB), make_reward("v2", NB)
    assert v2.evaluate(ref, ref, _A).info["slip"] > 0.9    # grounded + still → ~1
    assert v2.evaluate(sim, ref, _A).info["slip"] < 0.2    # grounded + sliding → ~0
    d1 = v1.evaluate(ref, ref, _A).total.mean() - v1.evaluate(sim, ref, _A).total.mean()
    d2 = v2.evaluate(ref, ref, _A).total.mean() - v2.evaluate(sim, ref, _A).total.mean()
    assert d2 > d1 + 0.05                                  # (2) V2 penalizes the slide beyond V1


def test_v2_rewards_forward_progress():
    # reference is moving forward; matching that root velocity should beat standing still.
    ref = _state(seed=3, root_lin=(1.0, 0.0, 0.0))
    v2 = make_reward("v2", NB)
    moving = v2.evaluate(_state(seed=3, root_lin=(1.0, 0.0, 0.0)), ref, _A)
    static = v2.evaluate(_state(seed=3, root_lin=(0.0, 0.0, 0.0)), ref, _A)
    assert moving.info["progress"] > static.info["progress"] + 0.3   # (3) progress rewards motion
    assert moving.total.mean() > static.total.mean()


def test_all_outputs_finite_and_shaped():
    ref, sim = _state(seed=4), _state(seed=5)
    for name in ("v1", "v2"):
        out = make_reward(name, NB).evaluate(sim, ref, _A)
        assert out.total.shape == (1,) and np.isfinite(out.total).all()
        assert out.failed.shape == (1,) and out.failed.dtype == bool
