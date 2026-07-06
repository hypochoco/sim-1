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
    for name in ("v1", "v2", "v3"):
        out = make_reward(name, NB).evaluate(sim, ref, _A)
        assert out.total.shape == (1,) and np.isfinite(out.total).all()
        assert out.failed.shape == (1,) and out.failed.dtype == bool


# --- v3: simplified, correctly-scaled pure-imitation reward -------------------------------------
def _rotate_bodies(st: BodyState, angle: float, bodies) -> BodyState:
    """Return a copy of `st` with the given bodies' orientations rotated by `angle` rad about +X."""
    from sim1.tasks.proprio import _quat_mul
    q = st.body_quat.copy()
    rot = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0], np.float32)
    for b in bodies:
        q[0, b] = _quat_mul(rot[None], q[0, b][None])[0]
    return BodyState(st.body_pos, q, st.body_linvel, st.body_angvel)


def test_v3_info_keys_and_reward_bounded_0_1():
    ref = _state(seed=7)
    v3 = make_reward("v3", NB)
    match = v3.evaluate(ref, ref, _A)
    assert set(match.info) == {"pose", "ee", "vel", "root"}      # no alive/progress/slip/ctrl
    assert abs(match.total.mean() - 1.0) < 1e-4                  # weights sum to 1 → match == 1
    off = v3.evaluate(_rotate_bodies(ref, 0.5, range(1, NB)), ref, _A).total.mean()
    assert 0.0 <= off < 1.0                                      # bounded in [0, 1]


def test_v3_mean_normalization_is_responsive_where_v2_saturates():
    # A realistic ~25 deg/body orientation error: v3 must stay in the responsive band; v2 saturates ~0.
    ref = _state(seed=8)
    sim = _rotate_bodies(ref, 0.44, range(1, NB))               # 0.44 rad ≈ 25 deg on every non-root body
    p_v3 = make_reward("v3", NB).evaluate(sim, ref, _A).info["pose"]
    p_v2 = make_reward("v2", NB).evaluate(sim, ref, _A).info["pose"]
    assert 0.2 < p_v3 < 0.8, p_v3                               # gradient available
    assert p_v2 < 0.02, p_v2                                    # v2 (summed) is dead here
    assert p_v3 > 20 * p_v2


def test_v3_reward_disfavors_shuffle_without_premature_termination():
    # Limb orientations wildly off but positions fine & upright: the reward must punish it (low pose),
    # yet NOT terminate on orientation — an untrained policy looks like this early and needs room to learn.
    ref = _state(seed=9)
    shuffle = _rotate_bodies(ref, 1.2, range(1, NB))           # ~69 deg/body, positions unchanged
    out = make_reward("v3", NB).evaluate(shuffle, ref, _A)
    assert out.info["pose"] < 0.05                             # orientation error is penalized in reward
    assert out.total.mean() < 1.0                              # below a perfect match
    assert bool(out.failed[0]) is False                        # but not terminated → bootstrapping room


def test_v3_terminates_on_position_divergence_and_fall():
    ref = _state(seed=10)
    v3 = make_reward("v3", NB)
    far = _state(seed=10)
    far.body_pos[0, 1:] = far.body_pos[0, 1:] + np.array([0.6, 0.0, 0.0], np.float32)  # limbs 0.6m off root
    assert bool(v3.evaluate(far, ref, _A).failed[0]) is True   # position backstop
    fallen = _state(seed=10)
    fallen.body_pos[0, 0, 1] = 0.2                             # root collapsed (< 0.5 · ref height 1.0)
    assert bool(v3.evaluate(fallen, ref, _A).failed[0]) is True  # fall
