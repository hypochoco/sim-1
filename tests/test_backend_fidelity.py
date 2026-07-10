"""Reduced-PGS vs diff-ABA backend fidelity.

The two backends will never be bit-identical (different solvers: PGS vs smoothed-contact ABA), but from
the SAME initial state under the SAME actions they must stay *somewhat similar* over a short horizon —
otherwise a policy trained on one is meaningless on the other, and the diff-ABA viz looks wrong.

Findings (2026-07-09 deep dive; RESOLVED 2026-07-10 — see investigations/2026-07-10-diff-backend-convergence.md):
  * reset STATE is bit-identical across backends (shared authoring) — asserted below.
  * The reduced↔diff gaps (joint limits dropped, no damping, and the rest-pose energy injection) are
    now fixed in the diff engine: joint limits + reduced-style damping, SemiImplicit/IMEX contact, and
    — the key one — PD torque recomputed EVERY substep (a fixed-per-control-step tau had a stale −kd·q̇
    term that destabilized stiff PD-hold). The diff humanoid now rests / respects limits like reduced,
    so all four checks below are strict PASSes.
"""

from __future__ import annotations

import numpy as np
import pytest

engine_py = pytest.importorskip("sim1.engine_py", reason="engine binding not built")

from sim1.config import EnvConfig
from sim1.envs.engine_vecenv import EngineVecEnv


def _mk(kind, substeps=48):
    return EngineVecEnv(EnvConfig(kind=kind, model="humanoid", num_envs=1, substeps=substeps,
                        action_mode="pd_target", kp=150.0, kd=15.0, ground_friction=0.9), seed=0)


def test_reset_state_is_identical_across_backends():
    r = _mk("engine"); r.reset(0)
    d = _mk("diff-cpu"); d.reset(0)
    assert r.ndof == d.ndof and r.nbody == d.nbody
    assert np.abs(r.root_pose - d.root_pose).max() < 1e-5     # shared authoring / RSI pose
    assert np.abs(r.joint_q - d.joint_q).max() < 1e-5


def _zero_action_rollout(kind, steps=40, substeps=48):
    e = _mk(kind, substeps); e.reset(0)
    y0 = float(e.root_pose[0, 1]); ymax = y0; qdmax = 0.0
    for _ in range(steps):
        e.actions[:] = 0.0
        e.step()
        rp = e.root_pose[0]; qd = e.joint_qd[0]
        if not (np.isfinite(rp).all() and np.isfinite(qd).all()):
            qdmax = np.inf; break
        ymax = max(ymax, float(rp[1])); qdmax = max(qdmax, float(np.abs(qd).max()))
    return y0, ymax, qdmax


def test_reduced_zero_action_is_stable():
    # Baseline: reduced backend holds the rest pose (no energy gain) under zero PD target.
    y0, ymax, qdmax = _zero_action_rollout("engine")
    assert ymax - y0 < 0.05          # root does not rise (no energy injection)
    assert qdmax < 5.0               # joints stay quiescent


def test_diff_zero_action_conserves_energy():
    # FIXED (2026-07-10, per-substep PD + SemiImplicit contact): a passive/PD-held diff humanoid must
    # not GAIN height or spin up under zero action — it now rests like reduced (was root→1.17, qd→500+).
    y0, ymax, qdmax = _zero_action_rollout("diff-cpu")
    assert ymax - y0 < 0.05
    assert qdmax < 20.0


def test_backends_agree_short_horizon_zero_action():
    # Same init + same (zero) actions → root height tracks within ~10 cm for a short horizon.
    r = _mk("engine"); r.reset(0)
    d = _mk("diff-cpu"); d.reset(0)
    for _ in range(20):
        r.actions[:] = 0.0; d.actions[:] = 0.0
        r.step(); d.step()
    assert abs(float(r.root_pose[0, 1]) - float(d.root_pose[0, 1])) < 0.10
