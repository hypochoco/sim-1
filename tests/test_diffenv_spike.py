"""DiffEnvironment binding spike — proves the differentiable-sim seam (sim-1 only, no engine change).

(1) The engine's analytic rollout gradient, taken THROUGH the binding, matches central finite
    differences (also through the binding) — the same check `tst/physics/integration/diff_environment.cpp`
    runs in-engine, now reproduced across the Python boundary.
(2) A short rollout stays finite; the per-step tangent Jacobian is finite and correctly shaped.
(3) Diff env vs the forward EngineVecEnv on identical (passive) actions — documents the trajectory
    drift between the smoothed-compliant diff contact and the hard-PGS forward contact (watch-list #2 in
    notes/investigations/2026-07-05-engine-change-watchlist.md).

Skipped when the engine binding isn't built."""

from __future__ import annotations

import numpy as np
import pytest

engine_py = pytest.importorskip("sim1.engine_py", reason="engine binding not built")


def _action(dim):
    a = np.zeros(dim, dtype=np.float64)
    a[:4] = [1.0, -0.3, 0.2, 0.5]   # waist ball xyz + shoulderL x (matches the engine test)
    return a


def test_diffenv_rollout_gradient_matches_fd():
    de = engine_py.DiffEnv(model="amp", contact="none")   # in-air: clean articulated-dynamics gradient
    n_steps = 5
    a = _action(de.action_dim)
    de.reset()
    grad = np.asarray(de.rollout_grad_qd0(a, n_steps))     # analytic ∂qd[0]/∂action[0:4]

    def qd0_after(action):
        de.reset()
        de.set_action(action)
        for _ in range(n_steps):
            de.step()
        return de.qd()[0]

    eps = 1e-6
    fd = np.empty(4)
    for j in range(4):
        ap = a.copy(); ap[j] += eps
        am = a.copy(); am[j] -= eps
        fd[j] = (qd0_after(ap) - qd0_after(am)) / (2 * eps)
    err = float(np.max(np.abs(grad - fd)))
    assert np.max(np.abs(grad)) > 1e-3          # the gradient is meaningfully non-zero
    assert err < 1e-4, f"analytic vs FD mismatch {err:.2e} (grad={grad}, fd={fd})"


def test_diffenv_rollout_stays_finite():
    de = engine_py.DiffEnv(model="amp", contact="all")
    rng = np.random.default_rng(0)
    for _ in range(30):
        de.set_action(rng.uniform(-2.0, 2.0, de.action_dim))
        de.step()
    lp = de.link_pos()
    assert lp.shape == (de.nbody, 3)
    assert np.isfinite(lp).all()
    assert np.abs(lp[:, [0, 2]]).max() < 5.0 and -12.0 < lp[:, 1].min()


def test_diffenv_jacobian_shape_and_finite():
    de = engine_py.DiffEnv(model="amp", contact="feet")
    de.reset()
    J = de.jacobian()
    ndof = de.action_dim
    assert J.shape == (2 * (6 + ndof), 2 * (6 + ndof) + ndof)   # nState × nInput = 68 × 96 (AMP)
    assert np.isfinite(J).all()


def test_diffenv_vs_forward_trajectory_consistency():
    # Passive (zero-torque) dynamics from the authored pose: isolates the articulated dynamics +
    # contact. Documents the diff↔forward root drift (different contact models ⇒ some drift expected).
    from sim1.config import EnvConfig
    from sim1.envs.engine_vecenv import EngineVecEnv

    dt, sub = 1.0 / 60.0, 48
    de = engine_py.DiffEnv(model="amp", contact="all", control_dt=dt, substeps=sub)
    de.reset()
    fwd = EngineVecEnv(EnvConfig(kind="engine", model="amp", backend="reduced", num_envs=1,
                                 substeps=sub, control_dt=dt, action_mode="torque"), seed=0)
    za = np.zeros(de.action_dim, dtype=np.float64)
    fwd.actions[:] = 0.0
    drift = []
    for _ in range(10):
        de.set_action(za); de.step()
        fwd.step()
        drift.append(float(np.linalg.norm(de.link_pos()[0] - fwd.root_pose[0, 0:3])))
    drift = np.array(drift)
    print(f"\ndiff↔forward root drift per step (m): {np.round(drift, 4)}  max={drift.max():.4f}")
    assert np.isfinite(drift).all()
    assert drift[0] < 0.05           # first control step agrees to a few cm (dynamics consistent)
    assert drift.max() < 0.5         # no blow-up over the short horizon (loose sanity bound)
