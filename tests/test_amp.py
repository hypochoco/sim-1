"""AMP (adversarial motion prior) low-risk pieces: feature layout parity (sim vs reference), the real
transition sampler, and the discriminator (shapes, style-reward bounds, and that its LSGAN loss +
gradient penalty actually separate real from fake on clearly-separable data)."""

from __future__ import annotations

import numpy as np
import torch

from sim1.models.discriminator import Discriminator
from sim1.motion.motion_lib import AmpMotionLib, ReferenceMotion, ReferenceState
from sim1.tasks.track_rewards import BodyState, amp_features, amp_obs_dim, amp_transition

NB = 15


def _rand_bodies(seed=0, n=4):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-0.3, 0.3, (n, NB, 3)).astype(np.float32) + np.array([0, 1.0, 0], np.float32)
    q = rng.standard_normal((n, NB, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    lin = rng.standard_normal((n, NB, 3)).astype(np.float32)
    ang = rng.standard_normal((n, NB, 3)).astype(np.float32)
    return pos, q, lin, ang


def test_amp_features_layout_parity_sim_vs_reference():
    # SAME body state fed as a sim BodyState and a reference ReferenceState → byte-identical features.
    pos, q, lin, ang = _rand_bodies(1)
    sim = BodyState(pos, q, lin, ang)
    ref = ReferenceState(root_pos=pos[:, 0], root_quat=q[:, 0], body_pos=pos, body_quat=q,
                         body_linvel=lin, body_angvel=ang)
    assert np.array_equal(amp_features(sim), amp_features(ref))


def test_amp_feature_and_transition_dims():
    pos, q, lin, ang = _rand_bodies(2)
    f = amp_features(BodyState(pos, q, lin, ang))
    assert f.shape == (4, NB * 15 + 1)                       # per-frame
    t = amp_transition(BodyState(pos, q, lin, ang), BodyState(pos, q, lin, ang))
    assert t.shape == (4, amp_obs_dim(NB)) == (4, 2 * (NB * 15 + 1))


def _toy_lib(frames=20):
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.uniform(-0.02, 0.02, (frames, NB, 3)), axis=0).astype(np.float32) + [0, 1.0, 0]
    q = rng.standard_normal((frames, NB, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    lin = rng.standard_normal((frames, NB, 3)).astype(np.float32) * 0.1
    ang = rng.standard_normal((frames, NB, 3)).astype(np.float32) * 0.1
    m = ReferenceMotion(fps=30.0, body_names=[str(i) for i in range(NB)], parents=np.zeros(NB, int),
                        body_pos=pos, body_quat=q, body_linvel=lin, body_angvel=ang, local_quat=q)
    return AmpMotionLib().add(m)


def test_sample_amp_transitions_shape_and_finite():
    lib = _toy_lib()
    x = lib.sample_amp_transitions(32, dt=1.0 / 30.0, rng=np.random.default_rng(3))
    assert x.shape == (32, amp_obs_dim(NB))
    assert np.isfinite(x).all()


def test_discriminator_shapes_and_style_reward_bounds():
    d = Discriminator(amp_obs_dim(NB))
    x = torch.randn(16, amp_obs_dim(NB))
    assert d(x).shape == (16,)
    r = d.style_reward(x)
    assert r.shape == (16,) and torch.all(r >= 0) and torch.all(r <= 1)
    assert d.grad_penalty(x).ndim == 0 and float(d.grad_penalty(x)) >= 0.0


def test_discriminator_learns_to_separate():
    # Real ~ N(+2, I), fake ~ N(-2, I): a few LSGAN+GP steps should separate them (accuracy ↑, loss ↓).
    torch.manual_seed(0)
    dim = amp_obs_dim(NB)
    d = Discriminator(dim, hidden_sizes=(64, 64))
    opt = torch.optim.Adam(d.parameters(), lr=1e-3)
    real_mean, fake_mean = 2.0, -2.0
    acc0 = d.accuracy(torch.randn(256, dim) + real_mean, torch.randn(256, dim) + fake_mean)
    for _ in range(150):
        real = torch.randn(256, dim) + real_mean
        fake = torch.randn(256, dim) + fake_mean
        loss = d.lsgan_loss(real, fake) + 10.0 * d.grad_penalty(real)
        opt.zero_grad(); loss.backward(); opt.step()
    acc1 = d.accuracy(torch.randn(256, dim) + real_mean, torch.randn(256, dim) + fake_mean)
    assert acc1 > 0.9 and acc1 > acc0
    # style reward: higher for real-like than fake-like inputs
    assert float(d.style_reward(torch.randn(256, dim) + real_mean).mean()) > \
           float(d.style_reward(torch.randn(256, dim) + fake_mean).mean())
