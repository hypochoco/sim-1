"""AMP discriminator (adversarial motion prior).

An MLP that scores an AMP transition (two consecutive frames of heading-frame body features) as
"looks like the reference mocap" vs "produced by the policy". Trained adversarially against the
policy; its score becomes a **style reward** that pulls the policy toward natural, mocap-like motion —
the principled fix for e.g. the stiff-arm look that a hand-tuned tracking reward under-penalizes.

We use the **least-squares GAN (LSGAN)** objective from the AMP paper (Peng et al. 2021): real → +1,
fake → −1, with a **gradient penalty on real samples** for stability. The style reward is the paper's
`r = max(0, 1 − ¼·(D − 1)²)`, bounded in [0, 1].

Training-only: the discriminator is never exported — the deployed policy (actor) is unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Discriminator(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [_layer_init(nn.Linear(last, h)), nn.ReLU()]
            last = h
        layers.append(_layer_init(nn.Linear(last, 1), std=1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)      # raw score (logit-like), (N,)

    @torch.no_grad()
    def style_reward(self, x: torch.Tensor) -> torch.Tensor:
        """AMP LSGAN style reward, bounded [0, 1]: 1 when D thinks it's real (D=1), → 0 otherwise."""
        d = self.forward(x)
        return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)

    def lsgan_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Least-squares GAN: push D(real)→+1, D(fake)→−1."""
        d_real, d_fake = self.forward(real), self.forward(fake)
        return 0.5 * ((d_real - 1.0) ** 2).mean() + 0.5 * ((d_fake + 1.0) ** 2).mean()

    def grad_penalty(self, real: torch.Tensor) -> torch.Tensor:
        """Penalize ||∇_x D(x)||² on real samples (AMP's stabilizer; keeps D from sharpening)."""
        real = real.detach().requires_grad_(True)
        d = self.forward(real)
        grad = torch.autograd.grad(d.sum(), real, create_graph=True)[0]
        return (grad.pow(2).sum(dim=-1)).mean()

    @torch.no_grad()
    def accuracy(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        """Fraction correctly classified by sign (real>0, fake<0) — a training-health readout."""
        correct = (self.forward(real) > 0).float().mean() + (self.forward(fake) < 0).float().mean()
        return float(correct.item() / 2.0)
