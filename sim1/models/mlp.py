"""Gaussian actor-critic MLP for continuous control (CleanRL-style).

Separate policy and value trunks; a state-independent learnable log-std for the diagonal Gaussian
policy; orthogonal init with a small final-layer gain on the policy mean and unit gain on the value
head. Suitable for the mock task now and the humanoid later.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_std: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [_layer_init(nn.Linear(last, h)), nn.Tanh()]
        last = h
    layers.append(_layer_init(nn.Linear(last, out_dim), std=out_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        hidden = tuple(hidden_sizes)
        self.actor_mean = _mlp(obs_dim, hidden, act_dim, out_std=0.01)
        self.critic = _mlp(obs_dim, hidden, 1, out_std=1.0)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, logprob, entropy, value
