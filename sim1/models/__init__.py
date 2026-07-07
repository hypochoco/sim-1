"""Neural network models: the Gaussian actor-critic policy + the AMP discriminator."""

from sim1.models.mlp import ActorCritic
from sim1.models.discriminator import Discriminator

__all__ = ["ActorCritic", "Discriminator"]
