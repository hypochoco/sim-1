"""GetUpTask — stand *and recover*: get up off the ground and back up after falling.

Identical observation and reward to `StandTask`, but with **no fall termination** — when the humanoid
tips over the episode keeps going, so maximizing return over a long episode requires pushing back up
to standing. Trains recovery from self-induced falls with zero engine changes; a strict superset of
`stand` and a clean warm-start target from a stand policy (same obs/action, so weights load directly).
Use a long episode (`-o env.episode_len=1000`) so there's time to fall and recover repeatedly.

Limitation (v1): without an engine/binding hook we can't reset to random *fallen* poses
(reference-state-init) or apply external shoves ("knocked over") during training — both need a small
engine addition (expose `Environment::setResetHook` / an initial-state randomizer / an impulse call).
Until then the fallen states learned from are the ones the policy stumbles into.
"""

from __future__ import annotations

from sim1.tasks.rewards import never_terminate
from sim1.tasks.stand import StandTask


class GetUpTask(StandTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminate_fn = never_terminate   # recover instead of ending the episode on a fall
