"""StandTask — keep the physically-simulated humanoid upright.

Now expressed on the composable base (`CompositeTask`): proprioception observation (no command),
reward = alive + upright + stay-near-standing-height − control, and fall termination. This is
**behaviorally identical** to the original hand-rolled version — same observation, same reward, same
termination, same action mapping — so weights trained by the previous implementation remain fully
usable (a test pins the reward to its closed form). Actuation: policy outputs (~unit) × `action_scale`
(= `max_torque` in torque mode, ≈ radians in PD-target mode); the standing reference height is
captured from the rig's authored pose at reset (rig-agnostic).
"""

from __future__ import annotations

from sim1.tasks.command import NoCommand
from sim1.tasks.composite import CompositeTask
from sim1.tasks.rewards import (
    RewardTerm,
    fall_termination,
    term_alive,
    term_ctrl_penalty,
    term_height,
    term_upright,
)


class StandTask(CompositeTask):
    def __init__(
        self,
        ndof: int,
        nbody: int,
        act_dim: int,
        action_scale: float,
        upright_weight: float = 1.0,
        height_weight: float = 1.0,
        alive_bonus: float = 1.0,
        action_weight: float = 0.01,
        fall_height_frac: float = 0.5,
        upright_fall: float = 0.3,
    ):
        super().__init__(
            ndof,
            nbody,
            act_dim,
            action_scale,
            command=NoCommand(),
            reward_terms=[
                RewardTerm("alive", alive_bonus, term_alive),
                RewardTerm("upright", upright_weight, term_upright),
                RewardTerm("height", height_weight, term_height),
                RewardTerm("ctrl", -action_weight, term_ctrl_penalty),
            ],
            terminate_fn=fall_termination(fall_height_frac, upright_fall),
            command_weight=0.0,
        )
