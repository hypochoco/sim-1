"""WalkTask — goal-conditioned locomotion, built entirely from the composable parts.

Reward = alive + upright + stay-tall − control  +  command (track the target planar velocity).
Because the goal is a `HeadingSpeedCommand`, the trained policy is **directly user-controllable**:
at inference, feed the user's chosen direction/speed into the command channels instead of sampling.

Transfer: WalkTask's observation is `[proprio | (vx, vz)]`, i.e. the exact stand/getup proprio block
plus 2 command channels — so a stand/getup policy warm-starts it with `--init-from` (the shared trunk
loads; only the widened input layer re-inits). This is the stand → getup → walk curriculum.

Naturalness (later): add an AMP-style discriminator term and/or a reference obs component from the
mocap library (see `sim1/motion/`), which slot in as another RewardTerm / ObsComponent — no other
changes.
"""

from __future__ import annotations

from sim1.tasks.command import HeadingSpeedCommand
from sim1.tasks.composite import CompositeTask
from sim1.tasks.rewards import (
    RewardTerm,
    fall_termination,
    term_alive,
    term_ctrl_penalty,
    term_height,
    term_upright,
)


class WalkTask(CompositeTask):
    def __init__(
        self,
        ndof: int,
        nbody: int,
        act_dim: int,
        action_scale: float,
        *,
        target_speed_range: tuple[float, float] = (0.5, 1.5),
        upright_weight: float = 1.0,
        height_weight: float = 0.5,
        alive_bonus: float = 1.0,
        action_weight: float = 0.01,
        command_weight: float = 1.0,
        fall_height_frac: float = 0.5,
        upright_fall: float = 0.3,
        rot_repr: str = "quat",
    ):
        super().__init__(
            ndof,
            nbody,
            act_dim,
            action_scale,
            command=HeadingSpeedCommand(speed_range=target_speed_range),
            reward_terms=[
                RewardTerm("alive", alive_bonus, term_alive),
                RewardTerm("upright", upright_weight, term_upright),
                RewardTerm("height", height_weight, term_height),
                RewardTerm("ctrl", -action_weight, term_ctrl_penalty),
            ],
            terminate_fn=fall_termination(fall_height_frac, upright_fall),
            command_weight=command_weight,
            rot_repr=rot_repr,
        )
