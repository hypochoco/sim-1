"""sim1.motion — reference-motion (mocap) support for the imitation milestone.

Currently interface-only (`reference.py`): the seam by which mocap plugs into the composable task
system (a reference `ObsComponent` + an imitation `RewardTerm`). Loader/retargeting land later.
"""

from sim1.motion.reference import ImitationReward, MotionLib, ReferenceObs

__all__ = ["MotionLib", "ReferenceObs", "ImitationReward"]
