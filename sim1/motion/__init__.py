"""sim1.motion — reference-motion (mocap) support for the imitation milestone.

Currently interface-only (`reference.py`): the seam by which mocap plugs into the composable task
system (a reference `ObsComponent` + an imitation `RewardTerm`). Loader/retargeting land later.
"""

from sim1.motion.reference import ImitationReward, MotionLib, ReferenceObs
from sim1.motion.motion_lib import (
    AmpMotionLib,
    ReferenceMotion,
    ReferenceState,
    default_motion_path,
    load_reference_motion,
    retarget_to_rig,
)

__all__ = [
    "MotionLib", "ReferenceObs", "ImitationReward",
    "AmpMotionLib", "ReferenceMotion", "ReferenceState",
    "load_reference_motion", "default_motion_path", "retarget_to_rig",
]
