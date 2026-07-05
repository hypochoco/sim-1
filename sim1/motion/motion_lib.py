"""MotionLib — load ASE `SkeletonMotion` clips and express them in our engine's rig frame.

Step 1 of the mocap tracking plan (`notes/investigations/2026-07-05-mocap-tracking-plan.md`). Pure
NumPy, no engine / torch dependency. The ASE clips (`humanoid-motion/ASE/ase/data/motions/*.npy`) are
already retargeted to the **amp_humanoid** topology, and their 15-body order is **identical** to our
`makeAMPHumanoid` body order (pelvis, torso, head, R-arm×3, L-arm×3, R-leg×3, L-leg×3), so the only
work is:

  1. **Forward kinematics** — the clip stores per-joint *local* rotations (`is_local`) + a rest
     skeleton (`local_translation`, `parent_indices`) + the per-frame `root_translation`. We compose
     them into per-body **global** transforms.
  2. **Frame change** — ASE is Z-up; our engine is Y-up with the map our=(amp.y, amp.z, amp.x) used to
     author the rig (see 2026-07-04-humanoid-rig-adoption.md). That map is the cyclic permutation
     P: (x,y,z)→(y,z,x), a proper rotation (det +1), so positions/velocities map as P·v and rotations
     as P·R·Pᵀ.
  3. **Quaternion convention** — ASE stores xyzw (scalar-last); our engine uses wxyz. Converted here.

Output (in OUR frame, wxyz quats): per-frame root pose/twist and per-body pos/quat/linvel/angvel,
plus continuous-time interpolation (`state_at`) and RSI sampling (`sample_times`). The reduced
generalized-coordinate (joint q/qd) conversion needed for the eventual RSI state-set is deferred to
when that engine API lands (it defines the exact expected representation)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ASE→ours basis change: our=(amp.y, amp.z, amp.x). P·v performs it; P is a proper rotation.
_P = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

# Our engine's articulation body order (makeAMPHumanoid) is **feet-last**: both feet are pulled to the
# end (…thighR, shinR, thighL, shinL, footR, footL), whereas the ASE clip interleaves them
# (…thighR, shinR, footR, thighL, shinL, footL). We reindex the clip into THIS order (matched by name)
# so every consumer — the viz replay AND the tracking reward vs `env.body_pos` — is body-aligned.
# Validated against the engine's authored geometry in tests/test_motion_lib.py.
_OUR_BODY_ORDER = [
    "pelvis", "torso", "head",
    "right_upper_arm", "right_lower_arm", "right_hand",
    "left_upper_arm", "left_lower_arm", "left_hand",
    "right_thigh", "right_shin", "left_thigh", "left_shin",
    "right_foot", "left_foot",
]

# Our rig's authored joint anchors (the parent-joint world position for each body, rest pose, our
# Y-up frame), transcribed from makeAMPHumanoid's addJoint() calls — index-aligned to _OUR_BODY_ORDER.
# Used by retarget_to_rig for proper articulated FK (place each body from its joint). Row 0 (pelvis =
# root) is unused. A test validates each anchor lies between its parent and child body centers.
_RIG_JOINT_ANCHORS = np.array([
    [0.0, 0.0, 0.0],          # pelvis (root — unused)
    [0.0, 1.118, 0.0],        # torso  ← abdomen
    [0.0, 1.342, 0.0],        # head   ← neck
    [-0.183, 1.362, -0.024],  # right_upper_arm ← shoulderR
    [-0.183, 1.087, -0.024],  # right_lower_arm ← elbowR
    [-0.183, 0.828, -0.024],  # right_hand      ← wristR
    [0.183, 1.362, -0.024],   # left_upper_arm  ← shoulderL
    [0.183, 1.087, -0.024],   # left_lower_arm  ← elbowL
    [0.183, 0.828, -0.024],   # left_hand       ← wristL
    [-0.085, 0.882, 0.0],     # right_thigh ← hipR
    [-0.085, 0.460, 0.0],     # right_shin  ← kneeR
    [0.085, 0.882, 0.0],      # left_thigh  ← hipL
    [0.085, 0.460, 0.0],      # left_shin   ← kneeL
    [-0.085, 0.050, 0.0],     # right_foot  ← ankleR
    [0.085, 0.050, 0.0],      # left_foot   ← ankleL
])

# Default location of the ASE motions, resolved relative to this file (…/research/sim-1/sim1/motion → …/research).
_ASE_MOTIONS = Path(__file__).resolve().parents[3] / "humanoid-motion" / "ASE" / "ase" / "data" / "motions"


def default_motion_path(name: str = "amp_humanoid_walk") -> Path:
    """Resolve a clip by name. `$SIM1_MOTIONS_DIR` overrides the default sibling ASE location — set it
    on machines where the ASE checkout isn't a sibling of the sim-1 repo (e.g. a training box)."""
    base = Path(os.environ["SIM1_MOTIONS_DIR"]) if os.environ.get("SIM1_MOTIONS_DIR") else _ASE_MOTIONS
    return base / f"{name}.npy"


# --- quaternion helpers (all wxyz unless noted) ------------------------------------------------
def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw unit quats → (..., 3, 3) rotation matrices."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - w * z); m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y); m[..., 2, 1] = 2 * (y * z + w * x); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices → (..., 4) wxyz unit quats (branch-per-element, vectorized)."""
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    w = np.sqrt(np.maximum(0.0, 1.0 + t)) / 2.0
    x = np.sqrt(np.maximum(0.0, 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2.0
    y = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2.0
    z = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2.0
    x = np.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    y = np.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    z = np.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    q = np.stack([w, x, y, z], axis=-1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _quat_slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Batched slerp between wxyz quats a,b (…,4) by t (…,)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b); dot = np.abs(dot)
    t = t[..., None]
    # near-parallel → linear (avoid div-by-zero), else spherical
    lin = a + t * (b - a)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    st = np.sin(theta)
    sph = (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.where(st < 1e-6, 1.0, st)
    out = np.where(dot > 0.9995, lin, sph)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by wxyz quats q. q (...,4); v (3,) or (...,3), broadcast → (...,3)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.broadcast_to(np.asarray(v, dtype=np.float64), q.shape[:-1] + (3,))
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


# --- data containers ---------------------------------------------------------------------------
@dataclass
class ReferenceState:
    """A batch of reference states (N frames), all in OUR rig frame, wxyz quats."""
    root_pos: np.ndarray      # (N, 3)
    root_quat: np.ndarray     # (N, 4)
    body_pos: np.ndarray      # (N, B, 3)
    body_quat: np.ndarray     # (N, B, 4)
    body_linvel: np.ndarray   # (N, B, 3)
    body_angvel: np.ndarray   # (N, B, 3)


@dataclass
class ReferenceMotion:
    fps: float
    body_names: list[str]
    parents: np.ndarray       # (B,)
    body_pos: np.ndarray      # (F, B, 3)   per-frame global body position (our frame)
    body_quat: np.ndarray     # (F, B, 4)   per-frame global body orientation (wxyz, our frame)
    body_linvel: np.ndarray   # (F, B, 3)
    body_angvel: np.ndarray   # (F, B, 3)
    local_quat: np.ndarray    # (F, B, 4)   per-joint LOCAL rotation (wxyz, our frame) — for joint-angle work

    @property
    def num_frames(self) -> int:
        return self.body_pos.shape[0]

    @property
    def num_bodies(self) -> int:
        return self.body_pos.shape[1]

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps


def _unwrap(v):
    """poselib serializes tensors as {'arr': ndarray, 'context': ...}; unwrap to the array."""
    if isinstance(v, dict) and "arr" in v:
        return np.asarray(v["arr"], dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def load_reference_motion(path: str | Path) -> ReferenceMotion:
    """Load an ASE SkeletonMotion .npy and express it in our rig frame (Y-up, wxyz)."""
    obj = np.load(Path(path), allow_pickle=True).item()
    if obj.get("__name__") != "SkeletonMotion":
        raise ValueError(f"{path}: not a SkeletonMotion (got {obj.get('__name__')!r})")

    local_xyzw = _unwrap(obj["rotation"])          # (F, B, 4) local joint rotations, xyzw
    root_trans = _unwrap(obj["root_translation"])  # (F, 3)
    lin_w = _unwrap(obj["global_velocity"])         # (F, B, 3) world linear velocity (ASE frame)
    ang_w = _unwrap(obj["global_angular_velocity"]) # (F, B, 3) world angular velocity (ASE frame)
    fps = float(obj["fps"])

    st = obj["skeleton_tree"]
    names = list(st["node_names"])
    parents = np.asarray(_unwrap(st["parent_indices"]), dtype=np.int64)
    rest_off = _unwrap(st["local_translation"])     # (B, 3) rest offset of each body from its parent
    F, B = local_xyzw.shape[0], local_xyzw.shape[1]

    # 1) forward kinematics in the ASE frame: compose local rotations along the tree.
    Rl = _quat_xyzw_to_matrix(local_xyzw)           # (F, B, 3, 3)
    Rg = np.empty_like(Rl)
    tg = np.empty((F, B, 3), dtype=np.float64)
    Rg[:, 0] = Rl[:, 0]
    tg[:, 0] = root_trans                            # root uses per-frame translation, not rest offset
    for j in range(1, B):                            # parents[j] < j for this skeleton → single pass
        p = parents[j]
        Rg[:, j] = Rg[:, p] @ Rl[:, j]
        tg[:, j] = tg[:, p] + np.einsum("fik,k->fi", Rg[:, p], rest_off[j])

    # 2) frame change ASE→ours (positions/velocities: P·v ; rotations: P·R·Pᵀ).
    tg = tg @ _P.T
    lin = lin_w @ _P.T
    ang = ang_w @ _P.T
    Rg = np.einsum("ij,fbjk,lk->fbil", _P, Rg, _P)
    Rl_our = np.einsum("ij,fbjk,lk->fbil", _P, Rl, _P)

    # 3) reindex ASE node order → our engine body order (feet-last), matched by name.
    perm = np.array([names.index(n) for n in _OUR_BODY_ORDER], dtype=np.int64)
    inv = {int(a): k for k, a in enumerate(perm)}                     # ase index → our index
    our_parents = np.array([-1 if parents[a] < 0 else inv[int(parents[a])] for a in perm], dtype=np.int64)

    return ReferenceMotion(
        fps=fps, body_names=list(_OUR_BODY_ORDER), parents=our_parents,
        body_pos=tg[:, perm], body_quat=_matrix_to_quat_wxyz(Rg)[:, perm],
        body_linvel=lin[:, perm], body_angvel=ang[:, perm],
        local_quat=_matrix_to_quat_wxyz(Rl_our)[:, perm],
    )


class AmpMotionLib:
    """MotionLib over one (extensible to many) ASE clip retargeted to our rig frame."""

    def __init__(self) -> None:
        self._clips: list[ReferenceMotion] = []

    def load(self, path: str | Path) -> "AmpMotionLib":
        self._clips.append(load_reference_motion(path))
        return self

    def add(self, motion: ReferenceMotion) -> "AmpMotionLib":
        """Append an already-built (e.g. retargeted) clip."""
        self._clips.append(motion)
        return self

    def num_clips(self) -> int:
        return len(self._clips)

    def clip(self, i: int = 0) -> ReferenceMotion:
        return self._clips[i]

    def sample_times(self, n: int, rng: np.random.Generator, clip: int = 0) -> np.ndarray:
        """n uniform random times in [0, duration] — the RSI phase samples."""
        return rng.uniform(0.0, self._clips[clip].duration, size=n)

    def state_at(self, times: np.ndarray, clip: int = 0) -> ReferenceState:
        """Reference state at continuous `times` (seconds), linearly/slerp-interpolated between frames."""
        m = self._clips[clip]
        times = np.clip(np.asarray(times, dtype=np.float64), 0.0, m.duration)
        f = times * m.fps
        i0 = np.floor(f).astype(np.int64)
        i0 = np.clip(i0, 0, m.num_frames - 2)
        i1 = i0 + 1
        a = (f - i0)[:, None, None]                  # (N,1,1) blend, broadcast over bodies/coords

        def lerp(arr):
            return (1 - a) * arr[i0] + a * arr[i1]

        body_pos = lerp(m.body_pos)
        body_lin = lerp(m.body_linvel)
        body_ang = lerp(m.body_angvel)
        aq = (f - i0)[:, None]                        # (N,1) for per-body slerp reshaped below
        N, Bn = times.shape[0], m.num_bodies
        bq = _quat_slerp(m.body_quat[i0].reshape(N * Bn, 4),
                         m.body_quat[i1].reshape(N * Bn, 4),
                         np.repeat(f - i0, Bn)).reshape(N, Bn, 4)
        return ReferenceState(
            root_pos=body_pos[:, 0], root_quat=bq[:, 0],
            body_pos=body_pos, body_quat=bq, body_linvel=body_lin, body_angvel=body_ang,
        )


def retarget_to_rig(motion: ReferenceMotion, rest_body_pos: np.ndarray, ground: bool = True) -> ReferenceMotion:
    """Rotation-based retarget: reproduce the motion on OUR rig's proportions.

    Uses only the reference **orientations** (per-body global rotation) + the reference root position,
    and rebuilds body positions by proper articulated forward kinematics — each body's center is placed
    from its **joint** (not the parent's center): `center[j] = center[par] + R[par]·(par_center→joint) +
    R[j]·(joint→center[j])`. This makes limbs pivot about their joints (thighs swing from the hip) and
    lands bodies at our body centers, fixing the "shifted-down" look from replaying ASE joint positions.

    `rest_body_pos` (B,3) is our rig's authored body-center positions (from the engine at reset), our
    frame + body order. Joint anchors are our rig's authored joint positions (`_RIG_JOINT_ANCHORS`).
    If `ground`, the clip is shifted vertically so its lowest foot matches the reference's lowest foot
    (our longer legs would otherwise sink through the floor)."""
    rest = np.asarray(rest_body_pos, dtype=np.float64)
    anch = _RIG_JOINT_ANCHORS
    parents = motion.parents
    F, B = motion.num_frames, motion.num_bodies
    a = np.zeros((B, 3))   # parent center → joint  (expressed in parent rest frame = world at rest)
    b = np.zeros((B, 3))   # joint → child center   (expressed in child rest frame)
    for j in range(B):
        if parents[j] >= 0:
            a[j] = anch[j] - rest[parents[j]]
            b[j] = rest[j] - anch[j]

    Rg = motion.body_quat                             # (F,B,4) reference global orientations (our frame)
    p = np.empty((F, B, 3), dtype=np.float64)
    p[:, 0] = motion.body_pos[:, 0]                   # root (pelvis center) follows the reference root
    for j in range(1, B):                             # parents[j] < j → single forward pass
        par = int(parents[j])
        p[:, j] = p[:, par] + _quat_rotate_wxyz(Rg[:, par], a[j]) + _quat_rotate_wxyz(Rg[:, j], b[j])

    if ground:
        feet = [i for i, n in enumerate(motion.body_names) if n.endswith("_foot")]
        shift = float(motion.body_pos[:, feet, 1].min() - p[:, feet, 1].min())
        p[:, :, 1] += shift                           # plant our (taller) rig at the reference's foot level

    lin = np.gradient(p, 1.0 / motion.fps, axis=0)    # our-proportion linear velocity (finite diff)
    return ReferenceMotion(
        fps=motion.fps, body_names=motion.body_names, parents=parents,
        body_pos=p, body_quat=Rg, body_linvel=lin,
        body_angvel=motion.body_angvel, local_quat=motion.local_quat,
    )
