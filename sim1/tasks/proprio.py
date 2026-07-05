"""Proprioception — the canonical body-state observation block, shared by every humanoid task.

Transfer-stable core of the observation: identical across stand / getup / walk / combat, so a policy's
input trunk stays aligned and warm-starts cleanly (task-specific channels — goals, references — are
appended after this block; see `composite.py`).

Layout (per env): `[root_height(1) | root_orientation(4 quat | 6 sixd) | root_linvel(3) |
root_angvel(3) | joint_q[ndof] | joint_qd[ndof] | contact_flags[nbody]]`.

Two representation options, both dim-preserving where noted:
- **rotation** (`quat` | `sixd`): how the root orientation is encoded (quat 4 / continuous 6D 6).
- **frame** (`world` | `local`): reference frame of the root features. `world` (default) = raw engine
  frame — byte-identical to the original layout. `local` = the character's **heading frame**: root
  linear+angular velocity are rotated into the heading frame and the orientation has its yaw removed,
  making the observation **invariant to world heading** (the key generalization win for locomotion /
  user-steered walk). `frame` does NOT change obs_dim — only the values.
"""

from __future__ import annotations

import numpy as np

from sim1.envs.vecenv import VecEnv


def _rot_dim(rot: str) -> int:
    return 6 if rot == "sixd" else 4


def proprio_dim(ndof: int, nbody: int, rot: str = "quat") -> int:
    return 1 + _rot_dim(rot) + 3 + 3 + 2 * int(ndof) + int(nbody)


# --- rotation / heading-frame helpers --------------------------------------------------------
def _yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """Heading angle from the character's forward axis (body local +Z) projected onto the ground,
    shape (n,). Unlike an Euler-yaw formula, this adds exactly δ under an added world-Y rotation
    (the forward vector co-rotates), which is what makes the heading frame truly yaw-invariant.
    Degenerate only when forward points near straight up/down (extreme tilt)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fwd_x = 2.0 * (x * z + w * y)      # world-space body +Z (third column of R(q)), x component
    fwd_z = 1.0 - 2.0 * (x * x + y * y)  # ... z component
    return np.arctan2(fwd_x, fwd_z)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (w, x, y, z), batched."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1).astype(np.float32)


def rotate_to_heading(vx: np.ndarray, vz: np.ndarray, yaw: np.ndarray):
    """Express world (x, z) vector components in the character's heading frame (rotate by −yaw about
    +Y). Heading-invariant: when the world vector co-rotates with the heading, the result is fixed."""
    c, s = np.cos(yaw), np.sin(yaw)
    return c * vx - s * vz, s * vx + c * vz


def remove_yaw(q: np.ndarray) -> np.ndarray:
    """Orientation with its heading removed: q_heading⁻¹ ⊗ q, leaving only tilt (roll/pitch). Invariant
    to additional world yaw (left-multiplication by a yaw quaternion)."""
    half = 0.5 * _yaw_from_quat(q)
    zeros = np.zeros_like(half)
    qh_inv = np.stack([np.cos(half), zeros, -np.sin(half), zeros], axis=1).astype(np.float32)  # conj(q_heading)
    return _quat_mul(qh_inv, q)


def quat_to_6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) → the continuous 6D rotation representation (Zhou et al. 2019): the
    first two columns of the rotation matrix, shape (n, 6). Continuous over SO(3) (no double-cover /
    gimbal discontinuity). Reconstruct by Gram-Schmidt on the two 3-vectors."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    c0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
    c1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1)
    return np.concatenate([c0, c1], axis=1).astype(np.float32)


def proprio_obs(env: VecEnv, rot: str = "quat", frame: str = "world") -> np.ndarray:
    """The canonical body-state block, shape (num_envs, proprio_dim). Translation-invariant (drops
    root x,z; keeps height). `rot` = orientation encoding; `frame` = `world` (raw) | `local` (heading
    frame: velocities rotated, orientation yaw-removed → heading-invariant)."""
    q = env.root_pose[:, 3:7]
    lv = env.root_twist[:, 0:3].astype(np.float32).copy()
    av = env.root_twist[:, 3:6].astype(np.float32).copy()
    if frame == "local":
        yaw = _yaw_from_quat(q)
        q = remove_yaw(q)
        lv[:, 0], lv[:, 2] = rotate_to_heading(lv[:, 0], lv[:, 2], yaw)
        av[:, 0], av[:, 2] = rotate_to_heading(av[:, 0], av[:, 2], yaw)
    rot_feat = quat_to_6d(q) if rot == "sixd" else q
    return np.concatenate(
        [env.root_pose[:, 1:2], rot_feat, lv, av, env.joint_q, env.joint_qd, env.contact_flags],
        axis=1,
    ).astype(np.float32)


def uprightness(env: VecEnv) -> np.ndarray:
    """World up-component of the torso's local +Y for quat (w, x, y, z): 1 - 2(x^2 + z^2)."""
    x = env.root_pose[:, 4]
    z = env.root_pose[:, 6]
    return 1.0 - 2.0 * (x * x + z * z)


def root_height(env: VecEnv) -> np.ndarray:
    return env.root_pose[:, 1]


def planar_velocity(env: VecEnv) -> np.ndarray:
    """Root planar (x, z) linear velocity, shape (num_envs, 2)."""
    return np.stack([env.root_twist[:, 0], env.root_twist[:, 2]], axis=1)


def root_yaw(env: VecEnv) -> np.ndarray:
    """Heading (yaw) about world +Y, shape (num_envs,)."""
    return _yaw_from_quat(env.root_pose[:, 3:7])


# --- per-body 6D (SuperTrack tracking representation) ---------------------------------------
# The reusable tracking base's proprioception: EVERY rigid body expressed in the root's frame, so
# the whole feature set is invariant to the character's global position AND orientation (SuperTrack
# §"State representation"). Per body: root-relative position, 6D rotation, linear + angular velocity,
# and world height above ground; plus a single local up-vector (root-frame gravity/tilt cue — what
# re-introduces the orientation information that the root-relative transform removes).
#
# This is a DIFFERENT, richer proprioception than `proprio_obs` (which is root-only + reduced joint
# coords). It is additive + config-gated (default off) so existing stand/getup/walk weights are
# untouched; it composes into the canonical obs order as an appended component (see BodyFeatures).

# per-body channels: pos(3) + rot(6D=6|quat=4) + linvel(3) + angvel(3) + height(1)
def _per_body_rot_dim(rot: str) -> int:
    return 6 if rot == "sixd" else 4


def per_body_dim(nbody: int, rot: str = "sixd") -> int:
    """Width of the per-body block: nbody · (3 + rot + 3 + 3 + 1) + 3 (the shared local up-vector)."""
    return int(nbody) * (3 + _per_body_rot_dim(rot) + 3 + 3 + 1) + 3


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternions) of (w, x, y, z), batched (M, 4)."""
    out = q.astype(np.float32).copy()
    out[:, 1:] *= -1.0
    return out


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (M, 3) by unit quaternions q (M, 4) — v' = v + 2w(u×v) + 2u×(u×v)."""
    w = q[:, 0:1]
    u = q[:, 1:4]
    uv = np.cross(u, v)
    return (v + 2.0 * (w * uv + np.cross(u, uv))).astype(np.float32)


def per_body_obs(env: VecEnv, rot: str = "sixd") -> np.ndarray:
    """SuperTrack per-body feature block, shape (num_envs, per_body_dim(nbody, rot)).

    Each body is expressed in the ROOT frame (root-relative position + orientation rotated by the
    root's inverse), making the block invariant to the character's world position and heading; the
    appended local up-vector carries the absolute tilt/gravity cue. `rot` selects the per-body
    orientation encoding (SuperTrack uses 6D)."""
    bp = np.asarray(env.body_pos, dtype=np.float32)      # (N, B, 3) world position
    bq = np.asarray(env.body_quat, dtype=np.float32)     # (N, B, 4) world orientation (w,x,y,z)
    blv = np.asarray(env.body_linvel, dtype=np.float32)  # (N, B, 3) world linear velocity
    bav = np.asarray(env.body_angvel, dtype=np.float32)  # (N, B, 3) world angular velocity
    n, b = bp.shape[0], bp.shape[1]

    p_r = env.root_pose[:, None, 0:3]                    # (N, 1, 3) root position
    q_r = env.root_pose[:, 3:7].astype(np.float32)       # (N, 4) root orientation
    qr_inv_flat = np.repeat(quat_conj(q_r), b, axis=0)   # (N*B, 4) root inverse, per body

    def _rot_flat(vec):   # rotate a (N,B,3) field by the root inverse, return (N,B,3)
        return quat_rotate(qr_inv_flat, vec.reshape(n * b, 3)).reshape(n, b, 3)

    loc_pos = _rot_flat(bp - p_r)                        # (N, B, 3) root-relative position
    loc_lv = _rot_flat(blv)                              # (N, B, 3) velocity in root frame
    loc_av = _rot_flat(bav)                              # (N, B, 3)
    loc_q = _quat_mul(qr_inv_flat, bq.reshape(n * b, 4)) # (N*B, 4) orientation in root frame
    loc_rot = (quat_to_6d(loc_q) if rot == "sixd" else loc_q).reshape(n, b, -1)
    height = bp[:, :, 1:2]                               # (N, B, 1) world height above ground (y=0)

    per = np.concatenate([loc_pos, loc_rot, loc_lv, loc_av, height], axis=2).reshape(n, -1)
    world_up = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (n, 1))
    local_up = quat_rotate(quat_conj(q_r), world_up)     # (N, 3) gravity direction in root frame
    return np.concatenate([per, local_up], axis=1).astype(np.float32)


class BodyFeatures:
    """An `ObsComponent` (composite.py seam) that appends the per-body 6D block. Added to a task's
    `extra_obs` when `task.body_obs` is on — keeps the canonical obs order
    `[proprio | command | per-body...]` and is dim-additive (default-off = no change to existing
    policies)."""

    def __init__(self, nbody: int, rot: str = "sixd"):
        self.nbody = int(nbody)
        self.rot = rot
        self.dim = per_body_dim(self.nbody, rot)

    def observe(self, env: VecEnv, task) -> np.ndarray:
        return env.compose_body()   # single source (C++ engine / Python oracle); rot is always sixd
