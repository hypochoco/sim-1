"""Proprioception — the canonical body-state observation block, shared by every humanoid task.

This is the transfer-stable core of the observation: it is **identical across stand / getup / walk /
combat**, so a policy's input trunk stays aligned and weights warm-start cleanly between scenarios
(task-specific channels — goals, references — are *appended after* this block; see `composite.py`).
Byte-identical to the layout `StandTask.observe` established (a test guards this equality).

Layout (per env): `[root_height(1) | root_quat_wxyz(4) | root_linvel(3) | root_angvel(3) |
joint_q[ndof] | joint_qd[ndof] | contact_flags[nbody]]`.
"""

from __future__ import annotations

import numpy as np

from sim1.envs.vecenv import VecEnv


def _rot_dim(rot: str) -> int:
    return 6 if rot == "sixd" else 4


def quat_to_6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) → the continuous 6D rotation representation (Zhou et al. 2019): the
    first two columns of the rotation matrix, shape (n, 6). Continuous over SO(3) (no double-cover /
    gimbal discontinuity), so nets learn rotation-dependent functions better than from quats/Euler.
    Reconstruct a rotation by Gram-Schmidt on the two 3-vectors."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    # columns 0 and 1 of R(q)
    c0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
    c1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1)
    return np.concatenate([c0, c1], axis=1).astype(np.float32)


def proprio_dim(ndof: int, nbody: int, rot: str = "quat") -> int:
    return 1 + _rot_dim(rot) + 3 + 3 + 2 * int(ndof) + int(nbody)


def proprio_obs(env: VecEnv, rot: str = "quat") -> np.ndarray:
    """The canonical body-state block, shape (num_envs, proprio_dim). Translation-invariant (drops
    root x,z; keeps height). `rot` selects the root-orientation encoding: `quat` (w,x,y,z) or the
    continuous `sixd` (6D)."""
    q = env.root_pose[:, 3:7]
    rot_feat = quat_to_6d(q) if rot == "sixd" else q
    return np.concatenate(
        [
            env.root_pose[:, 1:2],   # height
            rot_feat,                # root orientation (quat 4 | 6D 6)
            env.root_twist[:, 0:3],  # linear velocity
            env.root_twist[:, 3:6],  # angular velocity
            env.joint_q,
            env.joint_qd,
            env.contact_flags,
        ],
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
    """Heading (yaw) about world +Y from quat (w, x, y, z)."""
    w, x, y, z = env.root_pose[:, 3], env.root_pose[:, 4], env.root_pose[:, 5], env.root_pose[:, 6]
    return np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
