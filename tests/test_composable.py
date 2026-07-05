"""Tests for the composable task infrastructure (proprio / command / composite / walk).

The most important assertion is the **transfer invariant**: the canonical proprioception block is
byte-identical to what `StandTask.observe` produces, so a stand/getup policy warm-starts walk (and
future tasks) with an aligned input trunk.
"""

from __future__ import annotations

import numpy as np

from sim1.envs.mock_vecenv import MockVecEnv
from sim1.tasks.command import HeadingSpeedCommand, NoCommand
from sim1.tasks.composite import CompositeTask
from sim1.tasks.proprio import proprio_dim, proprio_obs
from sim1.tasks.rewards import RewardTerm, fall_termination, term_alive, term_upright
from sim1.tasks.stand import StandTask
from sim1.tasks.walk import WalkTask


def _env(n=8, ndof=28):
    return MockVecEnv(num_envs=n, ndof=ndof, seed=0)


def test_proprio_matches_stand_obs():
    # Transfer invariant: the canonical proprio block == StandTask's hand-rolled observation.
    env = _env()
    stand = StandTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0)
    stand.reset(env, 0)
    assert np.array_equal(proprio_obs(env), stand.observe(env))
    assert proprio_dim(env.ndof, 1) == stand.obs_dim


def test_nocommand_composite_is_proprio_only():
    env = _env()
    t = CompositeTask(env.ndof, 1, env.act_dim, 1.0, command=NoCommand())
    t.reset(env, 0)
    assert t.obs_dim == proprio_dim(env.ndof, 1)
    assert t.observe(env).shape == (env.num_envs, t.obs_dim)


def test_heading_speed_command():
    env = _env()
    cmd = HeadingSpeedCommand(speed_range=(0.5, 1.5))
    rng = np.random.default_rng(0)
    g = cmd.sample(env.num_envs, rng)
    assert g.shape == (env.num_envs, 2)
    speeds = np.linalg.norm(g, axis=1)
    assert np.all(speeds >= 0.5 - 1e-5) and np.all(speeds <= 1.5 + 1e-5)
    assert cmd.to_obs(env, g).shape == (env.num_envs, 2)
    # mock env has identity orientation (yaw 0), so root-local == world here
    assert np.allclose(cmd.to_obs(env, g), g)
    r = cmd.reward(env, g)
    assert r.shape == (env.num_envs,) and np.all((r >= 0) & (r <= 1))


def test_stand_behavior_matches_closed_form():
    # Weight-compatibility guard: the migrated (composable) StandTask must reproduce the original
    # observation + reward + termination exactly, so weights from prior runs stay usable.
    from sim1.tasks.proprio import root_height, uprightness
    env = _env()
    uw, hw, ab, aw = 1.3, 0.7, 1.0, 0.02
    t = StandTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0,
                  upright_weight=uw, height_weight=hw, alive_bonus=ab, action_weight=aw,
                  fall_height_frac=0.5, upright_fall=0.3)
    t.reset(env, 0)
    env.step()
    a = np.random.default_rng(1).standard_normal((env.num_envs, env.act_dim)).astype(np.float32)
    up = uprightness(env); h = root_height(env)
    expected = (ab + uw * up + hw * np.exp(-10.0 * (h - t._target_h) ** 2) - aw * np.mean(a ** 2, axis=1))
    assert np.allclose(t.reward(env, a), expected, atol=1e-6)
    assert np.array_equal(proprio_obs(env), t.observe(env))          # obs unchanged
    assert not t.done(env, np.zeros(env.num_envs)).any()             # upright mock → no fall


def test_walk_task_obs_and_reward():
    env = _env()
    t = WalkTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0)
    # obs = proprio + 2 command channels
    assert t.obs_dim == proprio_dim(env.ndof, 1) + 2
    t.reset(env, 0)
    obs = t.observe(env)
    assert obs.shape == (env.num_envs, t.obs_dim)
    r = t.reward(env, np.zeros((env.num_envs, env.act_dim), dtype=np.float32))
    assert r.shape == (env.num_envs,) and np.all(np.isfinite(r))
    # per-term logging populated (alive/upright/height/ctrl/command)
    assert {"alive", "upright", "height", "ctrl", "command"} <= set(t.reward_info)


def test_composite_reset_masked_resamples_subset():
    env = _env()
    t = WalkTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0)
    t.reset(env, 0)
    g0 = t._goals.copy()
    mask = np.zeros(env.num_envs, dtype=bool)
    mask[: env.num_envs // 2] = True
    t.reset_masked(env, mask, seed=1)
    assert not np.array_equal(t._goals[mask], g0[mask])          # masked envs resampled
    assert np.array_equal(t._goals[~mask], g0[~mask])            # others unchanged


def test_set_goal_for_user_control():
    env = _env()
    t = WalkTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0)
    t.reset(env, 0)
    t.set_goal(np.tile(np.array([1.0, 0.0], dtype=np.float32), (env.num_envs, 1)))
    assert np.array_equal(t.observe(env)[:, -2:], np.tile([1.0, 0.0], (env.num_envs, 1)))


def test_position_keeping_penalizes_drift():
    from sim1.tasks.rewards import position_keep_term

    class FakeEnv:
        def __init__(self, xz):
            n = xz.shape[0]
            self.root_pose = np.zeros((n, 7), dtype=np.float32)
            self.root_pose[:, 3] = 1.0
            self.root_pose[:, 0] = xz[:, 0]
            self.root_pose[:, 2] = xz[:, 1]

    class T:
        _start_xz = np.zeros((2, 2), dtype=np.float32)

    fn = position_keep_term(scale=1.0)
    e = FakeEnv(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    r = fn(e, np.zeros((2, 28), dtype=np.float32), T())
    assert r[0] == 1.0                         # at start → full reward
    assert np.isclose(r[1], np.exp(-1.0))      # drift of 1 m → exp(-1); monotonically penalized
    assert r[1] < r[0]


def test_quat_to_6d_identity_and_orthonormality():
    from sim1.tasks.proprio import quat_to_6d
    # identity quaternion → first two rotation-matrix columns = e0, e1
    ident = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    six = quat_to_6d(ident)
    assert np.allclose(six, [[1, 0, 0, 0, 1, 0]], atol=1e-6)
    # random unit quats → the two 3-vectors are unit-length and orthogonal (valid rotation columns)
    rng = np.random.default_rng(0)
    q = rng.standard_normal((16, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    six = quat_to_6d(q)
    c0, c1 = six[:, :3], six[:, 3:]
    assert np.allclose(np.linalg.norm(c0, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(c1, axis=1), 1, atol=1e-5)
    assert np.allclose(np.sum(c0 * c1, axis=1), 0, atol=1e-5)


def test_sixd_changes_obs_dim_not_reward():
    from sim1.tasks.proprio import proprio_dim
    env = _env()
    q = StandTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0, rot_repr="quat")
    s = StandTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0, rot_repr="sixd")
    q.reset(env, 0); s.reset(env, 0)
    env.step()
    assert q.obs_dim == proprio_dim(env.ndof, 1, "quat")
    assert s.obs_dim == proprio_dim(env.ndof, 1, "sixd") == q.obs_dim + 2   # 6D adds 2 channels
    assert s.observe(env).shape == (env.num_envs, s.obs_dim)
    # 6D changes only the observation representation, never the reward/termination
    a = np.zeros((env.num_envs, env.act_dim), dtype=np.float32)
    assert np.allclose(q.reward(env, a), s.reward(env, a))


# --- heading-local frame ---------------------------------------------------------------------
def _roty(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


class _FakeEnv:
    """One env with arbitrary root pose + twist (mock can't set orientation/velocity)."""
    def __init__(self, quat, linvel, angvel, ndof=28, nbody=15):
        self.root_pose = np.zeros((1, 7), dtype=np.float32)
        self.root_pose[0, 1] = 1.0                      # height
        self.root_pose[0, 3:7] = quat
        self.root_twist = np.zeros((1, 6), dtype=np.float32)
        self.root_twist[0, 0:3] = linvel
        self.root_twist[0, 3:6] = angvel
        self.joint_q = np.zeros((1, ndof), dtype=np.float32)
        self.joint_qd = np.zeros((1, ndof), dtype=np.float32)
        self.contact_flags = np.zeros((1, nbody), dtype=np.float32)


def test_world_frame_is_byte_identical_to_raw():
    from sim1.tasks.proprio import proprio_obs
    # a tilted, moving state; world frame must equal the raw concatenation (no transform)
    q = np.array([0.966, 0.259, 0.0, 0.0], dtype=np.float32)  # ~30° roll
    e = _FakeEnv(q, [0.5, 0.0, 1.0], [0.1, 0.2, 0.3])
    raw = np.concatenate([e.root_pose[:, 1:2], e.root_pose[:, 3:7], e.root_twist[:, 0:3],
                          e.root_twist[:, 3:6], e.joint_q, e.joint_qd, e.contact_flags], axis=1)
    assert np.allclose(proprio_obs(e, "quat", "world"), raw)


def test_remove_yaw_is_yaw_invariant_and_keeps_tilt():
    from sim1.tasks.proprio import remove_yaw, uprightness
    tilt = np.array([0.966, 0.259, 0.0, 0.0], dtype=np.float32)      # pure roll (no yaw)
    tilt /= np.linalg.norm(tilt)                                     # ensure unit quaternion
    # pure-yaw quaternion (90° about +Y) applied on top
    yaw_q = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0], dtype=np.float32)
    q_yawed = _qmul(yaw_q, tilt)[None, :]
    # removing yaw from the yawed state recovers the tilt-only orientation
    assert np.allclose(remove_yaw(q_yawed), tilt[None, :], atol=1e-5)
    # a pure-yaw orientation → identity after yaw removal
    assert np.allclose(remove_yaw(yaw_q[None, :]), np.array([[1.0, 0, 0, 0]]), atol=1e-5)


def test_local_frame_proprio_is_heading_invariant():
    from sim1.tasks.proprio import proprio_obs
    q1 = np.array([0.940, 0.171, 0.0, 0.296], dtype=np.float32)   # some tilt + heading
    q1 /= np.linalg.norm(q1)
    lv1 = np.array([0.7, -0.1, 0.4]); av1 = np.array([0.2, 0.5, -0.3])
    e1 = _FakeEnv(q1, lv1, av1)
    # add an arbitrary world heading δ: rotate orientation (left-mult) and world velocities by R_y(δ)
    d = 1.1
    yaw_q = np.array([np.cos(d / 2), 0, np.sin(d / 2), 0], dtype=np.float32)
    q2 = _qmul(yaw_q, q1); q2 /= np.linalg.norm(q2)
    e2 = _FakeEnv(q2, _roty(d) @ lv1, _roty(d) @ av1)
    # heading-local proprio must be identical (invariant to world heading)
    o1 = proprio_obs(e1, "quat", "local")
    o2 = proprio_obs(e2, "quat", "local")
    assert np.allclose(o1, o2, atol=1e-4), np.abs(o1 - o2).max()


# --- per-body 6D (SuperTrack) ----------------------------------------------------------------
class _FakeBodyEnv:
    """An env with arbitrary per-body world state (the mock can't set these)."""
    def __init__(self, root_pose, body_pos, body_quat, body_linvel, body_angvel):
        self.root_pose = root_pose.astype(np.float32)
        self.body_pos = body_pos.astype(np.float32)
        self.body_quat = body_quat.astype(np.float32)
        self.body_linvel = body_linvel.astype(np.float32)
        self.body_angvel = body_angvel.astype(np.float32)

    def compose_body(self):   # mirrors the real envs (BodyFeatures.observe delegates here)
        from sim1.tasks.proprio import per_body_obs
        return per_body_obs(self, "sixd")


def _randq(rng, m):
    q = rng.standard_normal((m, 4)).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _make_body_env(rng, n=2, b=5):
    root = np.concatenate([rng.standard_normal((n, 3)), _randq(rng, n)], axis=1).astype(np.float32)
    return _FakeBodyEnv(root, rng.standard_normal((n, b, 3)), _randq(rng, n * b).reshape(n, b, 4),
                        rng.standard_normal((n, b, 3)), rng.standard_normal((n, b, 3)))


def _apply_world_transform(env, g, t):
    """Rigidly move the whole scene (root + every body) by rotation g (quat) + translation t."""
    from sim1.tasks.proprio import _quat_mul, quat_rotate
    n, b = env.body_pos.shape[0], env.body_pos.shape[1]
    gN = np.tile(g, (n, 1)).astype(np.float32)
    gNB = np.tile(g, (n * b, 1)).astype(np.float32)
    rp = env.root_pose.copy()
    rp[:, 0:3] = quat_rotate(gN, env.root_pose[:, 0:3]) + t
    rp[:, 3:7] = _quat_mul(gN, env.root_pose[:, 3:7])
    rot3 = lambda f: quat_rotate(gNB, f.reshape(n * b, 3)).reshape(n, b, 3)
    return _FakeBodyEnv(rp, rot3(env.body_pos) + t, _quat_mul(gNB, env.body_quat.reshape(n * b, 4)).reshape(n, b, 4),
                        rot3(env.body_linvel), rot3(env.body_angvel))


def test_per_body_dim_and_shape():
    from sim1.tasks.proprio import BodyFeatures, per_body_dim, per_body_obs
    assert per_body_dim(5, "sixd") == 5 * (3 + 6 + 3 + 3 + 1) + 3       # 83
    assert per_body_dim(5, "quat") == 5 * (3 + 4 + 3 + 3 + 1) + 3       # 73
    env = _make_body_env(np.random.default_rng(1))
    assert per_body_obs(env, "sixd").shape == (2, per_body_dim(5, "sixd"))
    bf = BodyFeatures(5)
    assert bf.dim == per_body_dim(5, "sixd")
    assert bf.observe(env, None).shape == (2, bf.dim)


def test_per_body_geometry_invariant_under_any_world_pose():
    # THE money test: the root-relative per-body channels (position, 6D rotation, linear + angular
    # velocity) are invariant to the character's global position AND orientation — the SuperTrack
    # property. (Per-body HEIGHT and the up-vector are deliberately world-absolute cues, so they are
    # excluded here and checked separately below.)
    from sim1.tasks.proprio import per_body_obs
    rng = np.random.default_rng(2)
    env = _make_body_env(rng)
    g, t = _randq(rng, 1)[0], rng.standard_normal(3).astype(np.float32)
    moved = _apply_world_transform(env, g, t)

    def _root_relative(o, b=5, per=16):   # per body: [pos3|6D6|linvel3|angvel3|height1]; drop height + up
        body = o[:, :b * per].reshape(o.shape[0], b, per)
        return body[:, :, :15].reshape(o.shape[0], -1)

    o1, o2 = per_body_obs(env, "sixd"), per_body_obs(moved, "sixd")
    err = np.abs(_root_relative(o1) - _root_relative(o2)).max()
    assert err < 2e-4, err


def test_per_body_height_and_up_invariant_under_yaw_and_horizontal_move():
    # Height + up-vector are absolute cues: invariant only under a world YAW + HORIZONTAL move
    # (which preserve gravity direction and heights), so the FULL obs matches under such a transform.
    from sim1.tasks.proprio import per_body_obs
    rng = np.random.default_rng(4)
    env = _make_body_env(rng)
    yaw = 0.9
    gy = np.array([np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0], dtype=np.float32)
    t = np.array([rng.standard_normal(), 0.0, rng.standard_normal()], dtype=np.float32)  # horizontal
    moved = _apply_world_transform(env, gy, t)
    assert np.allclose(per_body_obs(env, "sixd"), per_body_obs(moved, "sixd"), atol=2e-4)
    # for an upright root (identity), the local up-vector reads world-up (0,1,0)
    up_env = _FakeBodyEnv(np.array([[0, 1, 0, 1, 0, 0, 0]], dtype=np.float32),
                          np.zeros((1, 1, 3), np.float32), np.array([[[1, 0, 0, 0]]], np.float32),
                          np.zeros((1, 1, 3), np.float32), np.zeros((1, 1, 3), np.float32))
    assert np.allclose(per_body_obs(up_env, "sixd")[0, -3:], [0, 1, 0], atol=1e-6)


def test_root_body_is_identity_in_its_own_frame():
    from sim1.tasks.proprio import per_body_obs
    rng = np.random.default_rng(5)
    env = _make_body_env(rng)
    env.body_pos[:, 0, :] = env.root_pose[:, 0:3]      # body 0 coincides with the root
    env.body_quat[:, 0, :] = env.root_pose[:, 3:7]
    b0 = per_body_obs(env, "sixd")[:, 0:16]            # first body's block (3+6+3+3+1)
    assert np.allclose(b0[:, 0:3], 0.0, atol=1e-5)                  # root-relative position ~ 0
    assert np.allclose(b0[:, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-5)   # 6D rotation ~ identity


def test_body_obs_is_additive_and_default_off():
    from sim1.tasks.proprio import per_body_dim
    env = _env()
    base = StandTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0)
    withb = StandTask(ndof=env.ndof, nbody=1, act_dim=env.act_dim, action_scale=1.0, body_obs=True)
    assert withb.obs_dim == base.obs_dim + per_body_dim(1, "sixd")   # dim-additive
    withb.reset(env, 0)
    env.step()
    assert withb.observe(env).shape == (env.num_envs, withb.obs_dim)
    # default off: the base task's observation is unchanged (proprio only)
    base.reset(env, 0)
    assert base.observe(env).shape == (env.num_envs, base.obs_dim)
