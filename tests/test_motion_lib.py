"""MotionLib loader — validates ASE clip loading + retarget into our rig frame (step 1 of the mocap
plan). Skipped when the ASE motion data isn't checked out locally."""

from __future__ import annotations

import numpy as np
import pytest

from sim1.motion.motion_lib import AmpMotionLib, default_motion_path, load_reference_motion

_WALK = default_motion_path("amp_humanoid_walk")
pytestmark = pytest.mark.skipif(not _WALK.exists(), reason=f"ASE motion not found at {_WALK}")

# our makeAMPHumanoid body order — FEET LAST (the engine pulls both feet to the end); the loader must
# reindex the ASE clip (which interleaves feet) into this order.
_RIG_ORDER = ["pelvis", "torso", "head", "right_upper_arm", "right_lower_arm", "right_hand",
              "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin",
              "left_thigh", "left_shin", "right_foot", "left_foot"]
_RFOOT, _LFOOT = 13, 14


def test_structure_and_order():
    m = load_reference_motion(_WALK)
    assert m.num_bodies == 15 and m.body_names == _RIG_ORDER
    assert m.fps == 30.0 and m.num_frames == 40
    assert m.duration == pytest.approx(39 / 30.0)
    for q in (m.body_quat, m.local_quat):
        assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5)


def test_matches_engine_body_layout():
    # The reordering must land each reference body in the SAME slot as our engine's articulation body
    # (else the viz shows feet on the wrong legs / meshes on the wrong bodies — the observed bug).
    engine_py = pytest.importorskip("sim1.engine_py", reason="engine binding not built")
    from sim1.config import EnvConfig
    from sim1.envs.engine_vecenv import EngineVecEnv

    env = EngineVecEnv(EnvConfig(kind="engine", model="amp", num_envs=1), seed=0)
    env.reset(0)
    eng = np.asarray(env.body_pos)[0]                 # (15,3) authored pose, our frame
    ref = load_reference_motion(_WALK).body_pos[0]    # (15,3) reference frame 0, reordered to our order

    # feet are the last two bodies and the lowest two — in BOTH
    assert set(np.argsort(eng[:, 1])[:2]) == {_RFOOT, _LFOOT}
    assert set(np.argsort(ref[:, 1])[:2]) == {_RFOOT, _LFOOT}
    # left/right sidedness (sign of lateral X) agrees per body for the limbs
    limbs = list(range(3, 15))
    assert np.array_equal(np.sign(np.round(eng[limbs, 0], 2)), np.sign(np.round(ref[limbs, 0], 2)))
    # head is the top engine body; after reorder the reference head sits above the pelvis too
    assert np.argmax(eng[:, 1]) == 2
    assert ref[2, 1] > ref[0, 1]


def test_joint_positions_not_body_centers_is_documented():
    # KNOWN LIMITATION: the clip stores ASE *joint* positions with ASE proportions, so replaying them
    # directly onto our body-CENTER meshes is not faithful — e.g. the ASE shoulder joints sit slightly
    # ABOVE the head joint, whereas our rig's head center is well above the shoulders. A faithful
    # replay needs a rotation-based retarget onto our rig's rest offsets. Pin the discrepancy so we
    # notice when that lands.
    engine_py = pytest.importorskip("sim1.engine_py", reason="engine binding not built")
    from sim1.config import EnvConfig
    from sim1.envs.engine_vecenv import EngineVecEnv

    env = EngineVecEnv(EnvConfig(kind="engine", model="amp", num_envs=1), seed=0)
    env.reset(0)
    eng = np.asarray(env.body_pos)[0]
    ref = load_reference_motion(_WALK).body_pos[0]
    head, r_sh, l_sh = 2, 3, 6
    assert eng[head, 1] > max(eng[r_sh, 1], eng[l_sh, 1])       # our rig: head above shoulders
    assert ref[head, 1] < max(ref[r_sh, 1], ref[l_sh, 1])       # ASE joints: shoulders above head


def test_frame_is_y_up_and_walks_forward():
    m = load_reference_motion(_WALK)
    # Y is up: pelvis height positive & plausible; feet reach near the ground.
    assert 0.7 < m.body_pos[:, 0, 1].mean() < 1.1
    assert min(m.body_pos[:, _RFOOT, 1].min(), m.body_pos[:, _LFOOT, 1].min()) < 0.15
    # head above pelvis above feet at every frame
    assert (m.body_pos[:, 2, 1] > m.body_pos[:, 0, 1]).all()
    # forward progress along +Z (our frame), lateral (X) small
    fwd = m.body_pos[-1, 0, 2] - m.body_pos[0, 0, 2]
    assert fwd > 0.8
    assert abs(np.ptp(m.body_pos[:, 0, 0])) < 0.3


def test_fk_matches_stored_velocity():
    # Independent check of FK + the coordinate map together: d(body_pos)/dt from our FK should match
    # the clip's own stored per-body velocity (both now in our frame). poselib stores *gaussian-
    # smoothed* velocities, so we validate the smooth root tightly + global direction (limb swings
    # diverge from raw finite-difference at foot-strike, which is expected, not a frame error).
    m = load_reference_motion(_WALK)
    dt = 1.0 / m.fps
    fd = (m.body_pos[1:] - m.body_pos[:-1]) / dt              # (F-1, B, 3) forward diff
    ref = 0.5 * (m.body_linvel[1:] + m.body_linvel[:-1])      # midpoint of stored velocity
    root_err = np.abs(fd[:, 0] - ref[:, 0]).max()
    assert root_err < 0.12, f"root FK/velocity mismatch {root_err:.3f} m/s — frame map or FK is off"
    assert np.abs(fd - ref).mean() < 0.06                     # mean over all bodies stays small
    # velocity *direction* agrees globally (would be destroyed by a wrong permutation)
    fdn, rn = fd.reshape(-1, 3), ref.reshape(-1, 3)
    mask = np.linalg.norm(rn, axis=1) > 0.3
    cos = np.sum(fdn[mask] * rn[mask], axis=1) / (np.linalg.norm(fdn[mask], axis=1) * np.linalg.norm(rn[mask], axis=1))
    assert cos.mean() > 0.95


def test_state_at_interpolates():
    lib = AmpMotionLib().load(_WALK)
    assert lib.num_clips() == 1
    m = lib.clip(0)
    # at an exact frame time we recover that frame
    t_frame5 = 5 / m.fps
    s = lib.state_at(np.array([t_frame5]))
    assert np.allclose(s.body_pos[0], m.body_pos[5], atol=1e-6)
    assert s.root_pos.shape == (1, 3) and s.body_quat.shape == (1, 15, 4)
    # a midpoint lies between the bracketing frames
    mid = lib.state_at(np.array([5.5 / m.fps])).body_pos[0, 0]
    lo, hi = m.body_pos[5, 0], m.body_pos[6, 0]
    assert np.all((mid >= np.minimum(lo, hi) - 1e-6) & (mid <= np.maximum(lo, hi) + 1e-6))


def test_sample_times_in_range():
    lib = AmpMotionLib().load(_WALK)
    t = lib.sample_times(64, np.random.default_rng(0))
    assert t.shape == (64,) and t.min() >= 0.0 and t.max() <= lib.clip(0).duration


def test_export_motion_roundtrip(tmp_path):
    # Locks the flat SIM1_MOTION_V1 format the C++ viz reads: header + F*B rows of pos(3)+quat(4).
    from sim1.export_motion import export_motion

    m = load_reference_motion(_WALK)
    out = export_motion(m, tmp_path / "w.motion.txt")
    lines = out.read_text().splitlines()
    assert lines[0] == "SIM1_MOTION_V2"
    assert lines[1].split() == ["fps", "30", "num_frames", "40", "num_bodies", "15"]
    assert len(lines) == 2 + m.num_frames * m.num_bodies
    row0 = np.array(lines[2].split(), dtype=np.float64)   # frame 0, body 0: pos3 quat4 linvel3 angvel3
    assert row0.shape == (13,)
    assert np.allclose(row0[:3], m.body_pos[0, 0], atol=1e-6)
    assert np.allclose(row0[3:7], m.body_quat[0, 0], atol=1e-6)
    assert np.allclose(row0[7:10], m.body_linvel[0, 0], atol=1e-6)
    assert np.allclose(row0[10:13], m.body_angvel[0, 0], atol=1e-6)


def _engine_env():
    engine_py = pytest.importorskip("sim1.engine_py", reason="engine binding not built")  # noqa: F841
    from sim1.config import EnvConfig
    from sim1.envs.engine_vecenv import EngineVecEnv
    return EngineVecEnv(EnvConfig(kind="engine", model="amp", num_envs=1, action_mode="pd_target"), seed=0)


def test_retarget_reproduces_engine_fk():
    # The definitive correctness check: driven by the ENGINE's own body orientations at an articulated
    # pose, our 2-term joint-anchored FK must land every body center exactly where the engine does.
    # This validates both the FK math AND the hardcoded joint anchors (wrong anchors → error at a flex).
    from sim1.motion.motion_lib import retarget_to_rig, ReferenceMotion

    m = load_reference_motion(_WALK)                 # for body_names / parents
    env = _engine_env(); env.reset(0)
    rest = np.asarray(env.body_pos)[0].copy()
    rng = np.random.default_rng(1)
    env.actions[:] = rng.uniform(-0.6, 0.6, env.act_dim)
    for _ in range(6):
        env.step()
    bp = np.asarray(env.body_pos)[0].copy()
    bq = np.asarray(env.body_quat)[0].copy()
    assert np.isfinite(bp).all() and np.isfinite(bq).all()
    # confirm the pose is actually articulated (else the anchor terms cancel and the test is vacuous)
    rel = np.abs(bq[1:] - bq[m.parents[1:]]).sum(axis=1)
    assert rel.max() > 0.05
    mm = ReferenceMotion(fps=30.0, body_names=m.body_names, parents=m.parents,
                         body_pos=np.tile(bp, (2, 1, 1)), body_quat=np.tile(bq, (2, 1, 1)),
                         body_linvel=np.zeros((2, 15, 3)), body_angvel=np.zeros((2, 15, 3)),
                         local_quat=np.tile(bq, (2, 1, 1)))
    rt = retarget_to_rig(mm, rest, ground=False)
    assert np.abs(rt.body_pos[0] - bp).max() < 1e-6


def test_retarget_fixes_proportions_and_grounding():
    # The reported bug: replaying ASE joint positions put the upper body a segment too low. The
    # rotation-based retarget must restore OUR proportions (head above shoulders, at our heights) and
    # keep the feet on the ground.
    from sim1.motion.motion_lib import retarget_to_rig

    env = _engine_env(); env.reset(0)
    rest = np.asarray(env.body_pos)[0].copy()
    m = load_reference_motion(_WALK)
    rt = retarget_to_rig(m, rest)
    # our proportions: pelvis/torso/head at our authored heights, strictly increasing
    assert abs(rt.body_pos[:, 0, 1].mean() - rest[0, 1]) < 0.12
    assert rt.body_pos[:, 2, 1].mean() > rt.body_pos[:, 1, 1].mean() > rt.body_pos[:, 0, 1].mean()
    # head is ABOVE the shoulders now (the raw-clip relationship is inverted; see the doc test above)
    assert rt.body_pos[:, 2, 1].mean() > max(rt.body_pos[:, 3, 1].mean(), rt.body_pos[:, 6, 1].mean())
    # feet grounded (not sunk through the floor) and the root still walks forward (+Z)
    assert rt.body_pos[:, [_RFOOT, _LFOOT], 1].min() > -0.02
    assert rt.body_pos[-1, 0, 2] - rt.body_pos[0, 0, 2] > 0.8
    assert np.isfinite(rt.body_linvel).all()


def test_motions_dir_env_override(monkeypatch, tmp_path):
    from sim1.motion.motion_lib import default_motion_path

    monkeypatch.setenv("SIM1_MOTIONS_DIR", str(tmp_path))
    assert default_motion_path("some_clip") == tmp_path / "some_clip.npy"
    monkeypatch.delenv("SIM1_MOTIONS_DIR")
    assert default_motion_path("amp_humanoid_walk").name == "amp_humanoid_walk.npy"  # sibling fallback
