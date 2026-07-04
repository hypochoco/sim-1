"""MockVecEnv: contract shapes + determinism (same seed + actions => identical trajectory)."""

import numpy as np

from sim1.envs.mock_vecenv import MockVecEnv


def _rollout(seed, actions_seq):
    env = MockVecEnv(num_envs=8, ndof=3, seed=seed)
    env.reset(seed)
    qs = []
    for a in actions_seq:
        env.actions[:] = a
        env.step()
        qs.append(env.joint_q.copy())
    return np.stack(qs)


def test_contract_shapes():
    env = MockVecEnv(num_envs=5, ndof=4, seed=0)
    assert env.num_envs == 5 and env.act_dim == 4
    assert env.actions.shape == (5, 4)
    assert env.joint_q.shape == (5, 4)
    assert env.joint_qd.shape == (5, 4)
    assert env.root_pose.shape == (5, 7)
    assert env.root_twist.shape == (5, 6)
    assert env.contact_flags.shape[0] == 5


def test_determinism():
    rng = np.random.default_rng(0)
    actions_seq = [rng.standard_normal((8, 3)).astype(np.float32) for _ in range(20)]
    a = _rollout(123, actions_seq)
    b = _rollout(123, actions_seq)
    assert np.array_equal(a, b)
