"""Checkpoint round-trip: save -> load into a fresh trainer -> identical params + restored state."""

import torch

from sim1.algos.ppo import PPOTrainer
from sim1.config import PPOConfig
from sim1.envs.mock_vecenv import MockVecEnv
from sim1.envs.task_env import TaskEnv
from sim1.tasks.reach import ReachTask
from sim1.utils.checkpoint import load_checkpoint, save_checkpoint


def _make_trainer(seed=0):
    env = TaskEnv(MockVecEnv(num_envs=16, ndof=2, seed=seed), ReachTask(ndof=2), episode_len=16, seed=seed)
    cfg = PPOConfig(total_steps=10_000, rollout_len=16, num_minibatches=2, update_epochs=2)
    return PPOTrainer(env, cfg, device="cpu", seed=seed)


def test_checkpoint_round_trip(tmp_path):
    t1 = _make_trainer()
    for _ in range(3):
        batch, _ = t1.collect()
        t1.update(batch)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, t1.state_dict())

    t2 = _make_trainer()
    # params differ before load (same seed init is identical, so perturb t1 was trained) — verify load syncs them
    t2.load_state_dict(load_checkpoint(path))

    for p1, p2 in zip(t1.model.parameters(), t2.model.parameters()):
        assert torch.allclose(p1, p2)
    assert t2.global_step == t1.global_step
    assert t2.iteration == t1.iteration
    if t1.obs_rms is not None:
        assert torch.allclose(t1.obs_rms.mean, t2.obs_rms.mean)
        assert torch.allclose(t1.obs_rms.var, t2.obs_rms.var)
