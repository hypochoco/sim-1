"""Exploration knobs: cyclical entropy schedule, target-KL early stop, lr floor, and adaptive
parameter-space noise. These keep exploration alive / periodically revive it to escape local minima."""

import numpy as np
import torch

from sim1.algos.ppo import PPOTrainer
from sim1.config import EnvConfig, PPOConfig, RunConfig, TaskConfig, TrainConfig
from sim1.envs.mock_vecenv import MockVecEnv
from sim1.envs.task_env import TaskEnv
from sim1.tasks.reach import ReachTask
from sim1.train import run_training


def _trainer(cfg: PPOConfig, seed=0) -> PPOTrainer:
    env = TaskEnv(MockVecEnv(num_envs=16, ndof=2, seed=seed), ReachTask(ndof=2), episode_len=16, seed=seed)
    return PPOTrainer(env, cfg, device="cpu", seed=seed)


# --- (1a) cyclical entropy ---
def test_entropy_cycle_peaks_then_decays():
    t = _trainer(PPOConfig(total_steps=10_000, rollout_len=16, ent_coef=0.001,
                           ent_cycle_iters=10, ent_cycle_max=0.05))
    t.iteration = 0
    start = t._current_ent_coef()          # cycle start → peak
    t.iteration = 9
    end = t._current_ent_coef()            # cycle end → near base
    t.iteration = 10
    restart = t._current_ent_coef()        # next cycle → back to peak
    assert abs(start - 0.05) < 1e-6
    assert end < 0.01 and end >= 0.001
    assert abs(restart - 0.05) < 1e-6      # exploration is periodically revived


def test_entropy_constant_when_cycle_disabled():
    t = _trainer(PPOConfig(total_steps=10_000, rollout_len=16, ent_coef=0.007))
    for it in (0, 5, 50):
        t.iteration = it
        assert t._current_ent_coef() == 0.007


# --- (1b) target-KL early stop ---
def test_target_kl_early_stops_epochs():
    t = _trainer(PPOConfig(total_steps=10_000, rollout_len=16, update_epochs=8,
                           target_kl=1e-6))   # absurdly tight → should stop after the first epoch
    batch, _ = t.collect()
    m = t.update(batch)
    assert m["losses/epochs_done"] < 8

    t2 = _trainer(PPOConfig(total_steps=10_000, rollout_len=16, update_epochs=8))  # disabled
    b2, _ = t2.collect()
    assert t2.update(b2)["losses/epochs_done"] == 8


# --- (1c) lr floor ---
def test_lr_anneals_to_floor_not_zero(tmp_path):
    cfg = TrainConfig(
        env=EnvConfig(kind="mock", num_envs=32, ndof=2, episode_len=16),
        task=TaskConfig(),
        ppo=PPOConfig(total_steps=20_000, rollout_len=16, lr=3e-4, anneal_lr=True, lr_final_frac=0.2),
        run=RunConfig(name="lrfloor", seed=0, device="cpu", runs_root=str(tmp_path)),
    )
    h = run_training(cfg)["history"]
    last_lr = h[-1]["charts/learning_rate"]
    assert last_lr >= 0.2 * 3e-4 - 1e-9      # never dropped below the floor
    assert last_lr < 3e-4                     # but did anneal


# --- (3) adaptive parameter-space noise ---
def test_param_noise_builds_perturbed_actor_and_adapts():
    t = _trainer(PPOConfig(total_steps=10_000, rollout_len=16, param_noise=True,
                           param_noise_init=0.02, param_noise_target=0.1))
    assert t._pnoise_actor is not None
    before = t._pnoise_std
    _, roll = t.collect()
    assert "charts/param_noise_std" in roll and "charts/param_noise_dist" in roll
    assert t._pnoise_std != before          # std adapted toward the target distance


def test_param_noise_trains_and_exports_the_clean_model():
    # The perturbed actor only ACTS; PPO must update self.model (the clean, exported policy).
    t = _trainer(PPOConfig(total_steps=10_000, rollout_len=16, update_epochs=2, param_noise=True))
    clean_before = [p.clone() for p in t.model.actor_mean.parameters()]
    for _ in range(2):
        batch, _ = t.collect()
        t.update(batch)
    moved = any(not torch.allclose(a, b) for a, b in zip(clean_before, t.model.actor_mean.parameters()))
    assert moved                            # the clean policy learned
    # perturbed actor stays a frozen (no-grad) scratch copy
    assert all(not p.requires_grad for p in t._pnoise_actor.parameters())


def test_param_noise_off_by_default():
    t = _trainer(PPOConfig(total_steps=10_000, rollout_len=16))
    assert t._pnoise_actor is None
    _, roll = t.collect()
    assert "charts/param_noise_std" not in roll


def test_logprobs_clean_equals_behavior_without_param_noise():
    t = _trainer(PPOConfig(total_steps=10_000, rollout_len=16))
    b, _ = t.collect()
    assert torch.allclose(b["logprobs"], b["logprobs_clean"])


def test_param_noise_kl_measured_against_clean_policy():
    # The trust-region KL must reflect true policy drift, not the exploration perturbation — otherwise
    # target_kl early-stops every update after 1 epoch (param-noise inflates the behavior-based KL).
    t = _trainer(PPOConfig(total_steps=10_000, rollout_len=16, param_noise=True, param_noise_init=0.05))
    b, _ = t.collect()
    assert not torch.allclose(b["logprobs"], b["logprobs_clean"])   # behavior != clean under perturbation
    with torch.no_grad():
        newlogp = t.model.get_action_and_value(b["obs"], b["actions"])[1]

    def kl(ref):
        lr = newlogp - ref
        return float(((lr.exp() - 1) - lr).mean())

    assert abs(kl(b["logprobs_clean"])) < 1e-4          # vs clean (== current model) → ~0 at update start
    assert kl(b["logprobs"]) > 1e-3                     # vs perturbed behavior → inflated (the old bug)
