"""PPO must demonstrably improve on the mock reach task (per-step reward rises over training)."""

import numpy as np

from sim1.config import EnvConfig, PPOConfig, RunConfig, TaskConfig, TrainConfig
from sim1.train import run_training


def _small_cfg(tmp_path) -> TrainConfig:
    return TrainConfig(
        env=EnvConfig(kind="mock", num_envs=64, ndof=2, episode_len=32, target_scale=1.0),
        task=TaskConfig(name="reach", action_weight=0.01),
        ppo=PPOConfig(total_steps=120_000, rollout_len=32, num_minibatches=4, update_epochs=4, lr=3e-4),
        run=RunConfig(name="test_ppo", seed=0, device="cpu", runs_root=str(tmp_path), checkpoint_interval=1000),
    )


def test_ppo_learns(tmp_path):
    result = run_training(_small_cfg(tmp_path))
    hist = result["history"]
    assert len(hist) >= 10

    def col(hs, k):
        return float(np.mean([h[k] for h in hs]))

    # (1) per-step reward rises as the policy learns (available every iteration).
    early_r, late_r = col(hist[:3], "charts/reward_mean"), col(hist[-3:], "charts/reward_mean")
    assert late_r > early_r + 0.2, f"reward did not improve: early={early_r:.3f} late={late_r:.3f}"

    # (2) the value function is being fit (value loss falls substantially).
    early_v, late_v = col(hist[:3], "losses/value_loss"), col(hist[-3:], "losses/value_loss")
    assert late_v < 0.8 * early_v, f"value loss did not fall: early={early_v:.1f} late={late_v:.1f}"

    # metrics are finite throughout.
    assert all(np.isfinite(h["losses/policy_loss"]) for h in hist)
