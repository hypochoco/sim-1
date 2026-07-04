"""Reward normalization keeps the value scale bounded (so value clipping / lr work) and helps
learning at large reward magnitudes."""

import numpy as np

from sim1.config import EnvConfig, PPOConfig, RunConfig, TaskConfig, TrainConfig
from sim1.train import run_training


def _run(norm_reward: bool, tmp_path):
    cfg = TrainConfig(
        env=EnvConfig(kind="mock", num_envs=64, ndof=2, episode_len=32),
        task=TaskConfig(pos_weight=20.0),  # large reward scale (un-normalized returns ~ hundreds)
        ppo=PPOConfig(total_steps=80_000, rollout_len=32, norm_reward=norm_reward),
        run=RunConfig(name="rn", seed=0, device="cpu", runs_root=str(tmp_path)),
    )
    h = run_training(cfg)["history"]
    vl = float(np.mean([x["losses/value_loss"] for x in h[-3:]]))
    early = float(np.mean([x["charts/reward_mean"] for x in h[:3]]))
    late = float(np.mean([x["charts/reward_mean"] for x in h[-3:]]))
    return vl, late - early


def test_reward_norm_bounds_value_scale(tmp_path):
    vl_on, gain_on = _run(True, tmp_path / "on")
    vl_off, gain_off = _run(False, tmp_path / "off")

    # With normalization the value target scale is O(1); without, it blows up with the reward scale.
    assert vl_on < 10.0, f"normalized value loss unexpectedly large: {vl_on:.3f}"
    assert vl_off > 100 * vl_on, f"normalization did not tame value scale: on={vl_on:.3f} off={vl_off:.3f}"

    # Reported reward_mean is always the RAW reward (normalization is internal).
    assert gain_on > gain_off, f"normalization did not help learning: on={gain_on:.2f} off={gain_off:.2f}"
