"""export_policy carries the training SimConfig knobs the visualizer needs to reproduce dynamics.

Regression guard for the config-propagation hole (2026-07-05 fidelity review): `ground_friction` was
set at train time but never exported, so `sim1_viz` silently used the engine default. This pins that
the V7 export emits it, at the training value (default *and* a non-default sweep value)."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim1.config import EnvConfig, TaskConfig, TrainConfig
from sim1.export_policy import export


def _write_run(tmp_path, ground_friction=0.9, kind="engine"):
    """Minimal run dir: config.json + a checkpoint whose actor_mean matches the stand obs/act dims."""
    ndof, nbody = 28, 15
    obs_dim = 1 + 4 + 3 + 3 + 2 * ndof + nbody           # quat/world proprio, no command/body (=82)
    cfg = TrainConfig(
        env=EnvConfig(kind=kind, model="amp", num_envs=1, ground_friction=ground_friction),
        task=TaskConfig(name="stand"),
    )
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()))
    # actor_mean Sequential: Linear(obs,32) -> (tanh) -> Linear(32,act); export reads these layers.
    sd = {
        "actor_mean.0.weight": torch.zeros(32, obs_dim), "actor_mean.0.bias": torch.zeros(32),
        "actor_mean.2.weight": torch.zeros(ndof, 32),    "actor_mean.2.bias": torch.zeros(ndof),
    }
    (tmp_path / "checkpoints").mkdir()
    torch.save({"model": sd}, tmp_path / "checkpoints" / "best.pt")   # no obs_rms → identity normalizer
    return tmp_path


@pytest.mark.parametrize("gf", [0.9, 0.35])
def test_export_carries_ground_friction(tmp_path, gf):
    run = _write_run(tmp_path, gf)
    export(str(run), checkpoint="best.pt")
    text = (run / "policy.txt").read_text()
    assert text.startswith("SIM1_POLICY_V8")
    m = re.search(r"ground_friction (\S+)", text)
    assert m is not None, "ground_friction token missing from export"
    assert float(m.group(1)) == pytest.approx(gf)


@pytest.mark.parametrize("kind,expected", [
    ("engine", "reduced"),        # PhysicsWorld solver
    ("diff-cpu", "diff-cpu"),     # diff-ABA runs must NOT mislabel as 'reduced'
    ("cuda", "cuda"),
])
def test_export_backend_label_is_truthful(tmp_path, kind, expected):
    run = _write_run(tmp_path, kind=kind)
    export(str(run), checkpoint="best.pt")
    m = re.search(r"backend (\S+)", (run / "policy.txt").read_text())
    assert m is not None and m.group(1) == expected
