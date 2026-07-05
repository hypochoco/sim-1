"""Regression tests for the two bugs found porting to Linux (checkpoint sort + latest symlink)."""

from __future__ import annotations

from pathlib import Path

from sim1.train import _newest_checkpoint
from sim1.utils.checkpoint import prune_checkpoints, step_number
from sim1.utils.run_dir import _update_latest


def test_prune_keeps_newest_by_step_across_digit_boundary(tmp_path: Path):
    # Steps that straddle a digit-count boundary; a lexicographic sort mis-orders these.
    for s in (901_120, 983_040, 1_024_000, 6_303_744):
        (tmp_path / f"step_{s}.pt").touch()
    prune_checkpoints(tmp_path, keep_last=3)
    survivors = {step_number(p) for p in tmp_path.glob("step_*.pt")}
    assert survivors == {983_040, 1_024_000, 6_303_744}     # oldest (901,120) pruned, newest kept
    assert 901_120 not in survivors


def test_newest_checkpoint_is_numeric(tmp_path: Path):
    for s in (983_040, 1_024_000):
        (tmp_path / f"step_{s}.pt").touch()
    # lexicographically 'step_983040.pt' > 'step_1024000.pt'; numerically 1,024,000 is newest.
    assert _newest_checkpoint(tmp_path).name == "step_1024000.pt"


def test_update_latest_replaces_real_directory(tmp_path: Path):
    runs = tmp_path
    run_dir = runs / "20260101-000000_run_abc"
    run_dir.mkdir(parents=True)
    # Simulate a copy/zip having dereferenced `latest` into a real directory (with contents).
    stale = runs / "latest"
    stale.mkdir()
    (stale / "some_copied_file").touch()
    _update_latest(runs, run_dir)
    assert (runs / "latest").is_symlink()
    assert (runs / "latest").resolve() == run_dir.resolve()
