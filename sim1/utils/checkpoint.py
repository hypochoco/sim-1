"""Resumable checkpointing: atomic writes + rolling retention."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    # Our own trusted checkpoints contain config dicts + RNG state, not just tensors.
    return torch.load(str(path), map_location=map_location, weights_only=False)


def prune_checkpoints(ckpt_dir: str | Path, keep_last: int) -> None:
    ckpts = sorted(Path(ckpt_dir).glob("step_*.pt"))
    for p in ckpts[: max(0, len(ckpts) - keep_last)]:
        p.unlink()
