"""Metric logging to TensorBoard + a machine-readable JSONL stream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    def __init__(self, run_dir: str | Path):
        run_dir = Path(run_dir)
        self._tb = SummaryWriter(log_dir=str(run_dir / "tb"))
        self._jsonl = open(run_dir / "metrics.jsonl", "a")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            self._tb.add_scalar(k, float(v), step)
        self._jsonl.write(json.dumps({"step": int(step), **{k: float(v) for k, v in metrics.items()}}) + "\n")
        self._jsonl.flush()

    def close(self) -> None:
        self._tb.close()
        self._jsonl.close()
