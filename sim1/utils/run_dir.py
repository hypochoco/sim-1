"""Run directory management: organized, reproducible experiment outputs.

Layout:  runs/<YYYYMMDD-HHMMSS>_<name>_<gitsha>/
             config.json        frozen config
             meta.json          seed, git sha, host, time
             metrics.jsonl      machine-readable metric stream
             tb/                TensorBoard event files
             checkpoints/       step_*.pt, best.pt, final.pt
A `runs/latest` symlink points at the most recent run.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def create_run_dir(runs_root: str, name: str, config_dict: dict[str, Any], seed: int) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    sha = _git_sha()
    run_dir = Path(runs_root) / f"{ts}_{name}_{sha}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    meta = {"time": ts, "git_sha": sha, "host": socket.gethostname(), "seed": seed}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    _update_latest(Path(runs_root), run_dir)
    return run_dir


def _update_latest(runs_root: Path, run_dir: Path) -> None:
    link = runs_root / "latest"
    try:
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            # A zip/copy that dereferenced the symlink turns `latest` into a real directory; unlink()
            # would raise IsADirectoryError (swallowed below) and `latest` would go stale forever.
            shutil.rmtree(link)
        link.symlink_to(run_dir.resolve(), target_is_directory=True)
    except OSError:
        pass  # symlinks may be unavailable on some filesystems; non-fatal
