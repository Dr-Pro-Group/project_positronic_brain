"""
Reproducibility utilities for Positronic Brain.

A single place to (a) seed every RNG the project touches, (b) capture the exact
environment a run was produced in (git SHA, library versions, command line,
device), and (c) serialise that provenance into a checkpoint so any reported
number can be traced back to a reproducible command.

Design goals
------------
* **No silent nondeterminism.** ``seed_everything`` seeds Python, NumPy and
  torch (CPU/CUDA), and can optionally request deterministic algorithms.
* **Cheap and dependency-light.** Everything degrades gracefully if torch or
  git are missing, so the module is importable in any environment.
* **Self-describing checkpoints.** ``run_metadata`` returns a JSON-serialisable
  dict that ``train_language.py`` stores alongside the weights.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """
    Seed Python, NumPy and torch RNGs from a single integer.

    Args:
        seed: the master seed.
        deterministic: if True, also request deterministic cuDNN/torch kernels
            and set ``CUBLAS_WORKSPACE_CONFIG``. This trades some throughput for
            bit-for-bit reproducibility where the backend supports it. Note that
            a few sparse-scatter kernels (``index_add`` on CUDA/MPS) are only
            *approximately* deterministic; we surface that honestly rather than
            pretend otherwise.

    Returns:
        The seed actually used (echoed for logging convenience).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # numpy should always be present, but never hard-fail
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    return seed


def git_sha(short: bool = True) -> Optional[str]:
    """Return the current git commit SHA, or ``None`` if unavailable."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.check_output(
            args, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def git_dirty() -> Optional[bool]:
    """Return True if the working tree has uncommitted changes (None if unknown)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode().strip())
    except Exception:
        return None


def library_versions() -> Dict[str, str]:
    """Best-effort version snapshot of the libraries that affect numerics."""
    versions: Dict[str, str] = {"python": platform.python_version()}
    for mod in ("torch", "numpy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "n/a"
    return versions


def run_metadata(
    seed: int,
    device: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble a JSON-serialisable provenance record for a training run.

    This is what makes a checkpoint self-describing: stored next to the weights,
    it records the seed, code version, environment and exact command used.
    """
    meta: Dict[str, Any] = {
        "seed": int(seed),
        "git_sha": git_sha(short=False),
        "git_dirty": git_dirty(),
        "device": str(device) if device is not None else None,
        "platform": platform.platform(),
        "argv": " ".join(sys.argv),
        "libraries": library_versions(),
    }
    if extra:
        meta.update(extra)
    return meta


@dataclass
class RunConfig:
    """Lightweight, serialisable container pairing a config dict with provenance."""

    config: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
