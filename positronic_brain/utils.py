"""
Utility functions for the Positronic Brain model.

Includes coordinate transforms, activation functions, and small math helpers.
All functions are CPU-friendly and vectorizable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize to [0, 1] range."""
    a_min, a_max = arr.min(), arr.max()
    if a_max - a_min < eps:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min + eps)


def grid_to_index(x: int, y: int, z: int, grid_size: int) -> int:
    """Convert 3D grid coordinates to flat neuron index (row-major)."""
    return x * grid_size * grid_size + y * grid_size + z


def index_to_grid(idx: int, grid_size: int) -> Tuple[int, int, int]:
    """Convert flat neuron index back to (x, y, z) coordinates."""
    G2 = grid_size * grid_size
    x = idx // G2
    rem = idx % G2
    y = rem // grid_size
    z = rem % grid_size
    return int(x), int(y), int(z)


def compute_distance(p1: np.ndarray | Tuple[float, float, float],
                     p2: np.ndarray | Tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    if isinstance(p1, tuple):
        p1 = np.asarray(p1, dtype=np.float32)
    if isinstance(p2, tuple):
        p2 = np.asarray(p2, dtype=np.float32)
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def zone_onehot(zones: np.ndarray, num_zones: int = 4) -> np.ndarray:
    """Convert zone id array (N,) to one-hot (N, num_zones)."""
    N = zones.shape[0]
    oh = np.zeros((N, num_zones), dtype=np.float32)
    oh[np.arange(N), zones] = 1.0
    return oh


def moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple 1D moving average with edge handling."""
    if window <= 1:
        return x.copy()
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(window) / window, mode="valid")[: len(x)]


# -----------------------------------------------------------------------------
# Device selection (MPS / Metal on Apple Silicon MacBook Pro, CUDA, CPU fallback)
# -----------------------------------------------------------------------------

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """
    Return the preferred torch.device for the Positronic Brain.

    This enables optional GPU acceleration on Apple Silicon Macs via the
    Metal Performance Shaders (MPS) backend without any code changes for
    most users.

    Args:
        prefer: One of "auto", "cpu", "mps", or "cuda".
            - "auto": Prefer MPS if available (Apple Silicon MacBook Pro),
              then CUDA, then CPU. This is the recommended default.
            - "mps": Force Apple Metal (will raise / fall back if unavailable
              depending on PyTorch version and hardware).
            - "cuda": Force NVIDIA CUDA.
            - "cpu": Force CPU (useful for reproducibility, benchmarking the
              tiny default model, or running on Intel Macs / non-Apple hardware).

    Returns:
        A torch.device object (e.g. device(type='mps'), device(type='cpu')).

    Notes for MacBook Pro users:
        - Requires a recent torch wheel (the project pins >=2.2). Standard
          `pip install torch` on macOS gives you the MPS-enabled build.
        - On M-series chips (M1/M2/M3/M4 and later) `torch.backends.mps.is_available()`
          will be True.
        - For the default 4×4×4 brain the CPU is often *faster* (lower dispatch
          overhead for tiny tensors). MPS shines when you increase `grid_size`
          to 6–8 or run many simulations.
        - On Intel Macs or older torch: "auto" gracefully returns CPU.

    Example:
        >>> from positronic_brain.utils import get_device
        >>> dev = get_device("auto")
        >>> brain = PositronicBrain(BrainConfig(), device=dev)
        >>> # or simply
        >>> brain = PositronicBrain(BrainConfig(), device="auto")
    """
    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        # Fall through to CPU if the user explicitly asked for mps but it isn't there
        return torch.device("cpu")

    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # "auto" (and any unknown value falls back to this logic)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
