#!/usr/bin/env python3
"""
visualize.py — Non-interactive visualization utilities for Positronic Brain v3.

Generates static 3D snapshots or animations (via matplotlib) of brain activity.
Useful for papers, reports, or quick sanity checks without launching Streamlit.

The brain is multi-modal and zone-count-agnostic: inputs are given either as a
named preset or as an explicit vector with one value per zone.

Examples:
    python visualize.py --preset threat --timesteps 0,6,12 --save assets/threat.png
    python visualize.py --input 0.1 0.1 0.0 0.9 0.0 0.3 --animate --save assets/replay.gif
    python visualize.py --device mps --preset calm --save assets/replay.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from positronic_brain.model import PositronicBrain, BrainConfig
from positronic_brain.utils import get_device


# Named presets expressed in terms of zone NAMES so they work for any config.
PRESETS: Dict[str, Dict[str, float]] = {
    "threat":   {"Visual": 0.92, "Auditory": 0.88, "Emotion": 0.5},
    "calm":     {"Memory": 0.95},
    "visual":   {"Visual": 0.85},
    "touch":    {"Somatosensory": 0.9, "Emotion": 0.6},
    "cross":    {"Visual": 0.6, "Auditory": 0.55, "Association": 0.8},
    "internal": {"Memory": 0.6, "Association": 0.7},
    "baseline": {},
}


def load_or_create_brain(
    model_path: Optional[str] = None,
    device: str = "auto",
) -> PositronicBrain:
    """Load a checkpoint or create a fresh brain, optionally on MPS / CUDA."""
    if model_path and Path(model_path).exists():
        return PositronicBrain.load(model_path, device=device, strict=False)
    b = PositronicBrain(BrainConfig(), device=device)
    b.eval()
    return b


def input_vector(brain: PositronicBrain, named: Dict[str, float]) -> np.ndarray:
    """Build a (Z,) input vector from a {zone_name: value} mapping."""
    names = brain.config.zone_names
    vec = np.zeros(len(names), dtype=np.float32)
    for zname, val in named.items():
        if zname in names:
            vec[names.index(zname)] = val
    return vec


def plot_3d_snapshot(
    positions: np.ndarray,
    activations: np.ndarray,
    zones: np.ndarray,
    zone_colors: Sequence[str],
    grid_size: int,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Matplotlib 3D scatter snapshot (good for static figures)."""
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

    colors = [zone_colors[int(z)] for z in zones]
    sizes = 28 + activations * 95
    alphas = 0.25 + activations * 0.72
    rgba = np.array([to_rgba(c, a) for c, a in zip(colors, alphas)])

    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=rgba, s=sizes, edgecolors="none", depthshade=True,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=38)
    lim = grid_size - 0.5
    ax.set_xlim(-0.5, lim); ax.set_ylim(-0.5, lim); ax.set_zlim(-0.5, lim)
    ax.grid(False)
    ax.set_facecolor("#f8f9fa")
    return ax


def save_snapshots(
    brain: PositronicBrain,
    inputs: np.ndarray,
    timesteps: List[int],
    out_path: Path,
) -> None:
    res = brain.run_with_inputs(inputs)
    trace = res["trace"][:, 0, :]   # (T, N): drop batch dim
    pos = res["positions"]
    zones = res["zones"]
    colors = brain_zone_colors(brain)
    G = brain.config.grid_size

    n = len(timesteps)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), subplot_kw={"projection": "3d"})
    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, timesteps):
        t = min(t, trace.shape[0] - 1)
        title = f"t={t}  |  out={float(res['output'].ravel()[0]):.2f}"
        plot_3d_snapshot(pos, trace[t], zones, colors, G, title=title, ax=ax)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved snapshot figure -> {out_path}")


def save_animation(
    brain: PositronicBrain,
    inputs: np.ndarray,
    out_path: Path,
    fps: int = 6,
) -> None:
    res = brain.run_with_inputs(inputs)
    trace = res["trace"][:, 0, :]
    pos = res["positions"]
    zones = res["zones"]
    colors = brain_zone_colors(brain)
    G = brain.config.grid_size
    T = trace.shape[0]

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.clear()
        act = trace[frame]
        c = [colors[int(z)] for z in zones]
        sizes = 22 + act * 88
        alphas = 0.22 + act * 0.75
        rgba = np.array([to_rgba(cc, a) for cc, a in zip(c, alphas)])
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=rgba, s=sizes, edgecolors="none")
        ax.set_title(f"Positronic Brain — t={frame}   out={float(res['output'].ravel()[0]):.3f}")
        lim = G - 0.5
        ax.set_xlim(-0.5, lim); ax.set_ylim(-0.5, lim); ax.set_zlim(-0.5, lim)
        ax.view_init(elev=20, azim=42 + frame * 1.2)
        ax.grid(False)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000 // fps, repeat=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close()
    print(f"Saved animation -> {out_path}")


def brain_zone_colors(brain: PositronicBrain) -> List[str]:
    """Resolve the colour list for the brain's active zones."""
    from positronic_brain.zones import DEFAULT_ZONES
    by_name = {z.name: z.color for z in DEFAULT_ZONES}
    return [by_name.get(nm, "#9CA3AF") for nm in brain.config.zone_names]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="trained_models/positronic_brain_v2.pt")
    parser.add_argument("--input", type=float, nargs="+", default=None,
                        help="Explicit per-zone input values (one per zone)")
    parser.add_argument("--preset", type=str, default=None,
                        help=f"Named preset: {', '.join(PRESETS)}")
    parser.add_argument("--timesteps", type=str, default="0,6,12",
                        help="Comma-separated timesteps for snapshots")
    parser.add_argument("--animate", action="store_true", help="Save a rotating animation instead")
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--save", type=str, required=True, help="Output path (.png or .gif)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    brain = load_or_create_brain(args.model, device=args.device)

    if args.input is not None:
        Z = brain.config.num_zones
        vec = np.zeros(Z, dtype=np.float32)
        vec[: min(Z, len(args.input))] = args.input[:Z]
        inputs = vec
    else:
        preset = (args.preset or "cross").lower()
        named = PRESETS.get(preset, PRESETS["cross"])
        inputs = input_vector(brain, named)
        print(f"Using preset '{preset}': {named}")

    out = Path(args.save)
    if args.animate or out.suffix.lower() == ".gif":
        save_animation(brain, inputs, out, fps=args.fps)
    else:
        ts = [int(x) for x in args.timesteps.split(",")]
        save_snapshots(brain, inputs, ts, out)


if __name__ == "__main__":
    main()
