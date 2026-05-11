"""
Zone definitions and spatial assignment for the Positronic Brain.

A "zone" is a spatially contiguous population of neurons specialised for a
functional role, loosely inspired by cortical / sub-cortical specialisation:

    Visual          - early sensory processing for visual-like inputs
    Auditory        - sensory processing for audio-like patterns
    Somatosensory   - touch / body-state processing
    Memory          - recurrent storage, pattern completion (hippocampus-like)
    Emotion         - valence / arousal modulation (limbic-like)
    Association     - cross-modal integration and higher-order dynamics

The design goal of v3 is to be *configurable and scalable* like an LLM: you can
declare any number of zones, give each one a 3D seed position inside the unit
cube, and every neuron is assigned to its nearest seed (a Voronoi tessellation).
This produces compact local territories with realistic boundaries that
generalise to any ``grid_size`` and any number of zones.

Backwards-compatibility: the historical zone names (Visual, Auditory, Memory,
Association) all remain present in the default configuration, so older code and
saved figures keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import numpy as np


class Zone(IntEnum):
    """Canonical functional zone identifiers for the default brain."""

    VISUAL = 0
    AUDITORY = 1
    SOMATOSENSORY = 2
    MEMORY = 3
    EMOTION = 4
    ASSOCIATION = 5


@dataclass(frozen=True)
class ZoneSpec:
    """
    Declarative description of a single functional zone.

    Attributes:
        name:  Human-readable label (e.g. "Visual").
        color: Hex colour for the visualizer (dark-theme friendly).
        seed:  (x, y, z) position in the normalised unit cube [0, 1]^3 used as
               the Voronoi centroid for spatial assignment.
        modality: True if the zone receives an external sensory/affective input
               channel. Association is purely integrative (``modality=False``)
               but still receives an input channel so the input vector stays
               aligned with the zone list.
    """

    name: str
    color: str
    seed: Tuple[float, float, float]
    modality: bool = True


# -----------------------------------------------------------------------------
# Default 6-zone multi-modal brain.
# Seeds are spread through the cube so each zone forms a compact territory.
# -----------------------------------------------------------------------------
DEFAULT_ZONES: List[ZoneSpec] = [
    ZoneSpec("Visual",        "#3B82F6", (0.15, 0.20, 0.80), modality=True),   # blue
    ZoneSpec("Auditory",      "#10B981", (0.15, 0.80, 0.80), modality=True),   # emerald
    ZoneSpec("Somatosensory", "#F472B6", (0.20, 0.50, 0.15), modality=True),   # pink (touch)
    ZoneSpec("Memory",        "#8B5CF6", (0.85, 0.25, 0.30), modality=True),   # violet
    ZoneSpec("Emotion",       "#EF4444", (0.80, 0.80, 0.25), modality=True),   # red (limbic)
    ZoneSpec("Association",   "#F59E0B", (0.50, 0.50, 0.55), modality=False),  # amber (hub)
]


def num_zones(zones: Optional[Sequence[ZoneSpec]] = None) -> int:
    """Number of zones in a configuration (defaults to the 6-zone brain)."""
    return len(zones if zones is not None else DEFAULT_ZONES)


def zone_names(zones: Optional[Sequence[ZoneSpec]] = None) -> List[str]:
    """List of zone names for a configuration."""
    return [z.name for z in (zones if zones is not None else DEFAULT_ZONES)]


def zone_colors(zones: Optional[Sequence[ZoneSpec]] = None) -> List[str]:
    """List of zone hex colours for a configuration."""
    return [z.color for z in (zones if zones is not None else DEFAULT_ZONES)]


# Backwards-compatible module-level constants (default configuration).
ZONE_NAMES: List[str] = zone_names()
ZONE_COLORS: List[str] = zone_colors()


def _seed_positions(zones: Sequence[ZoneSpec], grid_size: int) -> np.ndarray:
    """Scale each zone seed from the unit cube into grid coordinates."""
    scale = max(grid_size - 1, 1)
    return np.array([z.seed for z in zones], dtype=np.float32) * scale


def assign_zones(
    grid_size: int = 4,
    zones: Optional[Sequence[ZoneSpec]] = None,
) -> np.ndarray:
    """
    Assign every neuron in the cubic lattice to its nearest zone seed (Voronoi).

    Neurons are indexed in row-major order: ``index = x*G*G + y*G + z``.

    Args:
        grid_size: side length of the cubic volume (N = grid_size**3 neurons).
        zones: zone configuration; defaults to :data:`DEFAULT_ZONES`.

    Returns:
        (N,) int32 array of zone ids in ``[0, num_zones)``.
    """
    zspecs = list(zones if zones is not None else DEFAULT_ZONES)
    G = grid_size
    seeds = _seed_positions(zspecs, G)  # (Z, 3)

    xs, ys, zs = np.meshgrid(np.arange(G), np.arange(G), np.arange(G), indexing="ij")
    coords = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(np.float32)

    # Nearest seed (squared Euclidean distance) -> zone id
    d2 = ((coords[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)  # (N, Z)
    return d2.argmin(axis=1).astype(np.int32)


def get_zone_for_position(
    x: int,
    y: int,
    z: int,
    grid_size: int = 4,
    zones: Optional[Sequence[ZoneSpec]] = None,
) -> int:
    """Return the zone id for a single (x, y, z) grid position (Voronoi)."""
    zspecs = list(zones if zones is not None else DEFAULT_ZONES)
    seeds = _seed_positions(zspecs, grid_size)
    p = np.array([x, y, z], dtype=np.float32)
    d2 = ((seeds - p) ** 2).sum(axis=1)
    return int(d2.argmin())


@dataclass(frozen=True)
class ZoneInfo:
    """Per-zone metadata used by the visualizer and training code."""

    id: int
    name: str
    color: str
    neuron_count: int
    modality: bool


def get_zone_info(
    grid_size: int = 4,
    zones: Optional[Sequence[ZoneSpec]] = None,
) -> List[ZoneInfo]:
    """Compute per-zone statistics (neuron counts) for a given grid size."""
    zspecs = list(zones if zones is not None else DEFAULT_ZONES)
    zones_arr = assign_zones(grid_size, zspecs)
    info: List[ZoneInfo] = []
    for zid, spec in enumerate(zspecs):
        count = int(np.sum(zones_arr == zid))
        info.append(
            ZoneInfo(
                id=zid,
                name=spec.name,
                color=spec.color,
                neuron_count=count,
                modality=spec.modality,
            )
        )
    return info
