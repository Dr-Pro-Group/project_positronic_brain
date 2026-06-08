"""
Data for the multi-stream specialization study.

Two paths:

1. :func:`synthetic_scenes` — a **counterbalanced** synthetic generator used for
   development, tests, and validating the specialization metrics before spending GPU.
   Each "scene" shares a latent **content** class across all streams (same content,
   different modality) plus a per-stream component. This is the design that lets us
   dissociate *modality* specialization from *content* specialization — the central
   confound of the study.

2. :func:`encode_corpus` — the real path: stream a public corpus and encode it once
   with frozen pretrained encoders (CLIP / Wav2Vec2 / fallback) into a cached embedding
   store. Kept thin and dependency-optional; the heavy lifting (sharding to WebDataset,
   balanced multi-stream sampling) is layered on top for HPC runs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def synthetic_scenes(
    stream_dims: Dict[str, int],
    n: int,
    n_content: int = 6,
    latent_dim: int = 8,
    noise: float = 0.25,
    seed: int = 0,
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    """Generate ``n`` counterbalanced multimodal scenes.

    Returns ``(scenes, content_labels)`` where each scene is
    ``{stream: embedding(dim,)}`` and all streams in a scene share a content class.
    Embedding = shared content signal (projected per stream) + per-stream component +
    noise — so a downstream zone can specialise for *stream* (routing) or *content*.
    """
    rng = np.random.default_rng(seed)
    content_latent = rng.standard_normal((n_content, latent_dim))
    proj = {s: rng.standard_normal((latent_dim, d)) for s, d in stream_dims.items()}
    bias = {s: rng.standard_normal(d) for s, d in stream_dims.items()}
    scenes, labels = [], []
    for _ in range(n):
        c = int(rng.integers(n_content))
        scene = {}
        for s, d in stream_dims.items():
            emb = content_latent[c] @ proj[s] + bias[s] + noise * rng.standard_normal(d)
            scene[s] = emb.astype(np.float32)
        scenes.append(scene)
        labels.append(c)
    return scenes, labels


def batch_scenes(scenes: List[Dict[str, np.ndarray]], idx, device="cpu") -> Dict[str, torch.Tensor]:
    """Stack a list of scene indices into ``{stream: (B, dim)}`` tensors."""
    streams = list(scenes[0].keys())
    return {
        s: torch.as_tensor(np.stack([scenes[i][s] for i in idx]),
                           dtype=torch.float32, device=device)
        for s in streams
    }


def per_stream_samples(scenes: List[Dict[str, np.ndarray]], device="cpu") -> Dict[str, torch.Tensor]:
    """Collect all scenes into ``{stream: (N, dim)}`` for the specialization metrics."""
    streams = list(scenes[0].keys())
    return {
        s: torch.as_tensor(np.stack([sc[s] for sc in scenes]),
                           dtype=torch.float32, device=device)
        for s in streams
    }


def encode_corpus(stream, items, prefer_pretrained: bool = True) -> np.ndarray:
    """Encode raw items of one stream into embeddings with a frozen encoder.

    Real-data path: ``stream`` is "image" / "text" / "audio" / "video"; ``items`` is an
    iterable of raw inputs. Uses a pretrained encoder when available (CLIP / Wav2Vec2),
    otherwise the deterministic :class:`FallbackEncoder`. Encode once, cache the result
    (e.g. as sharded WebDataset / .npy) and train on the cached embeddings.
    """
    from .encoders import build_encoder
    enc = build_encoder(stream, prefer_pretrained=prefer_pretrained)
    return np.stack([np.asarray(enc.embed(x), dtype=np.float32) for x in items])
