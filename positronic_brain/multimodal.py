"""
Multi-stream brain: different data streams enter through different 3D zones.

This is the substrate for the *emergent functional specialization* study. Each
input stream (e.g. an image, audio or text embedding, or a domain-specific text
stream) is injected as current into the neurons of a **distinct entry zone** of the
3D brain; the network settles; and per-stream decode heads reconstruct the stream
embeddings from the final firing-rate pattern. Trained with cross-modal masking
(feed a subset, predict all), the network learns cross-stream associations — and we
then ask whether the zones near each entry door have **spontaneously specialized**
for their stream (see :mod:`positronic_brain.specialization`).

The per-stream entry/exit machinery already lives in :class:`PositronicBrain`
(``modality_to_zone`` / ``_scatter_sensory`` / ``reconstruct``); this module makes
it an ergonomic, trainable model and — crucially for the science — makes the
**routing a first-class, swappable variable** via :meth:`MultiStreamBrain.reroute`
(the cross-modal-rewiring experiment).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .model import BrainConfig, PositronicBrain


@dataclass
class StreamSpec:
    """One input stream: its name, embedding dimension, and target entry zone."""
    name: str
    dim: int
    zone: str


class MultiStreamBrain(nn.Module):
    """A 3D brain driven by several data streams through distinct entry zones.

    Args:
        streams:        the input streams (name, embedding dim, entry zone).
        grid_size:      cube side; N = grid_size**3 neurons.
        inner_steps:    integration steps per perception ("settling depth").
        brain_overrides: BrainConfig flags (e.g. ``use_divnorm=True``).
    """

    def __init__(
        self,
        streams: List[StreamSpec],
        grid_size: int = 12,
        inner_steps: int = 4,
        seed: int = 42,
        brain_overrides: Optional[Dict] = None,
        device: Union[str, torch.device] = "auto",
    ):
        super().__init__()
        self.streams = list(streams)
        self.inner_steps = int(inner_steps)
        overrides = dict(brain_overrides or {})
        overrides.setdefault("recurrent_steps", inner_steps)
        cfg = BrainConfig(
            grid_size=grid_size,
            seed=seed,
            sensory_embedding_dims={s.name: s.dim for s in streams},
            modality_to_zone={s.name: s.zone for s in streams},
            **overrides,
        )
        self.brain = PositronicBrain(cfg, device=device)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> torch.device:
        return self.brain.device

    @property
    def routing(self) -> Dict[str, str]:
        """Current stream -> entry-zone mapping."""
        return dict(self.brain.config.modality_to_zone)

    def _prep(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in inputs.items():
            t = torch.as_tensor(v, dtype=torch.float32, device=self.device)
            out[k] = t.unsqueeze(0) if t.dim() == 1 else t
        return out

    # ------------------------------------------------------------------ core
    def perceive(self, inputs: Dict[str, torch.Tensor], return_rates: bool = False):
        """Drive the brain with ``{stream: (B, dim)}``, settle, decode all streams.

        Returns the per-stream reconstructions ``{stream: (B, dim)}`` and, if
        requested, the final firing rates ``(B, N)``.
        """
        inputs = self._prep(inputs)
        B = next(iter(inputs.values())).shape[0]
        I_ext = self.brain._scatter_sensory(inputs, B)
        V = torch.full((B, self.brain.num_neurons), self.brain.config.E_L,
                       device=self.device, dtype=torch.float32)
        self.brain.stp_begin(B)
        V = self.brain.integrate(V, I_ext, self.inner_steps)
        self.brain.stp_end()
        rates = self.brain.firing_rate(V)
        recon = self.brain.reconstruct(rates)
        return (recon, rates) if return_rates else recon

    def loss(self, inputs: Dict[str, torch.Tensor], modality_dropout: float = 0.5) -> torch.Tensor:
        """Cross-modal masked-reconstruction loss: feed a subset, predict all.

        With dropout > 0 the network must recall the missing streams from the
        present ones, which is what drives cross-stream association.
        """
        targets = self._prep(inputs)
        names = list(targets)
        keep = [n for n in names if self._rng.random() >= modality_dropout]
        if not keep:
            keep = [self._rng.choice(names)]
        recon = self.perceive({n: targets[n] for n in keep})
        return sum(((recon[n] - targets[n]) ** 2).mean() for n in names) / len(names)

    # ------------------------------------------------------- routing experiments
    def reroute(self, stream: str, new_zone: str) -> None:
        """Move a stream's entry door to a different zone (cross-modal rewiring).

        Rebuilds the stream's zone-index buffer and re-initialises its (zone-sized)
        encode projection; the decode head is zone-independent and kept. After a
        reroute the stream's encoder is naive and must be re-learned — exactly the
        developmental/cross-modal-plasticity manipulation we want to study.
        """
        brain = self.brain
        cfg = brain.config
        if new_zone not in cfg.zone_names:
            raise ValueError(f"unknown zone {new_zone!r}")
        if stream not in cfg.sensory_embedding_dims:
            raise ValueError(f"unknown stream {stream!r}")
        zone_id = cfg.zone_names.index(new_zone)
        zidx = np.where(brain.zones.cpu().numpy() == zone_id)[0]
        if zidx.size == 0:
            raise ValueError(f"zone {new_zone!r} has no neurons")
        zidx_t = torch.as_tensor(zidx, dtype=torch.long, device=brain.device)
        brain.register_buffer(f"_zoneidx_{stream}", zidx_t)
        emb_dim = int(cfg.sensory_embedding_dims[stream])
        brain.sensory_encode[stream] = nn.Linear(emb_dim, int(zidx_t.numel())).to(brain.device)
        cfg.modality_to_zone[stream] = new_zone

    # ------------------------------------------------------- analysis hooks
    @torch.no_grad()
    def zone_activity(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean firing rate per zone ``(B, n_zones)`` when driven by ``inputs``."""
        _, rates = self.perceive(inputs, return_rates=True)
        Z = self.brain.config.num_zones
        pooled = torch.zeros((rates.shape[0], Z), device=rates.device, dtype=rates.dtype)
        pooled = pooled.index_add(1, self.brain.zones, rates)
        counts = torch.bincount(self.brain.zones, minlength=Z).clamp(min=1).to(rates.dtype)
        return pooled / counts.unsqueeze(0)

    @torch.no_grad()
    def neuron_rates(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-neuron firing rates ``(B, N)`` when driven by ``inputs``."""
        _, rates = self.perceive(inputs, return_rates=True)
        return rates
