"""
Emergent, input-anchored zones via activity-dependent plasticity — RESEARCH STUB.

Status: **design stub, unvalidated.** This module sketches the API for the
roadmap item in the paper (§7.4, "input-anchored emergent zones"): instead of
assigning functional zones by a static Voronoi partition (as
:class:`~positronic_brain.model.PositronicBrain` does today), we seed only the
*entry points* where sensory streams arrive — a "chiasm" for vision, an
"auditory-nerve" port for audio, a language port for text — and let zone identity
*emerge* under activity-dependent plasticity, in the spirit of self-organising
feature maps (Kohonen 1982; Willshaw & von der Malsburg 1976) and the
input-driven cortical map formation shown by rewiring experiments
(von Melchner, Pallas & Sur 2000).

It composes directly on the existing substrate:
  * the recurrent cell ``PositronicBrain.step(V, I_ext)`` provides the dynamics;
  * ``brain.edge_index`` / ``brain.edge_weight`` / ``brain.is_inhibitory`` are the
    sparse graph the plasticity rules edit;
  * ``positronic_brain.online`` (ReplayBuffer / OnlineLearner) supplies the
    consolidation loop that interleaves with structural plasticity.

What is real here vs. open research
------------------------------------
* ``inject`` and ``step_with_plasticity`` (port-anchored drive + Oja-normalised
  Hebbian magnitude updates) are implemented and runnable.
* ``emergent_zones`` (label each neuron by the port whose drive it most co-fires
  with) is implemented as the read-out of the self-organised map.
* ``structural_step`` (synaptic pruning + distance-prior synaptogenesis) is a
  first baseline; rewriting ``edge_index`` mid-training and keeping optimiser
  state coherent is the genuinely open part and is flagged inline.
We make no claim that the emergent maps match retinotopy/tonotopy yet — that is
exactly the experiment §7.4 proposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .model import BrainConfig, PositronicBrain
from .utils import get_device


# --------------------------------------------------------------------- configs
@dataclass
class PortSpec:
    """A sensory entry port — the anchored point where a modality arrives."""
    name: str                       # e.g. "chiasm", "auditory_nerve", "language"
    modality: str                   # "vision" | "audio" | "text" | ...
    embedding_dim: int              # width of the incoming encoder embedding
    center: Sequence[float] = (0.5, 0.5, 0.5)   # entry location in the unit cube
    radius: float = 2.0             # lattice radius of the seeded entry patch


@dataclass
class PlasticityConfig:
    rule: str = "hebbian"           # "static" | "hebbian" | "structural"
    eta: float = 1e-3               # Hebbian learning rate
    decay: float = 1e-2             # Oja-style weight decay (prevents runaway)
    act_ema: float = 0.99           # EMA factor for per-neuron activity tracking
    prune_thresh: float = 1e-3      # synaptic elimination below this |w|
    grow_per_step: int = 0          # synaptogenesis budget per structural step
    sigma_path: float = 1.75        # distance-prior scale for growth (matches wiring)


@dataclass
class EmergentZoneConfig:
    grid_size: int = 16
    ports: List[PortSpec] = field(default_factory=list)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    seed: int = 42


# --------------------------------------------------------------------- model
class EmergentZoneBrain(nn.Module):
    """A PositronicBrain whose functional zones are *grown*, not assigned.

    Zones are not passed in. We seed only the ``ports`` (entry patches), drive
    the network through them, and read the emergent organisation out of the
    activity statistics via :meth:`emergent_zones`.
    """

    def __init__(self, config: EmergentZoneConfig, device="auto"):
        super().__init__()
        self.config = config
        self._device = get_device(device) if isinstance(device, str) else device
        pl = config.plasticity
        self.eta, self.decay = pl.eta, pl.decay

        # The substrate: a standard cube with the distance prior, but its static
        # Voronoi zone labels are irrelevant here — organisation will emerge.
        self.brain = PositronicBrain(
            BrainConfig(grid_size=config.grid_size, seed=config.seed),
            device=self._device,
        )
        N = self.brain.num_neurons
        pos = self.brain.positions.detach().cpu().numpy()   # (N,3) integer coords
        G = config.grid_size

        # Map each port to the lattice neurons inside its entry patch, and learn a
        # projection from that modality's embedding onto those neurons.
        self.port_proj = nn.ModuleDict()
        self._ports: List[str] = []
        for p in config.ports:
            centre = np.asarray(p.center, dtype=np.float32) * (G - 1)
            d = np.linalg.norm(pos - centre[None, :], axis=1)
            idx = np.where(d <= p.radius)[0]
            if idx.size == 0:                       # fall back to nearest neuron
                idx = np.array([int(d.argmin())])
            self.register_buffer(f"_portidx_{p.name}",
                                 torch.as_tensor(idx, dtype=torch.long))
            self.port_proj[p.name] = nn.Linear(p.embedding_dim, int(idx.size))
            self._ports.append(p.name)

        # Activity statistics that drive emergence.
        self.register_buffer("_act_ema", torch.zeros(N))
        # _port_corr[p, i] accumulates how strongly neuron i co-fires when port p
        # is the driven modality -> argmax over p gives the emergent zone identity.
        self.register_buffer("_port_corr", torch.zeros(len(self._ports), N))
        self.to(self._device)

    # ------------------------------------------------------------- dynamics
    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.full((batch, self.brain.num_neurons), self.brain.config.E_L,
                          device=self._device, dtype=torch.float32)

    def inject(self, **modality_embeddings) -> torch.Tensor:
        """Scatter each provided modality's embedding onto its port's entry patch.

        Args are ``port_name=embedding`` (a 1-D or (B, emb_dim) tensor/array).
        Returns the (B, N) external drive.
        """
        B = 1
        for emb in modality_embeddings.values():
            e = torch.as_tensor(emb)
            if e.dim() == 2:
                B = e.shape[0]
                break
        I_ext = torch.zeros((B, self.brain.num_neurons), device=self._device)
        for name, emb in modality_embeddings.items():
            if name not in self.port_proj:
                continue
            e = torch.as_tensor(emb, device=self._device, dtype=torch.float32)
            if e.dim() == 1:
                e = e.unsqueeze(0).expand(B, -1)
            proj = self.port_proj[name](e)                       # (B, |patch|)
            idx = getattr(self, f"_portidx_{name}")
            I_ext = I_ext.index_add(1, idx, proj)
        return I_ext

    def step_with_plasticity(self, V: torch.Tensor, I_ext: torch.Tensor,
                             driven_port: Optional[str] = None) -> torch.Tensor:
        """One recurrent step plus the configured plasticity update.

        Reuses ``PositronicBrain.step`` for the conductance dynamics, then applies
        an Oja-normalised Hebbian update to the (existing) synaptic magnitudes and
        accumulates the per-port co-activation statistic used by
        :meth:`emergent_zones`.
        """
        V = self.brain.step(V, I_ext)
        rates = self.brain.firing_rate(V)                        # (B, N)
        with torch.no_grad():
            mean_rate = rates.mean(0)
            self._act_ema.mul_(self.config.plasticity.act_ema).add_(
                mean_rate, alpha=1 - self.config.plasticity.act_ema)
            if driven_port is not None and driven_port in self._ports:
                p = self._ports.index(driven_port)
                self._port_corr[p].add_(mean_rate)
            if self.config.plasticity.rule in ("hebbian", "structural"):
                self._hebbian_update(rates)
        return V

    @torch.no_grad()
    def _hebbian_update(self, rates: torch.Tensor) -> None:
        """Oja-normalised Hebbian update on existing edges:
        ``Δw_ij = η (⟨r_i r_j⟩ − λ w_ij)``, magnitudes kept non-negative (Dale's
        sign is fixed elsewhere). Strengthens synapses between co-active neurons
        while the decay term prevents runaway (Oja 1982)."""
        src, dst = self.brain.edge_index[0], self.brain.edge_index[1]
        pre = rates[:, src]                                       # (B, E)
        post = rates[:, dst]
        corr = (pre * post).mean(0)                              # (E,)
        w = self.brain.edge_weight.data
        w.add_(self.eta * (corr - self.decay * w)).clamp_(min=0.0)

    # ------------------------------------------------------- structural (experimental)
    @torch.no_grad()
    def structural_step(self) -> Dict[str, int]:
        """Prune weak synapses and grow new ones under the distance prior.

        Gated behind ``plasticity.rule == "structural"`` (an explicit experiment
        flag). Eliminates edges with magnitude below ``prune_thresh`` and adds
        ``grow_per_step`` new edges, with target ("postsynaptic") neurons drawn by
        recent activity and sources drawn under the same Gaussian distance bias as
        the original wiring (``exp(-d^2/2 sigma_path^2)``), modulated by activity.
        New edges inherit their presynaptic neuron's Dale sign, so the E/I
        constraint is preserved. Rebuilds ``edge_index`` / ``edge_sign`` (buffers)
        and ``edge_weight`` (parameter); returns ``{pruned, grown, edges}``.

        OPEN RESEARCH (flagged): rebuilding the ``edge_weight`` parameter
        **invalidates the optimiser's per-parameter state** for the new edge set,
        so the caller must rebuild the optimiser after each structural step, and
        the *stable schedule* for interleaving this with BPTT is exactly what §7.4
        proposes to study. This is a runnable baseline, not a validated mechanism.
        The growth sampler is O(grow_per_step * N) and is intended for modest
        budgets / per-area use, not whole-brain rewrites.
        """
        if self.config.plasticity.rule != "structural":
            raise RuntimeError(
                "structural_step requires plasticity.rule='structural' (it edits "
                "the graph topology); use rule='hebbian' for weight-only plasticity."
            )
        brain = self.brain
        pl = self.config.plasticity
        dev = brain.edge_index.device
        ei = brain.edge_index
        mag = brain.edge_weight.data.abs()

        # --- prune: synaptic elimination below threshold ---
        keep = mag >= pl.prune_thresh
        pruned = int((~keep).sum().item())
        src = ei[0][keep]
        dst = ei[1][keep]
        w_keep = brain.edge_weight.data[keep]

        # --- grow: activity-driven synaptogenesis under the distance prior ---
        grown = 0
        if pl.grow_per_step > 0:
            new_src, new_dst = self._sample_growth(pl.grow_per_step, src, dst)
            grown = int(new_src.numel())
            if grown:
                init_mag = 0.1 * float(brain.config.g_max) * torch.ones(grown, device=dev)
                src = torch.cat([src, new_src.to(dev)])
                dst = torch.cat([dst, new_dst.to(dev)])
                w_keep = torch.cat([w_keep, init_mag])

        # --- rebuild graph buffers + parameter (Dale sign from presynaptic) ---
        is_inh = brain.is_inhibitory.to(dev)
        new_sign = torch.where(is_inh[src],
                               -torch.ones(src.numel(), device=dev),
                               torch.ones(src.numel(), device=dev))
        brain.edge_index = torch.stack([src, dst]).long()
        brain.edge_sign = new_sign.float()
        brain.edge_weight = nn.Parameter(w_keep.clamp(min=0.0).contiguous())
        return {"pruned": pruned, "grown": grown, "edges": int(src.numel())}

    @torch.no_grad()
    def _sample_growth(self, n_grow: int, exist_src: torch.Tensor,
                       exist_dst: torch.Tensor):
        """Sample ``n_grow`` candidate (src, dst) edges: dst by recent activity,
        src under the Gaussian distance prior x activity, avoiding self-loops and
        existing edges. Sampling math runs on CPU (robust across CPU/MPS)."""
        brain = self.brain
        N = brain.num_neurons
        sigma = self.config.plasticity.sigma_path
        pos = brain.positions.detach().float().cpu()        # (N,3)
        act = self._act_ema.detach().float().cpu().clamp(min=0) + 1e-6   # (N,)
        existing = set(zip(exist_src.cpu().tolist(), exist_dst.cpu().tolist()))
        dst = torch.multinomial(act, n_grow, replacement=True)           # (n_grow,)
        src_out, dst_out = [], []
        for m in range(n_grow):
            d2 = (pos - pos[dst[m]]).pow(2).sum(-1)          # (N,)
            prob = torch.exp(-d2 / (2 * sigma ** 2)) * act
            prob[dst[m]] = 0.0                               # no self-loop
            if float(prob.sum()) <= 0:
                continue
            s = int(torch.multinomial(prob / prob.sum(), 1).item())
            if (s, int(dst[m])) in existing:                 # skip duplicates
                continue
            src_out.append(s)
            dst_out.append(int(dst[m]))
        return (torch.tensor(src_out, dtype=torch.long),
                torch.tensor(dst_out, dtype=torch.long))

    # ------------------------------------------------------------- read-out
    @torch.no_grad()
    def emergent_zones(self) -> torch.Tensor:
        """Label every neuron by the port it most co-activates with — the read-out
        of the self-organised map. Returns an (N,) tensor of port indices; compare
        against retinotopy/tonotopy expectations to test whether maps emerged."""
        if self._port_corr.abs().sum() == 0:
            raise RuntimeError("drive the brain through ports first "
                               "(step_with_plasticity with driven_port set).")
        return self._port_corr.argmax(dim=0)

    # ----------------------------------------------------------- convenience
    @classmethod
    def vision_audio_text(cls, grid_size: int = 16, device="auto",
                          vision_dim: int = 512, audio_dim: int = 768,
                          text_dim: int = 64) -> "EmergentZoneBrain":
        """A ready three-port brain: a visual 'chiasm', an 'auditory nerve', and a
        language port, seeded at separated corners of the cube so any topographic
        organisation must be *grown*, not inherited from the seed geometry."""
        ports = [
            PortSpec("chiasm", "vision", vision_dim, center=(0.15, 0.5, 0.5)),
            PortSpec("auditory_nerve", "audio", audio_dim, center=(0.85, 0.5, 0.5)),
            PortSpec("language", "text", text_dim, center=(0.5, 0.85, 0.5)),
        ]
        return cls(EmergentZoneConfig(grid_size=grid_size, ports=ports), device=device)
