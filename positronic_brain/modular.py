"""
Modular multi-area Positronic Brain — load one area at a time.

Instead of one giant cube ``N = G³``, the system is a **graph of areas**, each
a small :class:`~positronic_brain.model.PositronicBrain`. Pathways are sparse
inter-area edges. Training can:

  * keep only the **active** area's parameters on device;
  * freeze / offload other areas to disk (hard-drive-as-RAM for *weights*);
  * grow total neuron count without growing the BPTT graph beyond one area.

This is the architecture-level answer to "load layers one by one" for a
recurrent 3D brain (there are no transformer layers).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import BrainConfig, PositronicBrain
from .utils import get_device


@dataclass
class AreaSpec:
    name: str
    grid_size: int = 12
    role: str = "association"  # sensory | association | memory | motor
    seed: int = 42
    brain_overrides: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "AreaSpec":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class PathwaySpec:
    src: str
    dst: str
    k_path: int = 4
    seed: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModularConfig:
    areas: List[AreaSpec] = field(default_factory=list)
    pathways: List[PathwaySpec] = field(default_factory=list)
    embed_dim: int = 64
    inner_steps: int = 2
    token_gain: float = 3.0
    seed: int = 42
    # When True, only ``active_area`` params require grad (others frozen).
    train_one_area: bool = True
    active_area: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "areas": [a.to_dict() for a in self.areas],
            "pathways": [p.to_dict() for p in self.pathways],
            "embed_dim": self.embed_dim,
            "inner_steps": self.inner_steps,
            "token_gain": self.token_gain,
            "seed": self.seed,
            "train_one_area": self.train_one_area,
            "active_area": self.active_area,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ModularConfig":
        areas = [AreaSpec.from_dict(a) for a in d.get("areas", [])]
        pathways = [PathwaySpec(**p) for p in d.get("pathways", [])]
        return cls(
            areas=areas,
            pathways=pathways,
            embed_dim=int(d.get("embed_dim", 64)),
            inner_steps=int(d.get("inner_steps", 2)),
            token_gain=float(d.get("token_gain", 3.0)),
            seed=int(d.get("seed", 42)),
            train_one_area=bool(d.get("train_one_area", True)),
            active_area=d.get("active_area"),
        )

    @classmethod
    def default_chain(cls, grid_size: int = 12, n_areas: int = 3, seed: int = 42) -> "ModularConfig":
        """Sensory → Association → Motor chain (minimal modular brain)."""
        roles = ["sensory", "association", "motor"]
        names = ["Sensory", "Association", "Motor"]
        n = max(2, min(n_areas, 3))
        areas = [
            AreaSpec(name=names[i], grid_size=grid_size, role=roles[i], seed=seed + i)
            for i in range(n)
        ]
        pathways = [
            PathwaySpec(src=areas[i].name, dst=areas[i + 1].name, k_path=4, seed=seed + 100 + i)
            for i in range(n - 1)
        ]
        return cls(areas=areas, pathways=pathways, seed=seed, active_area=areas[-1].name)


class Pathway(nn.Module):
    """Sparse learned map from source-area rates → destination I_ext."""

    def __init__(
        self,
        n_src: int,
        n_dst: int,
        k_path: int,
        seed: int,
        device: torch.device,
    ):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        k = max(1, min(int(k_path), n_src))
        # For each dst neuron, k random sources
        src = torch.randint(0, n_src, (n_dst * k,), generator=g)
        dst = torch.arange(n_dst).repeat_interleave(k)
        self.register_buffer("edge_src", src.to(device))
        self.register_buffer("edge_dst", dst.to(device))
        self.weight = nn.Parameter(torch.randn(n_dst * k, device=device) * 0.02)
        self.n_dst = n_dst

    def forward(self, rates_src: torch.Tensor) -> torch.Tensor:
        # rates_src: (B, N_src) → I: (B, N_dst)
        B = rates_src.shape[0]
        contrib = rates_src[:, self.edge_src] * self.weight.unsqueeze(0)  # (B, E)
        I = torch.zeros(B, self.n_dst, device=rates_src.device, dtype=rates_src.dtype)
        I.index_add_(1, self.edge_dst, contrib)
        return I


class ModularBrainLM(nn.Module):
    """Language model over a modular multi-area brain.

    Token path:
      embed → inject into first (sensory) area
      for each inner step: step all resident areas, apply pathways
      readout from last (motor) area rates
    """

    def __init__(
        self,
        vocab_size: int,
        config: ModularConfig,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = int(vocab_size)
        self._device = get_device(device) if isinstance(device, str) else device

        self.areas = nn.ModuleDict()
        self.area_N: Dict[str, int] = {}
        for spec in config.areas:
            overrides = dict(spec.brain_overrides)
            overrides.setdefault("recurrent_steps", config.inner_steps)
            bcfg = BrainConfig(grid_size=spec.grid_size, seed=spec.seed, **overrides)
            brain = PositronicBrain(bcfg, device=self._device)
            self.areas[spec.name] = brain
            self.area_N[spec.name] = brain.num_neurons

        self.area_order = [a.name for a in config.areas]
        self.input_area = self.area_order[0]
        self.output_area = self.area_order[-1]

        self.embed = nn.Embedding(self.vocab_size, config.embed_dim)
        n_in = self.area_N[self.input_area]
        n_out = self.area_N[self.output_area]
        self.token_in = nn.Linear(config.embed_dim, n_in)
        self.head = nn.Linear(n_out, self.vocab_size)

        self.pathways = nn.ModuleList()
        self._pathway_meta: List[Tuple[str, str]] = []
        for ps in config.pathways:
            if ps.src not in self.areas or ps.dst not in self.areas:
                raise ValueError(f"pathway {ps.src}->{ps.dst} references missing area")
            self.pathways.append(
                Pathway(
                    self.area_N[ps.src],
                    self.area_N[ps.dst],
                    ps.k_path,
                    ps.seed,
                    self._device,
                )
            )
            self._pathway_meta.append((ps.src, ps.dst))

        # MPS-safe: every parameter/buffer on the same device (areas already are).
        self.embed.to(self._device)
        self.token_in.to(self._device)
        self.head.to(self._device)
        for path in self.pathways:
            path.to(self._device)

        self.token_gain = float(config.token_gain)
        self.inner_steps = int(config.inner_steps)
        if config.train_one_area:
            self.set_active_area(config.active_area or self.output_area)
        else:
            # ensure LM IO always has grad so loss attaches
            for p in list(self.embed.parameters()) + list(self.token_in.parameters()) + list(self.head.parameters()):
                p.requires_grad = True

    # ---------------------------------------------------------------- memory
    def set_active_area(self, name: str) -> None:
        """Freeze all areas except ``name`` (pathways touching it stay trainable).

        Embed / token_in / head stay trainable always so the loss graph remains
        attached even when only a middle area is updated (frozen modules still
        propagate grads to their inputs).
        """
        if name not in self.areas:
            raise ValueError(f"unknown area {name!r}; have {self.area_order}")
        self.config.active_area = name
        for aname, area in self.areas.items():
            trainable = aname == name
            for p in area.parameters():
                p.requires_grad = trainable
        # pathways: train if either end is active
        for (src, dst), path in zip(self._pathway_meta, self.pathways):
            train_p = src == name or dst == name
            for p in path.parameters():
                p.requires_grad = train_p
        # Always train IO so CE loss has a grad_fn even for middle-area stages.
        for p in list(self.embed.parameters()) + list(self.token_in.parameters()) + list(self.head.parameters()):
            p.requires_grad = True

    def offload_area_to_disk(self, name: str, directory: str) -> str:
        """Save area state_dict to disk and free parameters from this module.

        After offload, the area is replaced by a lightweight placeholder that
        must be reloaded before use. Use between sequential training stages.
        """
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"area_{name}.pt")
        area = self.areas[name]
        payload = {
            "name": name,
            "state_dict": {k: v.detach().cpu() for k, v in area.state_dict().items()},
            "grid_size": area.config.grid_size,
            "seed": area.config.seed,
            "config": area.config.to_dict(),
        }
        torch.save(payload, path)
        # Replace with tiny dummy to free graph memory (caller reloads later)
        del self.areas[name]
        if self._device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return path

    def reload_area_from_disk(self, name: str, directory: str) -> None:
        path = os.path.join(directory, f"area_{name}.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        bcfg = BrainConfig.from_dict(payload["config"])
        brain = PositronicBrain(bcfg, device=self._device)
        brain.load_state_dict(payload["state_dict"])
        self.areas[name] = brain
        self.area_N[name] = brain.num_neurons

    def save_all_areas(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for name in list(self.areas.keys()):
            path = os.path.join(directory, f"area_{name}.pt")
            area = self.areas[name]
            torch.save(
                {
                    "name": name,
                    "state_dict": {k: v.detach().cpu() for k, v in area.state_dict().items()},
                    "config": area.config.to_dict(),
                },
                path,
            )
        with open(os.path.join(directory, "modular_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        # shared LM head
        torch.save(
            {
                "embed": self.embed.state_dict(),
                "token_in": self.token_in.state_dict(),
                "head": self.head.state_dict(),
                "pathways": [p.state_dict() for p in self.pathways],
                "vocab_size": self.vocab_size,
            },
            os.path.join(directory, "lm_shared.pt"),
        )

    # ---------------------------------------------------------------- forward
    def init_states(self, batch: int) -> Dict[str, torch.Tensor]:
        states = {}
        for name, area in self.areas.items():
            states[name] = torch.full(
                (batch, area.num_neurons),
                float(area.config.E_L),
                device=self._device,
            )
        return states

    def _inject(self, token_ids: torch.Tensor) -> torch.Tensor:
        e = self.embed(token_ids)
        return self.token_in(e) * self.token_gain  # (B, N_in)

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """tokens (B, T) → logits (B, T, V), final states."""
        B, T = tokens.shape
        if states is None:
            states = self.init_states(B)
        logits_list = []
        for t in range(T):
            I_tok = self._inject(tokens[:, t])
            # Rates from previous membrane state seed pathways *before* the first
            # step so a single inner_step still couples areas (otherwise pathways
            # only affect micro-step ≥ 2 and never reach the readout).
            rates = {
                name: area.firing_rate(states[name])
                for name, area in self.areas.items()
            }
            for _ in range(self.inner_steps):
                I = {
                    n: torch.zeros(B, self.area_N[n], device=self._device)
                    for n in self.areas
                }
                I[self.input_area] = I_tok
                for (src, dst), path in zip(self._pathway_meta, self.pathways):
                    if src in rates and dst in I:
                        I[dst] = I[dst] + path(rates[src])
                for name, area in self.areas.items():
                    states[name] = area.step(states[name], I[name])
                    rates[name] = area.firing_rate(states[name])

            logits_list.append(self.head(rates[self.output_area]))
        logits = torch.stack(logits_list, dim=1)
        return logits, states

    def loss_on(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(self._device)
        logits, _ = self.forward_tokens(tokens[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens[:, 1:].reshape(-1),
        )

    def count_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def total_neurons(self) -> int:
        return sum(self.area_N.values())
