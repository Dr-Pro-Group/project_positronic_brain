"""
Online / continual learning for the Positronic Brain.

This is the "learns while you interact with it" machinery. It follows the
**periodic gradient replay** strategy:

1. Every observation (a set of multimodal embeddings) is pushed into a bounded
   :class:`ReplayBuffer`.
2. Every ``train_every`` observations the :class:`OnlineLearner` runs a few
   gradient steps on a random replay batch, optimising a **self-supervised
   reconstruction** objective: the brain must regenerate the embeddings it was
   shown from its own final firing-rate state (an autoencoder over the recurrent
   dynamics). No labels are required.

Because the same internal brain state has to reconstruct *all* co-occurring
modalities, cross-modal associations emerge: drive the brain with an image and
its text/audio reconstruction heads predict the matching embedding (recall).

:class:`LiveBrain` wires encoders + brain + learner into a single ``perceive(...)``
call for interactive use.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
import torch

from .model import BrainConfig, PositronicBrain
from .encoders import Encoder, build_encoders, embedding_dims


Sample = Dict[str, np.ndarray]  # modality -> embedding vector


class ReplayBuffer:
    """Bounded FIFO buffer of multimodal embedding samples."""

    def __init__(self, capacity: int = 512, seed: int = 0):
        self.capacity = capacity
        self._buf: Deque[Sample] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, sample: Sample) -> None:
        self._buf.append({m: np.asarray(v, dtype=np.float32) for m, v in sample.items()})

    def sample(self, batch_size: int) -> List[Sample]:
        n = min(batch_size, len(self._buf))
        if n == 0:
            return []
        return self._rng.sample(list(self._buf), n)

    def all(self) -> List[Sample]:
        return list(self._buf)


@dataclass
class OnlineConfig:
    """Hyperparameters for periodic gradient replay."""

    lr: float = 5e-3
    train_every: int = 4         # observations between consolidation rounds
    train_steps: int = 2         # gradient steps per consolidation round
    batch_size: int = 16
    buffer_capacity: int = 512
    recon_weight: float = 1.0    # weight of the reconstruction loss
    modality_dropout: float = 0.5  # prob. of masking each input modality (enables recall)
    seed: int = 0


class OnlineLearner:
    """
    Periodic gradient-replay trainer for a :class:`PositronicBrain`.

    Usage:
        learner = OnlineLearner(brain)
        learner.observe({"image": emb_img, "text": emb_txt})   # call repeatedly
        # consolidation happens automatically every `train_every` observations
    """

    def __init__(self, brain: PositronicBrain, config: Optional[OnlineConfig] = None):
        self.brain = brain
        self.cfg = config or OnlineConfig()
        self.buffer = ReplayBuffer(self.cfg.buffer_capacity, seed=self.cfg.seed)
        self.opt = torch.optim.Adam(brain.parameters(), lr=self.cfg.lr)
        self._obs_count = 0
        self._rng = random.Random(self.cfg.seed + 1)
        self.loss_history: List[float] = []

    # --------------------------------------------------------------- observe
    def observe(self, sensory: Sample, consolidate: bool = True) -> Optional[float]:
        """
        Register one multimodal observation and (optionally) trigger periodic
        consolidation. Returns the latest consolidation loss, if one ran.
        """
        self.buffer.add(sensory)
        self._obs_count += 1
        loss = None
        if consolidate and self._obs_count % self.cfg.train_every == 0:
            loss = self.consolidate(self.cfg.train_steps, self.cfg.batch_size)
        return loss

    # ------------------------------------------------------------ consolidate
    def consolidate(self, steps: Optional[int] = None, batch_size: Optional[int] = None) -> float:
        """
        Run a few self-supervised gradient steps on replay batches (the "sleep"
        phase). Returns the mean loss over the steps.
        """
        steps = steps or self.cfg.train_steps
        batch_size = batch_size or self.cfg.batch_size
        if len(self.buffer) == 0:
            return float("nan")

        self.brain.train()
        device = self.brain.device
        losses: List[float] = []

        for _ in range(steps):
            batch = self.buffer.sample(batch_size)
            if not batch:
                continue
            # modalities present in every sample of the batch
            common = set(batch[0].keys())
            for s in batch[1:]:
                common &= set(s.keys())
            common &= set(self.brain.modalities)
            if not common:
                continue

            # Full targets for every common modality...
            targets = {
                m: torch.as_tensor(np.stack([s[m] for s in batch]), dtype=torch.float32,
                                   device=device)
                for m in common
            }
            # ...but only a (possibly masked) subset is fed in, so the brain must
            # learn to fill in / recall the missing modalities from the rest.
            common_list = sorted(common)
            keep = [m for m in common_list if self._rng.random() >= self.cfg.modality_dropout]
            if not keep:
                keep = [self._rng.choice(common_list)]
            sensory = {m: targets[m] for m in keep}

            out, recon = self.brain(None, sensory=sensory, return_recon=True)
            loss = sum(
                ((recon[m] - targets[m]) ** 2).mean() for m in common
            ) / len(common) * self.cfg.recon_weight

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        if not np.isnan(mean_loss):
            self.loss_history.append(mean_loss)
        return mean_loss


class LiveBrain:
    """
    High-level orchestrator: encoders + brain + online learner.

    Build one with :meth:`create` (which wires a brain whose sensory dimensions
    match the chosen encoders), then call :meth:`perceive` with raw data.
    """

    def __init__(self, brain: PositronicBrain, encoders: Dict[str, Encoder],
                 learner: OnlineLearner):
        self.brain = brain
        self.encoders = encoders
        self.learner = learner

    # ------------------------------------------------------------ factory
    @classmethod
    def create(
        cls,
        modalities: Optional[List[str]] = None,
        grid_size: int = 12,
        prefer_pretrained: bool = True,
        modality_to_zone: Optional[Dict[str, str]] = None,
        online_config: Optional[OnlineConfig] = None,
        device: str = "auto",
        seed: int = 42,
        brain_overrides: Optional[Dict] = None,
    ) -> "LiveBrain":
        """
        Create a ready-to-use live brain.

        Args:
            modalities: subset of {"image","text","audio","video"}.
            grid_size:  cube side (grid_size**3 neurons). 12 -> 1728, 16 -> 4096.
            prefer_pretrained: use real encoders when available, else fallback.
            modality_to_zone: override which zone each modality feeds.
        """
        modalities = modalities or ["image", "text", "audio"]
        encoders = build_encoders(modalities, prefer_pretrained=prefer_pretrained)
        dims = embedding_dims(encoders)

        default_map = {
            "image": "Visual",
            "video": "Visual",
            "audio": "Auditory",
            "text": "Association",
        }
        mod_zone = {m: (modality_to_zone or default_map).get(m, "Association") for m in modalities}

        cfg_kwargs = dict(
            grid_size=grid_size,
            sensory_embedding_dims=dims,
            modality_to_zone=mod_zone,
            seed=seed,
        )
        if brain_overrides:
            cfg_kwargs.update(brain_overrides)
        brain = PositronicBrain(BrainConfig(**cfg_kwargs), device=device)
        learner = OnlineLearner(brain, online_config)
        return cls(brain, encoders, learner)

    # ------------------------------------------------------------ encode
    def encode(self, **inputs) -> Sample:
        """Encode raw modality inputs (image=, text=, audio=, video=) to embeddings."""
        sample: Sample = {}
        for m, x in inputs.items():
            if x is None or m not in self.encoders:
                continue
            sample[m] = self.encoders[m].embed(x)
        return sample

    # ------------------------------------------------------------ perceive
    def perceive(self, learn: bool = True, **inputs) -> Dict[str, object]:
        """
        Encode raw inputs, drive the brain, optionally learn online, and return
        the brain state + reconstructions.

        Example:
            live.perceive(image="cat.jpg", text="a photo of a cat")
        """
        sample = self.encode(**inputs)
        if not sample:
            raise ValueError("perceive() needs at least one known modality input.")
        state = self.brain.run_multimodal(sample)
        loss = self.learner.observe(sample) if learn else None
        state["loss"] = loss
        state["encoded"] = sample
        return state

    def recall(self, **inputs) -> Dict[str, np.ndarray]:
        """
        Cross-modal recall: drive the brain with the given modality(ies) and
        return the decoded embeddings for *all* modalities (no learning).
        """
        sample = self.encode(**inputs)
        state = self.brain.run_multimodal(sample)
        return state["recon"]

    def save(self, path: str) -> None:
        self.brain.save(path)
