"""
Biomimetic generative language model — the 3D brain *as* the language generator.

This is a from-scratch generative architecture (no pretrained weights, no
transformer) in which the :class:`~positronic_brain.model.PositronicBrain` itself
produces text, one token at a time, through its **recurrent conductance-based
membrane dynamics**.

The idea
--------
A human cortex does not predict tokens with attention; it is a recurrent network
of spiking neurons whose *persistent internal state* carries context through
time. We mimic that:

1. A character is embedded and injected as **external current** into the neurons
   of a chosen "language" zone of the 3D brain.
2. The brain then **reverberates** for a few integration steps — this recurrent
   settling is the model's "thinking" between characters.
3. The **firing-rate pattern** of all neurons is read out into a probability
   distribution over the next character.
4. Crucially, the membrane potential ``V`` is **carried over** to the next step,
   so the brain's evolving 3D activity *is* the context/memory. There is no
   attention window — only the living state of the network.

Trained by next-character prediction with backpropagation-through-time, the brain
learns to generate language from its own dynamics.

Honest scope
------------
At ~1k–5k neurons, character-level, trained on a laptop, this behaves like a
small biological char-RNN: it produces novel, locally-coherent text, not
GPT-quality prose. But it is genuinely generative, recurrent, 3D and 100% from
scratch — and every part scales by raising ``grid_size`` and the data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import BrainConfig, PositronicBrain
from .utils import get_device


# --------------------------------------------------------------------- tokenizer
class CharTokenizer:
    """Minimal, from-scratch character-level tokenizer.

    Index 0 is a genuine **unknown** token (the Unicode replacement character
    ``\\ufffd``). Out-of-vocabulary characters are mapped to it rather than
    silently dropped, so held-out evaluation scores the *true* string instead of
    an altered (shortened) one — and the index-0 slot no longer aliases a real
    content character. The UNK char is a single code point, so ``decode`` of an
    UNK id emits exactly one character and stays in-vocabulary.
    """

    UNK = "�"  # U+FFFD REPLACEMENT CHARACTER

    def __init__(self, chars: Optional[List[str]] = None):
        # NOTE: ``__init__``/``from_dict`` preserve the given vocabulary EXACTLY
        # (no forced UNK prepend) so older checkpoints load with their original
        # index mapping intact. Only ``from_text`` reserves the UNK slot.
        self.itos: List[str] = list(chars) if chars else []
        self.stoi: Dict[str, int] = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer from text, reserving index 0 as a real UNK token.

        IMPORTANT: build this from the **training split only** so held-out
        characters that never appear in training are scored as UNK, exactly as
        a deployed model would see them.
        """
        chars = [c for c in sorted(set(text)) if c != cls.UNK]
        return cls([cls.UNK] + chars)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def unk_id(self) -> int:
        """Index of the UNK token if this tokenizer reserves one, else -1."""
        return self.stoi.get(self.UNK, -1)

    def encode(self, text: str) -> List[int]:
        # Map OOV characters to UNK (index 0) instead of dropping them.
        return [self.stoi.get(c, 0) for c in text]

    def unk_rate(self, text: str) -> float:
        """Fraction of characters in ``text`` that map to UNK (OOV diagnostic)."""
        if not text:
            return 0.0
        ids = self.encode(text)
        return sum(1 for i in ids if i == 0) / len(ids)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids if 0 <= int(i) < len(self.itos))

    def to_dict(self) -> Dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: Dict) -> "CharTokenizer":
        return cls(d["itos"])


# ------------------------------------------------------------------------ config
@dataclass
class LMConfig:
    """Configuration for a :class:`BrainLanguageModel`."""

    grid_size: int = 32          # cube side; N = grid_size**3 neurons ("cortex" size).
                                 # 32 -> 32,768 neurons: the largest that still trains
                                 # at a workable speed on a 16 GB Apple M1 Pro (MPS).
                                 # Construction is O(N) so far bigger brains *build*
                                 # fine; training compute is the real ceiling.
    embed_dim: int = 64          # character-embedding width
    inner_steps: int = 3         # brain integration steps per character ("thinking depth")
    input_zone: str = "Association"   # which 3D zone the language input drives
    token_gain: float = 3.0      # scale of injected character current
    state_leak: float = 1.0      # 1.0 = carry full membrane state between chars (0 = reset)
    grad_checkpoint: bool = False  # checkpoint the inner reverberation loop (memory<->compute)
    seed: int = 42
    brain_overrides: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "LMConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})


# ------------------------------------------------------------------------- model
class BrainLanguageModel(nn.Module):
    """
    A generative language model whose recurrent core is a 3D ``PositronicBrain``.

    Pipeline per character ``c`` with carried membrane state ``V``:

        e        = Embedding(c)                       # (B, embed_dim)
        I_zone   = Linear(e) * token_gain             # (B, n_zone_neurons)
        I_ext    = scatter I_zone onto language-zone neurons   # (B, N)
        repeat inner_steps:  V = brain.step(V, I_ext) # recurrent reverberation
        rates    = firing_rate(V)                     # (B, N)
        logits   = Linear(rates)                      # (B, vocab)

    ``V`` persists across characters, so the brain's evolving 3D activity is the
    model's entire context/memory (no attention, no fixed window).
    """

    def __init__(
        self,
        vocab_size: int,
        config: Optional[LMConfig] = None,
        device: Union[str, torch.device] = "auto",
    ):
        super().__init__()
        self.config = config or LMConfig()
        cfg = self.config
        self.vocab_size = int(vocab_size)
        self._device = get_device(device) if isinstance(device, str) else device

        torch.manual_seed(cfg.seed)

        # The 3D recurrent "cortex". Keep the brain's own integration depth
        # (recurrent_steps) consistent with the LM's per-character inner_steps so
        # there is a single, non-divergent reverberation depth — previously
        # recurrent_steps=12 sat unused while the LM hard-coded inner_steps=3.
        overrides = dict(cfg.brain_overrides)
        overrides.setdefault("recurrent_steps", cfg.inner_steps)
        brain_cfg = BrainConfig(grid_size=cfg.grid_size, seed=cfg.seed, **overrides)
        self.brain = PositronicBrain(brain_cfg, device=self._device)
        N = self.brain.num_neurons

        # Language-zone neuron indices (where characters are injected).
        zone_names = brain_cfg.zone_names
        zone_id = zone_names.index(cfg.input_zone) if cfg.input_zone in zone_names else 0
        zidx = np.where(self.brain.zones.cpu().numpy() == zone_id)[0]
        if zidx.size == 0:  # fall back to all neurons if the zone is empty
            zidx = np.arange(N)
        self.register_buffer("_lang_idx", torch.as_tensor(zidx, dtype=torch.long))

        # From-scratch trainable parameters of the language head.
        self.embed = nn.Embedding(self.vocab_size, cfg.embed_dim)
        self.token_in = nn.Linear(cfg.embed_dim, int(self._lang_idx.numel()))
        self.head = nn.Linear(N, self.vocab_size)

        # Optional per-neuron activity tracking for the metabolic sparse-coding
        # penalty (only accumulated when track_activity is set, to avoid cost).
        self.track_activity = False
        self._last_activity = None

        self.to(self._device)

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> torch.device:
        return self.brain.device

    @property
    def num_neurons(self) -> int:
        return self.brain.num_neurons

    def init_state(self, batch: int) -> torch.Tensor:
        """Fresh membrane state V (B, N) at resting potential."""
        return torch.full(
            (batch, self.num_neurons), self.brain.config.E_L,
            device=self.device, dtype=torch.float32,
        )

    def _token_current(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map a (B,) batch of character ids to (B, N) external drive."""
        e = self.embed(token_ids)                                   # (B, embed_dim)
        inj = self.token_in(e) * self.config.token_gain            # (B, n_zone)
        I_ext = torch.zeros((token_ids.shape[0], self.num_neurons),
                            device=self.device, dtype=torch.float32)
        return I_ext.index_add(1, self._lang_idx, inj)

    def _reverberate(self, V: torch.Tensor, I_ext: torch.Tensor) -> torch.Tensor:
        """Run ``inner_steps`` recurrent integration steps (the 'thinking').

        Delegates to the brain's shared :meth:`~PositronicBrain.integrate` loop so
        the standalone brain and the language model use one integration path.
        """
        return self.brain.integrate(V, I_ext, self.config.inner_steps)

    def step_token(self, V: torch.Tensor, token_ids: torch.Tensor):
        """
        Advance one character: inject current, reverberate, read out logits.

        Returns (V_next, logits) where logits is (B, vocab).

        When ``grad_checkpoint`` is set and we are training, the inner
        reverberation loop is wrapped in ``torch.utils.checkpoint`` so its
        intermediate membrane states are recomputed in the backward pass instead
        of being stored — trading ~1 extra forward for a large activation-memory
        saving, which is what lets ``inner_steps`` and batch size grow at
        ``grid_size`` 48-64+.
        """
        I_ext = self._token_current(token_ids)
        if self.config.grad_checkpoint and V.requires_grad:
            import torch.utils.checkpoint as _cp

            V = _cp.checkpoint(self._reverberate, V, I_ext, use_reentrant=False)
        else:
            V = self._reverberate(V, I_ext)
        rates = self.brain.firing_rate(V)
        logits = self.head(rates)
        # Optionally relax the carried state toward rest (state_leak < 1).
        if self.config.state_leak < 1.0:
            V = self.brain.config.E_L + self.config.state_leak * (V - self.brain.config.E_L)
        return V, logits

    # -------------------------------------------------------------------- train
    def forward(self, tokens: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        Run the brain over a (B, T) batch of character-id sequences.

        Returns:
            logits: (B, T, vocab) next-character logits at every position.
            state:  (B, N) final membrane potential (for truncated BPTT).
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        V = self.init_state(B) if state is None else state
        # Short-term-plasticity state spans the whole window (reset per forward;
        # the membrane state V may still be carried across windows by the caller).
        self.brain.stp_begin(B)
        logits_seq: List[torch.Tensor] = []
        act = None
        for t in range(T):
            V, logits = self.step_token(V, tokens[:, t])
            logits_seq.append(logits)
            if self.track_activity:
                r = self.brain.firing_rate(V).mean(0)      # (N,) per-neuron, mean over batch
                act = r if act is None else act + r
        self.brain.stp_end()
        if self.track_activity and act is not None:
            self._last_activity = act / T                  # (N,) per-neuron mean rate
        return torch.stack(logits_seq, dim=1), V

    def sparse_penalty(self, target_rate: float) -> torch.Tensor:
        """Metabolic sparse-coding penalty: mean over neurons of (rate − target)².

        Pushes every neuron's mean firing rate toward a low set-point, discouraging
        both dead units and saturation (efficient-coding regularisation). Requires
        ``track_activity`` to have been set before the forward pass.
        """
        if self._last_activity is None:
            return torch.zeros((), device=self.device)
        return ((self._last_activity - target_rate) ** 2).mean()

    def loss_on(self, tokens: torch.Tensor) -> torch.Tensor:
        """Next-character cross-entropy over a (B, T) batch (full BPTT)."""
        tokens = tokens.to(self.device)
        logits, _ = self.forward(tokens[:, :-1])
        target = tokens[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size), target.reshape(-1)
        )

    def loss_with_state(self, tokens: torch.Tensor, state: Optional[torch.Tensor] = None):
        """Next-char CE over a (B, T) batch starting from a carried ``state``.

        Returns ``(loss, V_final)``. The final membrane state is returned so the
        caller can detach it and feed it into the next batch — the basis of
        persistent cross-window (stateful) training, where the brain's evolving 3D
        activity becomes a genuine multi-window context instead of being reset to
        rest every window.
        """
        tokens = tokens.to(self.device)
        logits, V = self.forward(tokens[:, :-1], state=state)
        target = tokens[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), target.reshape(-1)
        )
        return loss, V

    def tbptt_step(
        self,
        tokens: torch.Tensor,
        optimizer,
        chunk: int = 64,
        grad_clip: float = 0.5,
    ) -> float:
        """
        One truncated-BPTT optimisation step over a (B, T) batch.

        The membrane state ``V`` is carried *forward* across the whole sequence,
        but the backward graph is cut every ``chunk`` characters by detaching
        ``V``. This decouples the trainable sequence length from activation
        memory: a length-1024 sequence with ``chunk=64`` costs the memory of a
        length-64 sequence while still propagating context (the state) across
        the full 1024 characters. Returns the mean per-character loss.

        Unlike :meth:`loss_on`, this method performs the optimiser update(s)
        itself (one ``backward``/``step`` per chunk), so call it directly in the
        training loop instead of ``loss_on``.
        """
        tokens = tokens.to(self.device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        V = self.init_state(B)
        self.brain.stp_begin(B)
        total, n = 0.0, 0
        # Walk the sequence in windows of `chunk` characters.
        for start in range(0, T - 1, chunk):
            end = min(start + chunk, T - 1)
            V = V.detach().requires_grad_(True)  # cut the graph, keep the state
            self.brain.stp_detach()              # carry STP state, cut its graph too
            logits_seq: List[torch.Tensor] = []
            for t in range(start, end):
                V, logits = self.step_token(V, tokens[:, t])
                logits_seq.append(logits)
            logits = torch.stack(logits_seq, dim=1)
            target = tokens[:, start + 1 : end + 1]
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), target.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.item()) * (end - start)
            n += end - start
        self.brain.stp_end()
        return total / max(n, 1)

    # ----------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        tokenizer: CharTokenizer,
        prompt: str = "",
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        stop: Optional[str] = None,
    ) -> str:
        """
        Autoregressively generate text from the brain's dynamics.

        The prompt is streamed through the brain to build up its membrane state,
        then new characters are sampled one at a time and fed back in.
        """
        self.eval()
        V = self.init_state(1)
        self.brain.stp_begin(1)
        ids = tokenizer.encode(prompt)

        # Warm up the brain state on the prompt (all but the last character).
        for tid in ids[:-1]:
            V, _ = self.step_token(V, torch.tensor([tid], device=self.device))

        last = ids[-1] if ids else 0
        out_ids: List[int] = []
        for _ in range(max_new_tokens):
            V, logits = self.step_token(V, torch.tensor([last], device=self.device))
            logits = logits[0] / max(temperature, 1e-5)
            if top_k is not None and 0 < top_k < self.vocab_size:
                vals, idx = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter_(0, idx, F.softmax(vals, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)
            last = int(torch.multinomial(probs, 1).item())
            out_ids.append(last)
            if stop and tokenizer.decode([last]) and stop in tokenizer.decode(out_ids[-len(stop):]):
                break
        self.brain.stp_end()
        return tokenizer.decode(out_ids)

    # -------------------------------------------------------------- persistence
    def save(self, path: str, tokenizer: Optional[CharTokenizer] = None) -> None:
        torch.save(
            {
                "lm_config": self.config.to_dict(),
                "vocab_size": self.vocab_size,
                "tokenizer": tokenizer.to_dict() if tokenizer else None,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Union[str, torch.device] = "auto"):
        """Load a model; returns (model, tokenizer_or_None)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = LMConfig.from_dict(ckpt.get("lm_config", {}))
        model = cls(ckpt["vocab_size"], cfg, device=device)
        model.load_state_dict(ckpt["state_dict"])
        tok = CharTokenizer.from_dict(ckpt["tokenizer"]) if ckpt.get("tokenizer") else None
        return model, tok
