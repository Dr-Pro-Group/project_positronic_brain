#!/usr/bin/env python
"""
Matched-budget experiment: is the brain's biology a help or a cost?

This is the decisive test behind the paper's §7.1 split verdict. We hold the
*data*, *tokenizer*, *sequence length*, *optimizer*, *step count*, and *trainable
parameter count* fixed, then ask:

    baseline  — does a plain char-LSTM with the SAME parameter budget reach a
                better / worse / equal held-out perplexity than the brain?

    ablation  — within the brain, how much does each biological constraint
                actually buy? We knock out, one at a time:
                  * Dale's law          (freely-signed synapses, no E/I split)
                  * conductance dynamics (current-based I_syn = gE - gI)
                  * spatial wiring       (random graph, same edge count)

All models train on identical batches (same seed -> same sampled indices) and are
scored on a held-out validation tail. Run from the repository root:

    python experiments/matched_experiment.py --mode all --grid-size 12 --steps 400

Defaults use the offline built-in conversational corpus so the comparison is
fully reproducible with no downloads. Pass --hf-chat soda to match the flagship
training corpus.
"""

from __future__ import annotations

import argparse
import math
import time
import types
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.utils import get_device


# --------------------------------------------------------------------- data utils
def make_batch(data: torch.Tensor, seq_len: int, batch_size: int, device) -> torch.Tensor:
    """Sample `batch_size` random contiguous chunks of length `seq_len + 1`."""
    n = data.numel() - seq_len - 1
    ix = torch.randint(0, max(n, 1), (batch_size,))
    chunks = torch.stack([data[i : i + seq_len + 1] for i in ix])
    return chunks.to(device)


def train_eval(
    model: nn.Module,
    loss_on,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    *,
    steps: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    seed: int,
    device,
    label: str,
    val_batches: int = 16,
) -> Tuple[float, float]:
    """Train `model` for `steps` and return (final_train_loss, val_perplexity).

    Re-seeds the RNG before the loop so every model sees the *same* sequence of
    sampled training batches -> a fair, matched comparison.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    running = 0.0
    last_avg = float("nan")
    for step in range(1, steps + 1):
        batch = make_batch(train_data, seq_len, batch_size, device)
        loss = loss_on(batch)
        opt.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        running += float(loss.item())
        if step % 20 == 0 or step == steps:
            last_avg = running / (step if step < 20 else 20)
            running = 0.0
            print(f"  [{label}] step {step:>4}/{steps}  loss={last_avg:.4f}  "
                  f"({step / max(time.time() - t0, 1e-6):.1f} it/s)", flush=True)

    # Held-out perplexity (fixed seed so every model is scored on identical chunks).
    model.eval()
    torch.manual_seed(seed + 999)
    tot = 0.0
    with torch.no_grad():
        for _ in range(val_batches):
            vb = make_batch(val_data, seq_len, batch_size, device)
            tot += float(loss_on(vb).item())
    val_ce = tot / val_batches
    ppl = math.exp(val_ce)
    print(f"  [{label}] -> val CE={val_ce:.4f}  perplexity={ppl:.2f}  "
          f"(trained in {time.time() - t0:.0f}s)", flush=True)
    return last_avg, ppl


# ------------------------------------------------------------------ LSTM baseline
class CharLSTM(nn.Module):
    """Plain character-level LSTM language model (the conventional baseline)."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden: int, layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, tokens: torch.Tensor):
        e = self.embed(tokens)
        h, _ = self.lstm(e)
        return self.head(h)

    def loss_on(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(next(self.parameters()).device)
        logits = self.forward(tokens[:, :-1])
        target = tokens[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size), target.reshape(-1)
        )


class CharRNN(nn.Module):
    """Plain character-level *vanilla* (Elman) RNN — the simplest dense baseline.

    A single ``tanh`` recurrent layer with no gating, the most direct dense
    counterpart to the brain's recurrent state. Parameter-matched like the LSTM
    so the comparison isolates *architecture*, not capacity.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden: int, layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden, num_layers=layers,
                          nonlinearity="tanh", batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, tokens: torch.Tensor):
        e = self.embed(tokens)
        h, _ = self.rnn(e)
        return self.head(h)

    def loss_on(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(next(self.parameters()).device)
        logits = self.forward(tokens[:, :-1])
        target = tokens[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size), target.reshape(-1)
        )


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_rnn_hidden(vocab: int, embed_dim: int, target: int, layers: int = 1) -> int:
    """Smallest vanilla-RNN hidden width whose param count >= target."""
    best_h, best_gap = 8, float("inf")
    for h in range(8, 8192, 4):
        probe = CharRNN(vocab, embed_dim, h, layers)
        gap = abs(count_params(probe) - target)
        if gap < best_gap:
            best_gap, best_h = gap, h
        if count_params(probe) > target:
            break
    return best_h


def match_lstm_hidden(vocab: int, embed_dim: int, target: int, layers: int = 1) -> int:
    """Smallest hidden width whose LSTM param count >= target (closest match)."""
    best_h, best_gap = 8, float("inf")
    for h in range(8, 4096, 4):
        probe = CharLSTM(vocab, embed_dim, h, layers)
        gap = abs(count_params(probe) - target)
        if gap < best_gap:
            best_gap, best_h = gap, h
        if count_params(probe) > target:
            break
    return best_h


# ------------------------------------------------------------- brain ablations
def ablate_no_dale(model: BrainLanguageModel) -> None:
    """Remove Dale's law: synapses become freely signed (no E/I sign constraint)."""
    brain = model.brain
    brain.signed_weights = types.MethodType(lambda self: self.edge_weight, brain)


def ablate_no_conductance(model: BrainLanguageModel) -> None:
    """Replace conductance dynamics with plain current-based synapses (no driving
    force): I_syn = gE - gI instead of gE*(E_E - V) + gI*(E_I - V)."""
    brain = model.brain

    def step_current(self, V, I_ext):
        src, dst = self.edge_index[0], self.edge_index[1]
        r = self.firing_rate(V)
        w = self.signed_weights()
        contrib = w.unsqueeze(0) * r[:, src]
        gE = torch.zeros_like(V).index_add_(1, dst, contrib.clamp(min=0.0))
        gI = torch.zeros_like(V).index_add_(1, dst, (-contrib).clamp(min=0.0))
        I_syn = gE - gI                       # current-based: no reversal driving force
        dV = -(V - self.config.E_L) + I_syn + I_ext + self.neuron_bias.unsqueeze(0)
        return V + self.alpha * dV

    brain.step = types.MethodType(step_current, brain)


def ablate_no_spatial(model: BrainLanguageModel, seed: int) -> None:
    """Destroy distance-biased wiring: shuffle edges into a random graph with the
    same neuron count, edge count, and Dale signs (only geometry is removed)."""
    brain = model.brain
    N = brain.num_neurons
    E = brain.edge_index.shape[1]
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, N, (E,), generator=g)
    dst = torch.randint(0, N, (E,), generator=g)
    same = src == dst                          # avoid self-loops
    dst[same] = (dst[same] + 1) % N
    brain.edge_index = torch.stack([src, dst]).to(brain.edge_index.device)
    # Recompute Dale signs from the (unchanged) presynaptic inhibitory mask.
    is_inh = brain.is_inhibitory
    src_dev = src.to(brain.edge_sign.device)
    brain.edge_sign = torch.where(is_inh[src_dev],
                                  -torch.ones(E, device=brain.edge_sign.device),
                                  torch.ones(E, device=brain.edge_sign.device))


def build_brain(vocab: int, cfg: LMConfig, device) -> BrainLanguageModel:
    return BrainLanguageModel(vocab, cfg, device=device)


# ------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["baseline", "ablation", "all"], default="all")
    p.add_argument("--grid-size", type=int, default=12,
                   help="brain cube side (12 -> 1,728 neurons; small & fast for a "
                        "matched sweep). Raise for a heavier, closer-to-flagship run.")
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--input-zone", default="Association")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--lstm-layers", type=int, default=1)
    # corpus (same knobs as train_language.py)
    p.add_argument("--text", default=None)
    p.add_argument("--hf", default=None)
    p.add_argument("--hf-limit", type=int, default=2000)
    p.add_argument("--hf-chat", default=None)
    p.add_argument("--hf-chat-limit", type=int, default=2000)
    p.add_argument("--repeats", type=int, default=60)
    p.add_argument("--rnn-layers", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds for mean±std error bars "
                        "(e.g. '42,43,44'); overrides --seed when given")
    p.add_argument("--json", default=None,
                   help="optional path to write the full results as JSON")
    args = p.parse_args()

    device = get_device(args.device)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed])

    # results[name] -> list of (params, ppl) across seeds
    agg: Dict[str, list] = {}
    for si, seed in enumerate(seeds):
        if len(seeds) > 1:
            print("\n" + "#" * 60 + f"\n# SEED {seed} ({si + 1}/{len(seeds)})\n" + "#" * 60)
        for name, (params, ppl) in run_once(args, seed, device).items():
            agg.setdefault(name, []).append((params, ppl))

    # ---- aggregated summary table ------------------------------------------
    def mean_std(xs):
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return m, 0.0
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, var ** 0.5

    print("\n" + "=" * 64)
    print(f"{'model':<28}{'params':>12}{'val ppl (mean±std)':>24}")
    print("-" * 64)
    summary: Dict[str, Dict] = {}
    for name, runs in agg.items():
        params = runs[0][0]
        ppls = [r[1] for r in runs]
        finite = [p for p in ppls if p == p]  # drop NaNs
        m, s = mean_std(finite) if finite else (float("nan"), 0.0)
        cell = f"{m:.2f} ± {s:.2f}" if len(seeds) > 1 else f"{m:.2f}"
        if len(finite) < len(ppls):
            cell += f"  (NaN×{len(ppls) - len(finite)})"
        print(f"{name:<28}{params:>12,}{cell:>24}")
        summary[name] = {"params": params, "ppl_mean": m, "ppl_std": s,
                         "ppl_runs": ppls}
    print("=" * 64)
    print("Lower perplexity is better. A matched LSTM/RNN that BEATS the brain "
          "means biology costs accuracy at this budget. Each ablation row shows "
          "what that constraint is worth (NaN = training diverged).")

    if args.json:
        import json
        with open(args.json, "w") as fh:
            json.dump({"seeds": seeds, "args": vars(args), "results": summary},
                      fh, indent=2)
        print(f"[exp] wrote {args.json}")


def run_once(args, seed: int, device) -> Dict[str, Tuple[int, float]]:
    """Run the full matched comparison for a single seed; return name->(params, ppl)."""
    torch.manual_seed(seed)

    # ---- shared corpus + tokenizer + CONTENT-DISJOINT train/val split ------
    # Use load_corpus_splits (not a positional tail-split): the built-in corpus
    # repeats ~50 fixed dialogues, so a tail-split puts the SAME sentences in both
    # train and val and the table measures memorisation, not generalisation.
    # load_corpus_splits partitions the seed dialogues (and any streamed
    # conversations) into disjoint groups BEFORE duplication, so no val material
    # appears in train. Tokenizer is built from TRAIN only.
    train_text, val_text, _ = load_corpus_splits(
        text_path=args.text, hf=args.hf, hf_limit=args.hf_limit,
        hf_chat=args.hf_chat, hf_chat_limit=args.hf_chat_limit,
        builtin=True, repeats=args.repeats, seed=seed,
        val_frac=args.val_frac, test_frac=0.0)
    tok = CharTokenizer.from_text(train_text)
    train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)
    seq_len = min(args.seq_len, max(2, train_data.numel() - 2))
    print(f"[exp] vocab={tok.vocab_size} train_tokens={train_data.numel()} "
          f"val_tokens={val_data.numel()} val_unk={tok.unk_rate(val_text):.2%} "
          f"device={device} seed={seed} (content-disjoint split)\n")

    def cfg(**overrides) -> LMConfig:
        return LMConfig(grid_size=args.grid_size, embed_dim=args.embed_dim,
                        inner_steps=args.inner_steps, input_zone=args.input_zone,
                        seed=seed, brain_overrides=overrides)

    common = dict(train_data=train_data, val_data=val_data, steps=args.steps,
                  seq_len=seq_len, batch_size=args.batch_size, lr=args.lr,
                  grad_clip=args.grad_clip, seed=seed, device=device)

    results: Dict[str, Tuple[int, float]] = {}

    # ---- reference brain (full biology) ------------------------------------
    print("=== Brain (full biology) ===")
    brain = build_brain(tok.vocab_size, cfg(), device)
    brain_params = count_params(brain)
    print(f"  neurons={brain.num_neurons} synapses={brain.brain.num_edges} "
          f"trainable params={brain_params}")
    _, brain_ppl = train_eval(brain, brain.loss_on, label="brain", **common)
    results["brain (full biology)"] = (brain_params, brain_ppl)

    # ---- matched dense baselines (LSTM + vanilla RNN) ----------------------
    if args.mode in ("baseline", "all"):
        print("\n=== LSTM baseline (parameter-matched) ===")
        hidden = match_lstm_hidden(tok.vocab_size, args.embed_dim, brain_params,
                                   args.lstm_layers)
        lstm = CharLSTM(tok.vocab_size, args.embed_dim, hidden, args.lstm_layers).to(device)
        lstm_params = count_params(lstm)
        print(f"  hidden={hidden} layers={args.lstm_layers} "
              f"trainable params={lstm_params} (target {brain_params})")
        _, lstm_ppl = train_eval(lstm, lstm.loss_on, label="lstm", **common)
        results["lstm (matched)"] = (lstm_params, lstm_ppl)

        print("\n=== Vanilla RNN baseline (parameter-matched) ===")
        rh = match_rnn_hidden(tok.vocab_size, args.embed_dim, brain_params,
                              args.rnn_layers)
        rnn = CharRNN(tok.vocab_size, args.embed_dim, rh, args.rnn_layers).to(device)
        rnn_params = count_params(rnn)
        print(f"  hidden={rh} layers={args.rnn_layers} "
              f"trainable params={rnn_params} (target {brain_params})")
        _, rnn_ppl = train_eval(rnn, rnn.loss_on, label="rnn", **common)
        results["dense RNN (matched)"] = (rnn_params, rnn_ppl)

    # ---- ablations ----------------------------------------------------------
    # We keep the runtime-patch ablations here (rather than the config-driven
    # BrainConfig switches used by train_language.py) because these are exactly
    # the patches that produced the reported matched-experiment numbers, so this
    # script remains bit-for-bit reproducible against the paper's table. The
    # BrainConfig switches (use_dale / use_conductance / spatial_wiring) are the
    # equivalent first-class path for full training runs.
    if args.mode in ("ablation", "all"):
        for name, fn in [
            ("no Dale's law", lambda m: ablate_no_dale(m)),
            ("no conductance", lambda m: ablate_no_conductance(m)),
            ("no spatial wiring", lambda m: ablate_no_spatial(m, seed)),
        ]:
            print(f"\n=== Ablation: {name} ===")
            abl = build_brain(tok.vocab_size, cfg(), device)
            fn(abl)
            _, ppl = train_eval(abl, abl.loss_on, label=name, **common)
            results[f"brain − {name}"] = (count_params(abl), ppl)

    return results


if __name__ == "__main__":
    main()
