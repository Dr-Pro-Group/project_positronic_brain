"""
How many of the neurons are actually doing anything?

A network can have 32,768 units and behave like a much smaller one. If a unit's
firing rate barely moves as the input changes, it carries no information about the
input, and — because it also feeds the recurrence — it cannot pass any along either.
Counting parameters or units then overstates the machine that is really running.

This measures it directly. Units are ranked by how much their rate is modulated
across a held-out passage, the least-modulated k are CLAMPED to their own mean
*inside the recurrence* (so they neither compute nor transmit), and held-out
bits-per-char is re-measured. Sweeping k traces how much of the network can be
switched off before performance moves.

The comparison that makes it meaningful is the random-order control at matched k.
Clamping k units always removes some capacity; the question is whether clamping the
*least-modulated* k costs conspicuously less than clamping *any* k. If the two
curves sit on top of each other, every unit matters and the network's effective
size is its actual size. If the least-modulated curve stays flat while the random
curve climbs, the network is running on a small participating subset.

    python experiments/effective_n.py --checkpoint trained_models/brain_lm.pt
    python experiments/effective_n.py --grid-size 12 --train-steps 400

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import types
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.utils import get_device


def modulation_depth(model, data: torch.Tensor, seq_len: int, device,
                     windows: int = 8) -> torch.Tensor:
    """Per-unit standard deviation of firing rate across a held-out passage.

    A unit whose rate never changes carries nothing about the input, whatever its
    mean level happens to be — so spread, not magnitude, is the right statistic.
    """
    brain = model.brain
    model.eval()
    sums = torch.zeros(brain.num_neurons, device=device)
    sqs = torch.zeros(brain.num_neurons, device=device)
    n = 0
    with torch.no_grad():
        for w in range(windows):
            start = w * seq_len
            if start + seq_len + 1 > data.numel():
                break
            chunk = data[start:start + seq_len].to(device).unsqueeze(0)
            V = model.init_state(1)
            brain.stp_begin(1)
            for t in range(chunk.shape[1]):
                V = brain.integrate(V, model._token_current(chunk[:, t]),
                                    model.config.inner_steps)
                r = brain.firing_rate(V).squeeze(0)
                sums += r
                sqs += r * r
                n += 1
            brain.stp_end()
    mean = sums / max(n, 1)
    var = (sqs / max(n, 1)) - mean * mean
    return var.clamp(min=0).sqrt(), mean


def bpc_with_clamped(model, data: torch.Tensor, seq_len: int, device,
                     clamp_idx: torch.Tensor, clamp_val: torch.Tensor,
                     windows: int = 32) -> float:
    """Held-out bits-per-char with `clamp_idx` units pinned inside the recurrence.

    The clamp is applied to firing_rate, which is what feeds both the synaptic
    current and the read-out — so a clamped unit is frozen for the rest of the
    network too, not merely hidden from the head.
    """
    brain = model.brain
    original = brain.firing_rate

    def clamped_rate(self, V):
        r = original(V)
        if clamp_idx.numel():
            r = r.index_copy(-1, clamp_idx,
                             clamp_val[clamp_idx].to(r.dtype).expand(r.shape[:-1] + (clamp_idx.numel(),)))
        return r

    brain.firing_rate = types.MethodType(clamped_rate, brain)
    try:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for w in range(windows):
                start = w * seq_len
                if start + seq_len + 1 > data.numel():
                    break
                chunk = data[start:start + seq_len + 1].to(device).unsqueeze(0)
                logits, _ = model(chunk[:, :-1])
                ce = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, model.vocab_size), chunk[:, 1:].reshape(-1),
                    reduction="sum")
                total += float(ce.item())
                count += chunk[:, 1:].numel()
    finally:
        brain.firing_rate = original
    return (total / max(count, 1)) / math.log(2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help="a trained .pt to analyse; otherwise a fresh model is trained")
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--hf-chat", default="soda")
    p.add_argument("--hf-chat-limit", type=int, default=2000)
    p.add_argument("--fractions", default="0,0.25,0.5,0.75,0.9,0.95",
                   help="fractions of the population to clamp")
    p.add_argument("--control-repeats", type=int, default=3,
                   help="random-order draws per fraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--json", default="runs/effective_n.json")
    args = p.parse_args()

    device = get_device(args.device)
    torch.manual_seed(args.seed)

    if args.checkpoint:
        model, tok = BrainLanguageModel.load(args.checkpoint, device=device)
        _, val_text, _ = load_corpus_splits(hf_chat=args.hf_chat,
                                            hf_chat_limit=args.hf_chat_limit,
                                            builtin=True, repeats=60, seed=args.seed,
                                            val_frac=0.1, test_frac=0.0)
        val = torch.tensor(tok.encode(val_text), dtype=torch.long)
        print(f"[eff-N] loaded {args.checkpoint}: {model.brain.num_neurons} neurons")
    else:
        tr_text, val_text, _ = load_corpus_splits(hf_chat=args.hf_chat,
                                                  hf_chat_limit=args.hf_chat_limit,
                                                  builtin=True, repeats=60, seed=args.seed,
                                                  val_frac=0.1, test_frac=0.0)
        tok = CharTokenizer.from_text(tr_text)
        tr = torch.tensor(tok.encode(tr_text), dtype=torch.long)
        val = torch.tensor(tok.encode(val_text), dtype=torch.long)
        model = BrainLanguageModel(tok.vocab_size,
                                   LMConfig(grid_size=args.grid_size, embed_dim=64,
                                            inner_steps=3, seed=args.seed),
                                   device=device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"[eff-N] training grid-{args.grid_size} for {args.train_steps} steps", flush=True)
        model.train()
        for step in range(1, args.train_steps + 1):
            ix = torch.randint(0, tr.numel() - args.seq_len - 1, (args.batch_size,))
            batch = torch.stack([tr[i:i + args.seq_len + 1] for i in ix]).to(device)
            loss = model.loss_on(batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

    N = model.brain.num_neurons
    spread, mean_rate = modulation_depth(model, val, args.seq_len, device)
    order = torch.argsort(spread)                      # least-modulated first
    print(f"[eff-N] modulation depth: median {spread.median():.5f} "
          f"max {spread.max():.5f}; {(spread < 1e-4).float().mean():.1%} of units "
          f"have essentially none", flush=True)

    base = bpc_with_clamped(model, val, args.seq_len, device,
                            torch.empty(0, dtype=torch.long, device=device), mean_rate)
    print(f"[eff-N] baseline held-out bpc = {base:.4f}\n", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for frac in [float(f) for f in args.fractions.split(",")]:
        k = int(round(frac * N))
        if k == 0:
            rows.append({"fraction": 0.0, "k": 0, "targeted_bpc": base,
                         "random_bpc_mean": base, "random_bpc_std": 0.0})
            continue
        targeted = bpc_with_clamped(model, val, args.seq_len, device,
                                    order[:k].to(device), mean_rate)
        ctrl = []
        for rep in range(args.control_repeats):
            idx = torch.as_tensor(rng.choice(N, size=k, replace=False),
                                  dtype=torch.long, device=device)
            ctrl.append(bpc_with_clamped(model, val, args.seq_len, device, idx, mean_rate))
        rows.append({"fraction": frac, "k": k, "targeted_bpc": targeted,
                     "random_bpc_mean": statistics.fmean(ctrl),
                     "random_bpc_std": statistics.pstdev(ctrl) if len(ctrl) > 1 else 0.0,
                     "random_bpc_runs": ctrl})
        print(f"  clamp {frac:>5.0%} ({k:>6} units)  least-modulated {targeted:7.4f} "
              f"(+{targeted-base:6.4f})   random {statistics.fmean(ctrl):7.4f} "
              f"(+{statistics.fmean(ctrl)-base:6.4f})", flush=True)

    record = {"args": vars(args), "neurons": N, "baseline_bpc": base, "rows": rows}
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(record, fh, indent=2)

    # The verdict must not be read off the largest fraction alone: at extreme
    # clamping the two orders necessarily converge, because no network survives
    # losing 95% of its units however they are chosen. The informative quantity is
    # how far the targeted curve stays flat while the random curve climbs.
    TOL = 0.01          # bpc; below this the network is effectively unchanged
    free = [r for r in rows if r["fraction"] > 0 and r["targeted_bpc"] - base < TOL]
    largest_free = max(free, key=lambda r: r["fraction"]) if free else None
    ratios = [(r["random_bpc_mean"] - base) / max(r["targeted_bpc"] - base, 1e-9)
              for r in rows if r["fraction"] > 0]

    print("\n" + "=" * 72)
    if largest_free:
        k = largest_free["k"]
        eff = N - k
        print(f"{largest_free['fraction']:.0%} of units ({k:,} of {N:,}) can be clamped for "
              f"< {TOL} bpc, while clamping")
        print(f"the same number at random costs "
              f"{largest_free['random_bpc_mean'] - base:+.4f} bpc "
              f"({ratios[rows.index(largest_free) - 1]:.0f}x more).")
        print(f"=> effective population is at most ~{eff:,} units ({eff / N:.0%} of {N:,}).")
        print("   Parameter and neuron counts overstate the machine that is running.")
    else:
        print("No clamped fraction was free: the population participates broadly and")
        print("effective size is close to actual size.")
    print(f"peak targeted-vs-random cost ratio across the sweep: {max(ratios):.0f}x")
    print("=" * 72)
    print(f"[saved] {args.json}")


if __name__ == "__main__":
    main()
