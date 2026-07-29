#!/usr/bin/env python
"""
Two axes, each spanning two orders of magnitude.

The earlier ladder moved neuron count from 512 to 4,096 and read a slope off it.
That is an eightfold range, and biology suggests it is far too narrow to see
anything: C. elegans has 302 neurons and a Drosophila larva about 3,016, a tenfold
jump with no dramatic difference in what either animal can do. If capability
changes at all it may change by thresholds or by orders of magnitude, and a
short ladder would miss either.

So both axes are widened here and each is controlled for its own confound:

  NEURONS   512 -> 32,768 (64x) at a fixed step budget, run twice — once with the
            default read-out, which grows with the population, and once with a
            frozen projection into a fixed-width head. The gap between the two
            curves is the part of any "more neurons help" that is really "more
            read-out helps".

  TRAINING  400 -> 40,000 steps (100x) at fixed size, with the CORPUS SCALED so
            the number of passes over the data stays low. Without that the long
            runs stop measuring learning and start measuring memorisation: 40,000
            steps consumes 30M characters, and the default corpus is 3M.

Both are expensive; the script writes each result as it lands and can be resumed.

    python experiments/wide_sweep.py --axis neurons
    python experiments/wide_sweep.py --axis training
    python experiments/wide_sweep.py --axis both

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.utils import get_device


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save(rec: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)


def batch_of(data, seq_len, batch, device):
    ix = torch.randint(0, max(data.numel() - seq_len - 1, 1), (batch,))
    return torch.stack([data[i:i + seq_len + 1] for i in ix]).to(device)


@torch.no_grad()
def evaluate(model, data, seq_len, batch, device, max_windows=256) -> float:
    model.eval()
    starts = list(range(0, max(data.numel() - seq_len - 1, 1), seq_len))[:max_windows]
    tot, n = 0.0, 0
    for i in range(0, len(starts), batch):
        chunk = torch.stack([data[s:s + seq_len + 1] for s in starts[i:i + batch]]).to(device)
        logits, _ = model(chunk[:, :-1])
        tot += float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size), chunk[:, 1:].reshape(-1),
            reduction="sum").item())
        n += chunk[:, 1:].numel()
    return (tot / max(n, 1)) / math.log(2)


def run(tok, tr, va, grid, steps, seq_len, batch, lr, device, seed, overrides,
        readout_width=None) -> Dict:
    """Train one configuration, halving the batch on out-of-memory rather than dying.

    A fallback changes the compute budget, so it is recorded in the result: a run
    that had to shrink its batch is not directly comparable with one that did not.
    """
    while True:
        try:
            cfg = LMConfig(grid_size=grid, embed_dim=64, inner_steps=3, seed=seed,
                           readout_width=readout_width, brain_overrides=dict(overrides))
            model = BrainLanguageModel(tok.vocab_size, cfg, device=device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            t0 = time.time()
            model.train()
            diverged = None
            for step in range(1, steps + 1):
                loss = model.loss_on(batch_of(tr, seq_len, batch, device))
                if not math.isfinite(float(loss.item())):
                    diverged = step
                    break
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
            wall = time.time() - t0
            bpc = float("nan") if diverged else evaluate(model, va, seq_len, batch, device)
            return {"grid": grid, "neurons": grid ** 3, "steps": steps, "batch": batch,
                    "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "readout_width": readout_width, "val_bpc": bpc,
                    "diverged_at": diverged, "wall_s": round(wall, 1),
                    "chars_seen": batch * seq_len * steps}
        except RuntimeError as exc:
            if "memory" not in str(exc).lower() or batch <= 2:
                return {"grid": grid, "steps": steps, "error": str(exc)[:200]}
            batch //= 2
            log(f"    out of memory -> retrying at batch {batch}")
            if device.type == "mps":
                torch.mps.empty_cache()


def corpus(hf_limit: int, seed: int = 42):
    tr_txt, va_txt, _ = load_corpus_splits(hf_chat="soda", hf_chat_limit=hf_limit,
                                           builtin=True, repeats=60, seed=seed,
                                           val_frac=0.1, test_frac=0.0)
    tok = CharTokenizer.from_text(tr_txt)
    return (tok, torch.tensor(tok.encode(tr_txt), dtype=torch.long),
            torch.tensor(tok.encode(va_txt), dtype=torch.long))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", choices=["neurons", "training", "both"], default="both")
    p.add_argument("--grids", default="8,16,24,32")
    p.add_argument("--neuron-steps", type=int, default=3000)
    p.add_argument("--train-steps", default="400,4000,40000")
    p.add_argument("--train-grid", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=float, default=3.0,
                   help="corpus is grown so no run exceeds this many passes over it")
    p.add_argument("--device", default="mps")
    p.add_argument("--json", default="runs/wide_sweep.json")
    args = p.parse_args()

    device = get_device(args.device)
    rec: Dict = {"args": vars(args), "neurons": [], "training": []}

    if args.axis in ("neurons", "both"):
        tok, tr, va = corpus(4000)
        log(f"NEURON AXIS — corpus {tr.numel():,} chars, vocab {tok.vocab_size}")
        for grid in [int(g) for g in args.grids.split(",")]:
            for width in (None, 128):
                r = run(tok, tr, va, grid, args.neuron_steps, args.seq_len,
                        args.batch_size, args.lr, device, args.seed, {}, width)
                rec["neurons"].append({"readout": "fixed128" if width else "growing", **r})
                save(rec, args.json)
                tag = f"grid{grid}/{'fixed' if width else 'growing'}"
                log(f"  {tag:<20} {r.get('val_bpc', float('nan')):.4f}  "
                    f"N={r.get('neurons', 0):,}  batch={r.get('batch','?')}  "
                    f"({r.get('wall_s', 0)/60:.0f} min)")

    if args.axis in ("training", "both"):
        steps_list = [int(s) for s in args.train_steps.split(",")]
        # Grow the corpus with the longest run so passes stay bounded; a 40k-step run
        # over a 3M-character corpus is ten passes, and would score memorisation.
        need = max(steps_list) * args.batch_size * args.seq_len / args.max_epochs
        limit = max(4000, int(need / 750))          # ~750 chars per SODA conversation
        log(f"TRAINING AXIS — need ~{need/1e6:.1f}M chars for <= {args.max_epochs} "
            f"epochs at {max(steps_list):,} steps; requesting {limit:,} conversations")
        tok, tr, va = corpus(limit)
        log(f"  corpus {tr.numel():,} chars, vocab {tok.vocab_size}")
        for steps in steps_list:
            r = run(tok, tr, va, args.train_grid, steps, args.seq_len,
                    args.batch_size, args.lr, device, args.seed, {})
            ep = r.get("chars_seen", 0) / max(tr.numel(), 1)
            rec["training"].append({"epochs": round(ep, 2), **r})
            save(rec, args.json)
            log(f"  {steps:>6} steps  bpc {r.get('val_bpc', float('nan')):.4f}  "
                f"{ep:.2f} epochs  ({r.get('wall_s', 0)/60:.0f} min)")

    log(f"WIDE SWEEP COMPLETE -> {args.json}")


if __name__ == "__main__":
    main()
