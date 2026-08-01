#!/usr/bin/env python
"""
Neuron-count scaling study: how does held-out bits-per-char move as the brain grows?

Capacity scaling at FIXED data and compute: the corpus, tokenizer, sequence
length, optimizer, and step count are held constant; only ``grid_size`` (N =
grid_size**3 neurons) varies. So tokens-seen is identical across points and the
curve isolates the effect of *more neurons*, not more data.

For each grid size we train the full brain and report held-out bits-per-char
(content-disjoint SODA val), trainable params, wall-clock, and peak accelerator
memory. Optional controls:
  --no-conductance-too : also run the current-based variant at each N (does the
                         conductance cost shrink or grow with scale?)
  --frozen-too         : also run a frozen-reservoir variant at each N (does the
                         recurrent core contribute more as N grows, or does the
                         linear read-out keep doing the work?)

Run:
  python experiments/scaling_study.py --grids 6,8,10,12,16,20 \
      --hf-chat soda --hf-chat-limit 2000 --steps 500 --device mps \
      --json runs/scaling.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.utils import get_device


def make_batch(data, seq_len, batch_size, device):
    n = data.numel() - seq_len - 1
    ix = torch.randint(0, max(n, 1), (batch_size,))
    return torch.stack([data[i:i + seq_len + 1] for i in ix]).to(device)


@torch.no_grad()
def eval_bpc(model, data, seq_len, batch_size, device, max_windows=400):
    if data.numel() < seq_len + 2:
        return float("nan")
    model.eval()
    starts = list(range(0, data.numel() - seq_len - 1, seq_len))[:max_windows]
    tot, ntok = 0.0, 0
    for i in range(0, len(starts), batch_size):
        idx = starts[i:i + batch_size]
        batch = torch.stack([data[s:s + seq_len + 1] for s in idx]).to(device)
        logits, _ = model(batch[:, :-1])
        tgt = batch[:, 1:]
        tot += float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size), tgt.reshape(-1), reduction="sum"))
        ntok += tgt.numel()
    return (tot / max(ntok, 1)) / math.log(2)


def peak_mem_gb(device):
    if device.type == "mps":
        try:
            return torch.mps.driver_allocated_memory() / 1e9
        except Exception:
            return float("nan")
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def train_one(vocab, grid, tr, va, seq_len, batch_size, steps, lr, grad_clip,
              device, seed, overrides, frozen, readout_width=None):
    cfg = LMConfig(grid_size=grid, embed_dim=64, inner_steps=3, seed=seed,
                   readout_width=readout_width, brain_overrides=overrides)
    model = BrainLanguageModel(vocab, cfg, device=device)
    if frozen:
        model.brain.edge_weight.requires_grad_(False)
        model.brain.neuron_bias.requires_grad_(False)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = make_batch(tr, seq_len, batch_size, device)
        loss = model.loss_on(batch)
        opt.zero_grad(); loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    wall = time.time() - t0
    bpc = eval_bpc(model, va, seq_len, batch_size, device)
    mem = peak_mem_gb(device)
    return {"grid": grid, "neurons": grid ** 3, "edges": model.brain.num_edges,
            "params": params,
            "head_params": sum(q.numel() for q in model.head.parameters()),
            "val_bpc": bpc, "wall_s": round(wall, 1),
            "peak_mem_gb": round(mem, 2) if mem == mem else None,
            "it_s": round(steps / max(wall, 1e-6), 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grids", default="6,8,10,12,16,20")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--hf-chat", default="soda")
    p.add_argument("--hf-chat-limit", type=int, default=2000)
    p.add_argument("--no-conductance-too", action="store_true")
    p.add_argument("--fixed-readout", type=int, default=0,
                   help="also run the curve with a frozen random projection into a "
                        "read-out of this fixed width, so head capacity stops "
                        "growing with N and the neuron count is isolated")
    p.add_argument("--frozen-too", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", default="runs/scaling.json")
    args = p.parse_args()

    device = get_device(args.device)
    torch.manual_seed(args.seed)
    grids = [int(g) for g in args.grids.split(",")]

    # Stream the corpus ONCE; identical data + tokenizer for every grid size.
    tr_txt, va_txt, _ = load_corpus_splits(hf_chat=args.hf_chat,
                                           hf_chat_limit=args.hf_chat_limit,
                                           builtin=False, val_frac=0.1, test_frac=0.0,
                                           seed=args.seed)
    tok = CharTokenizer.from_text(tr_txt)
    tr = torch.tensor(tok.encode(tr_txt), dtype=torch.long)
    va = torch.tensor(tok.encode(va_txt), dtype=torch.long)
    print(f"[scaling] vocab={tok.vocab_size} train_tok={tr.numel()} val_tok={va.numel()} "
          f"device={device} steps={args.steps}", flush=True)

    variants = [("full", dict(), False)]
    if args.no_conductance_too:
        variants.append(("no_conductance", dict(use_conductance=False), False))
    if args.frozen_too:
        variants.append(("frozen_reservoir", dict(), True))
    if args.fixed_readout:
        variants.append(("fixed_readout", dict(), False))

    results = {v: [] for v, _, _ in variants}
    for grid in grids:
        for vname, ov, frozen in variants:
            width = args.fixed_readout if vname == "fixed_readout" else None
            r = train_one(tok.vocab_size, grid, tr, va, args.seq_len, args.batch_size,
                          args.steps, args.lr, args.grad_clip, device, args.seed, ov,
                          frozen, readout_width=width)
            results[vname].append(r)
            print(f"[scaling] {vname:<16} grid={grid:>2} N={r['neurons']:>6} "
                  f"params={r['params']:>9,} val_bpc={r['val_bpc']:.4f} "
                  f"{r['it_s']}it/s mem={r['peak_mem_gb']}GB", flush=True)

    out = {"args": vars(args), "vocab": tok.vocab_size, "val_tokens": int(va.numel()),
           "results": results}
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[scaling] wrote {args.json}")
    # Compact curve for the full brain.
    print("\ngrid  neurons   params     val_bpc   it/s   mem(GB)")
    for r in results["full"]:
        print(f"{r['grid']:>4} {r['neurons']:>8} {r['params']:>10,} "
              f"{r['val_bpc']:>8.4f} {r['it_s']:>6} {r['peak_mem_gb']}")


if __name__ == "__main__":
    main()
