#!/usr/bin/env python
"""
The overnight program: screen organizations, then train the ones that earn it.

Phase A left one question standing. The recurrent core barely expands its input and
retains about three timesteps of it, and neither figure improves as units are added
— so the interesting question is no longer "does more biology help?" but "is there
an organization of this substrate that computes at all?". Screening answers that in
seconds per candidate; training answers whether it converts into held-out
prediction, and costs a quarter-hour per candidate. Doing them in that order is the
only reason this fits in a night.

Stages:
  1. screen    every candidate organization on expansion / memory capacity / growth
  2. train     the best few, plus every named biological mechanism, at a real budget
  3. ladder    the scaling curve for baseline and best organization, with the
               fixed-width read-out control so neuron count is not confounded
  4. report    one JSON holding all of it

Everything is appended to the JSON as it completes, so an interrupted run keeps
whatever it has finished.

    python experiments/long_program.py --grid-train 16 --steps 3000

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

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.screen_organization import candidates, expansion, growth, memory_capacity
from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.model import BrainConfig, PositronicBrain
from positronic_brain.utils import get_device


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(record: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)


def make_batch(data: torch.Tensor, seq_len: int, batch: int, device):
    ix = torch.randint(0, max(data.numel() - seq_len - 1, 1), (batch,))
    return torch.stack([data[i:i + seq_len + 1] for i in ix]).to(device)


@torch.no_grad()
def evaluate(model, data: torch.Tensor, seq_len: int, batch: int, device,
             max_windows: int = 256) -> float:
    model.eval()
    starts = list(range(0, max(data.numel() - seq_len - 1, 1), seq_len))[:max_windows]
    total, ntok = 0.0, 0
    for i in range(0, len(starts), batch):
        idx = starts[i:i + batch]
        chunk = torch.stack([data[s:s + seq_len + 1] for s in idx]).to(device)
        logits, _ = model(chunk[:, :-1])
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size), chunk[:, 1:].reshape(-1), reduction="sum")
        total += float(ce.item())
        ntok += chunk[:, 1:].numel()
    return (total / max(ntok, 1)) / math.log(2)


def train_one(tok, tr, va, grid: int, steps: int, seq_len: int, batch: int, lr: float,
              device, seed: int, overrides: Dict, readout_width=None) -> Dict:
    cfg = LMConfig(grid_size=grid, embed_dim=64, inner_steps=3, seed=seed,
                   readout_width=readout_width, brain_overrides=dict(overrides))
    model = BrainLanguageModel(tok.vocab_size, cfg, device=device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    model.train()
    diverged = None
    for step in range(1, steps + 1):
        loss = model.loss_on(make_batch(tr, seq_len, batch, device))
        if not math.isfinite(float(loss.item())):
            diverged = step
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
    wall = time.time() - t0
    bpc = float("nan") if diverged else evaluate(model, va, seq_len, batch, device)
    return {"grid": grid, "neurons": grid ** 3, "params": params, "steps": steps,
            "val_bpc": bpc, "diverged_at": diverged, "wall_s": round(wall, 1),
            "readout_width": readout_width}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-screen", type=int, default=12)
    p.add_argument("--grid-train", type=int, default=16)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--seeds", default="42")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--ladder", default="8,12,16")
    p.add_argument("--hf-chat", default="soda")
    p.add_argument("--hf-chat-limit", type=int, default=4000)
    p.add_argument("--device", default="mps")
    p.add_argument("--json", default="runs/long_program.json")
    args = p.parse_args()

    device = get_device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    record: Dict = {"args": vars(args), "screen": [], "trained": [], "ladder": []}

    # ---------------------------------------------------------------- 1. screen
    log(f"STAGE 1 — screening organizations at grid {args.grid_screen}")
    for cand in candidates(args.grid_screen):
        cfg = BrainConfig(grid_size=args.grid_screen, **cand["kw"])
        try:
            b = PositronicBrain(cfg, device="cpu")
            row = {"label": cand["label"], "kw": cand["kw"],
                   "growth": growth(b), "MC": memory_capacity(b), "expansion": expansion(b)}
        except Exception as exc:
            row = {"label": cand["label"], "kw": cand["kw"],
                   "error": f"{type(exc).__name__}: {exc}"}
        record["screen"].append(row)
        save(record, args.json)
        if "error" in row:
            log(f"  {cand['label']:<32} FAILED {row['error'][:40]}")
        else:
            log(f"  {cand['label']:<32} growth {row['growth']:7.3f}  "
                f"MC {row['MC']:6.2f}  expansion {row['expansion']:.3f}")

    usable = [r for r in record["screen"]
              if "error" not in r and r["expansion"] == r["expansion"]
              and r["MC"] == r["MC"] and r["growth"] < 10]
    usable.sort(key=lambda r: -(r["expansion"] + r["MC"] / 50.0))
    picks = usable[:args.top_k]
    log(f"selected {len(picks)} organizations to train: "
        f"{', '.join(r['label'] for r in picks)}")

    # -------------------------------------------------------------- 2. corpus
    log(f"loading corpus ({args.hf_chat}, limit {args.hf_chat_limit})")
    tr_txt, va_txt, _ = load_corpus_splits(hf_chat=args.hf_chat,
                                           hf_chat_limit=args.hf_chat_limit,
                                           builtin=True, repeats=60, seed=42,
                                           val_frac=0.1, test_frac=0.0)
    tok = CharTokenizer.from_text(tr_txt)
    tr = torch.tensor(tok.encode(tr_txt), dtype=torch.long)
    va = torch.tensor(tok.encode(va_txt), dtype=torch.long)
    log(f"vocab {tok.vocab_size}, train {tr.numel():,} chars, val {va.numel():,} chars")

    # ---------------------------------------------------------------- 3. train
    to_train = [{"label": "baseline", "kw": {}}]
    to_train += [{"label": r["label"], "kw": r["kw"]} for r in picks
                 if not r["label"].startswith("baseline")]
    # Every named mechanism gets a run at the real budget, screened or not: the
    # point of the flags is that they are measurable, and none has ever been given
    # more than 400 steps.
    for lbl, kw in [("stp", {"use_stp": True}), ("adaptation", {"use_adaptation": True}),
                    ("delays", {"use_delays": True}), ("divnorm", {"use_divnorm": True}),
                    ("laminar", {"use_laminar": True}),
                    ("homeostasis", {"use_homeostasis": True}),
                    ("dendrites", {"use_dendrites": True}),
                    ("oscillation", {"use_oscillation": True})]:
        if not any(t["kw"] == kw for t in to_train):
            to_train.append({"label": lbl, "kw": kw})

    log(f"STAGE 2 — training {len(to_train)} configs x {len(seeds)} seeds "
        f"at grid {args.grid_train}, {args.steps} steps")
    for cand in to_train:
        for seed in seeds:
            try:
                res = train_one(tok, tr, va, args.grid_train, args.steps, args.seq_len,
                                args.batch_size, args.lr, device, seed, cand["kw"])
            except Exception as exc:
                res = {"error": f"{type(exc).__name__}: {exc}"}
            row = {"label": cand["label"], "kw": cand["kw"], "seed": seed, **res}
            record["trained"].append(row)
            save(record, args.json)
            if "error" in row:
                log(f"  {cand['label']:<28} s{seed} FAILED {row['error'][:50]}")
            else:
                b = row["val_bpc"]
                log(f"  {cand['label']:<28} s{seed} bpc {b:.4f}"
                    f"{'  DIVERGED' if row['diverged_at'] else ''}  ({row['wall_s']:.0f}s)")

    # --------------------------------------------------------------- 4. ladder
    best = None
    done = [r for r in record["trained"] if r.get("val_bpc") == r.get("val_bpc")
            and r.get("val_bpc") is not None and not r.get("diverged_at")]
    if done:
        best = min(done, key=lambda r: r["val_bpc"])
        log(f"best trained config: {best['label']} at {best['val_bpc']:.4f} bpc")

    grids = [int(g) for g in args.ladder.split(",")]
    log(f"STAGE 3 — scaling ladder over grids {grids}, with fixed-read-out control")
    ladder_cfgs = [("baseline", {})]
    if best and best["label"] != "baseline":
        ladder_cfgs.append((best["label"], best["kw"]))
    for label, kw in ladder_cfgs:
        for grid in grids:
            for width in (None, 128):
                try:
                    res = train_one(tok, tr, va, grid, args.steps, args.seq_len,
                                    args.batch_size, args.lr, device, seeds[0], kw,
                                    readout_width=width)
                except Exception as exc:
                    res = {"error": f"{type(exc).__name__}: {exc}"}
                row = {"label": label, "kw": kw, "grid": grid,
                       "readout": "fixed128" if width else "growing", **res}
                record["ladder"].append(row)
                save(record, args.json)
                tag = f"{label}/grid{grid}/{'fixed' if width else 'growing'}"
                if "error" in row:
                    log(f"  {tag:<34} FAILED {row['error'][:40]}")
                else:
                    log(f"  {tag:<34} bpc {row['val_bpc']:.4f}  ({row['wall_s']:.0f}s)")

    log(f"PROGRAM COMPLETE -> {args.json}")


if __name__ == "__main__":
    main()
