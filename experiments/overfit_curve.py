#!/usr/bin/env python
"""
Overfit / generalization curves: brain vs matched LSTM vs dense RNN.

Hypothesis under test
---------------------
On a FIXED train set, with many epochs of training:

  * unconstrained models (LSTM / dense RNN) drive train loss down hard and
    eventually overfit (val CE rises or train–val gap explodes);
  * a large enough brain keeps generalizing (val CE stays near its best, gap
    stays smaller).

This is intentionally the opposite design of the 40k "wide training" sweep,
which grew the corpus so epochs stayed low. Here the corpus is frozen and we
train for many epochs.

    python experiments/overfit_curve.py \\
        --steps 50000 --log-every 500 --grid-size 12 \\
        --json runs/overfit_curve_g12.json

Run from the repository root. Prefer Mini/M1 overnight.
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.utils import get_device

# Reuse matched baselines from matched_experiment
from experiments.matched_experiment import (  # type: ignore
    CharCNN,
    CharLSTM,
    CharRNN,
    count_params,
    match_cnn_config,
    match_lstm_hidden,
    match_rnn_hidden,
)
from experiments.public_lm_eval import CharGPT, match_gpt_config  # type: ignore


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_batch(data: torch.Tensor, seq_len: int, batch_size: int, device) -> torch.Tensor:
    n = data.numel() - seq_len - 1
    ix = torch.randint(0, max(n, 1), (batch_size,))
    return torch.stack([data[i : i + seq_len + 1] for i in ix]).to(device)


@torch.no_grad()
def eval_ce(
    model: nn.Module,
    loss_on,
    data: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    seed: int,
    device,
) -> float:
    """Mean CE over a fixed sequence of windows (deterministic given seed)."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = data.numel() - seq_len - 1
    tot = 0.0
    for _ in range(n_batches):
        ix = torch.randint(0, max(n, 1), (batch_size,), generator=g)
        batch = torch.stack([data[int(i) : int(i) + seq_len + 1] for i in ix]).to(device)
        tot += float(loss_on(batch).item())
    model.train()
    return tot / n_batches


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--steps", type=int, default=50_000, help="total optimiser steps (many epochs)")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--eval-batches", type=int, default=32, help="windows for train/val CE probe")
    p.add_argument("--hf-chat", default="soda")
    p.add_argument("--hf-chat-limit", type=int, default=4000)
    p.add_argument("--repeats", type=int, default=60)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps")
    p.add_argument(
        "--models",
        default="brain,lstm,rnn,cnn,gpt",
        help="comma list: brain,lstm,rnn,cnn,gpt (keep full suite when possible)",
    )
    p.add_argument("--json", default="runs/overfit_curve.json")
    p.add_argument("--checkpoint-dir", default="", help="optional dir to save mid-run state dicts")
    args = p.parse_args()

    dev = get_device(args.device)
    # FIXED corpus — do not grow with steps (overfit stress).
    tr_t, va_t, _ = load_corpus_splits(
        hf_chat=args.hf_chat,
        hf_chat_limit=args.hf_chat_limit,
        builtin=True,
        repeats=args.repeats,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=0.0,
    )
    tok = CharTokenizer.from_text(tr_t)
    train_data = torch.tensor(tok.encode(tr_t), dtype=torch.long)
    val_data = torch.tensor(tok.encode(va_t), dtype=torch.long)
    chars_per_step = args.batch_size * args.seq_len
    epochs_total = args.steps * chars_per_step / max(train_data.numel(), 1)
    log(
        f"FIXED corpus train={train_data.numel():,} val={val_data.numel():,} "
        f"vocab={tok.vocab_size} device={dev}"
    )
    log(
        f"steps={args.steps}  chars/step={chars_per_step}  "
        f"≈{epochs_total:.2f} epochs over fixed train"
    )

    want = {m.strip().lower() for m in args.models.split(",") if m.strip()}

    # Build models (param-matched to brain).
    cfg = LMConfig(
        grid_size=args.grid_size,
        embed_dim=args.embed_dim,
        inner_steps=args.inner_steps,
        seed=args.seed,
    )
    brain = BrainLanguageModel(tok.vocab_size, cfg, device=dev)
    target = count_params(brain)
    h_lstm = match_lstm_hidden(tok.vocab_size, args.embed_dim, target)
    h_rnn = match_rnn_hidden(tok.vocab_size, args.embed_dim, target)
    emb_cnn, ch_cnn, n_cnn = match_cnn_config(tok.vocab_size, target)
    d_model, n_layer, n_head = match_gpt_config(
        tok.vocab_size, target, max_seq=max(args.seq_len, 256)
    )
    lstm = CharLSTM(tok.vocab_size, args.embed_dim, h_lstm).to(dev)
    rnn = CharRNN(tok.vocab_size, args.embed_dim, h_rnn).to(dev)
    cnn = CharCNN(
        tok.vocab_size, embed_dim=emb_cnn, channels=ch_cnn, n_layers=n_cnn
    ).to(dev)
    gpt = CharGPT(
        tok.vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        max_seq=max(args.seq_len, 256),
    ).to(dev)

    catalog = {
        "brain": (brain, brain.loss_on, count_params(brain)),
        "lstm": (lstm, lstm.loss_on, count_params(lstm)),
        "rnn": (rnn, rnn.loss_on, count_params(rnn)),
        "cnn": (cnn, cnn.loss_on, count_params(cnn)),
        "gpt": (gpt, gpt.loss_on, count_params(gpt)),
    }
    model_order = ("lstm", "rnn", "cnn", "gpt", "brain")
    for name in model_order:
        if name in want:
            log(f"  {name}: params={catalog[name][2]:,}")

    rec: Dict = {
        "args": vars(args),
        "corpus": {
            "train_chars": int(train_data.numel()),
            "val_chars": int(val_data.numel()),
            "vocab": tok.vocab_size,
            "chars_per_step": chars_per_step,
            "epochs_total": epochs_total,
        },
        "params": {k: catalog[k][2] for k in catalog if k in want},
        "curves": {},
        "summary": {},
    }

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    for name in model_order:
        if name not in want:
            continue
        model, loss_on, n_params = catalog[name]
        log(f"==== TRAIN {name} ({n_params:,} params) ====")
        torch.manual_seed(args.seed)
        if hasattr(torch, "mps") and dev.type == "mps":
            try:
                torch.mps.manual_seed(args.seed)
            except Exception:
                pass
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.train()
        t0 = time.time()
        curve: List[Dict] = []
        best_val = float("inf")
        best_step = 0
        running = 0.0
        n_run = 0

        for step in range(1, args.steps + 1):
            batch = make_batch(train_data, args.seq_len, args.batch_size, dev)
            loss = loss_on(batch)
            if not math.isfinite(float(loss.item())):
                log(f"  {name} DIVERGED at step {step}")
                rec["curves"][name] = curve
                rec["summary"][name] = {"diverged_at": step, "params": n_params}
                break
            opt.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running += float(loss.item())
            n_run += 1

            if step % args.log_every == 0 or step == args.steps:
                train_ce = eval_ce(
                    model, loss_on, train_data,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                    n_batches=args.eval_batches, seed=args.seed + 1000, device=dev,
                )
                val_ce = eval_ce(
                    model, loss_on, val_data,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                    n_batches=args.eval_batches, seed=args.seed + 2000, device=dev,
                )
                gap = val_ce - train_ce
                epochs = step * chars_per_step / max(train_data.numel(), 1)
                row = {
                    "step": step,
                    "epochs": epochs,
                    "train_ce": train_ce,
                    "val_ce": val_ce,
                    "gap": gap,
                    "train_ppl": math.exp(train_ce),
                    "val_ppl": math.exp(val_ce),
                    "recent_batch_loss": running / max(n_run, 1),
                    "wall_s": round(time.time() - t0, 1),
                }
                curve.append(row)
                running = 0.0
                n_run = 0
                if val_ce < best_val:
                    best_val = val_ce
                    best_step = step
                log(
                    f"  {name} step {step}/{args.steps}  ep={epochs:.2f}  "
                    f"train_ce={train_ce:.4f}  val_ce={val_ce:.4f}  gap={gap:+.4f}  "
                    f"val_ppl={math.exp(val_ce):.2f}"
                )
                # incremental save
                rec["curves"][name] = curve
                rec["summary"][name] = {
                    "params": n_params,
                    "best_val_ce": best_val,
                    "best_val_ppl": math.exp(best_val),
                    "best_step": best_step,
                    "final_train_ce": train_ce,
                    "final_val_ce": val_ce,
                    "final_gap": gap,
                    "late_degradation": val_ce - best_val,
                    "wall_min": round((time.time() - t0) / 60, 1),
                }
                os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
                with open(args.json, "w") as f:
                    json.dump(rec, f, indent=2)

        else:
            # completed without break
            final = curve[-1] if curve else {}
            rec["summary"][name] = {
                "params": n_params,
                "best_val_ce": best_val,
                "best_val_ppl": math.exp(best_val) if math.isfinite(best_val) else None,
                "best_step": best_step,
                "final_train_ce": final.get("train_ce"),
                "final_val_ce": final.get("val_ce"),
                "final_gap": final.get("gap"),
                "late_degradation": (final.get("val_ce", best_val) - best_val)
                if final
                else None,
                "wall_min": round((time.time() - t0) / 60, 1),
            }
            rec["curves"][name] = curve
            with open(args.json, "w") as f:
                json.dump(rec, f, indent=2)
            log(
                f"  {name} DONE  best_val_ppl={math.exp(best_val):.2f} @ step {best_step}  "
                f"final_gap={final.get('gap', float('nan')):+.4f}  "
                f"late_deg={final.get('val_ce', best_val) - best_val:+.4f}"
            )

    # Cross-model comparison at end
    log("==== SUMMARY (lower val better; late_deg>0 suggests overfit) ====")
    for name, s in rec["summary"].items():
        if "diverged_at" in s:
            log(f"  {name}: DIVERGED @ {s['diverged_at']}")
            continue
        log(
            f"  {name}: best_val_ppl={s['best_val_ppl']:.3f} @ {s['best_step']}  "
            f"final_gap={s['final_gap']:+.4f}  late_deg={s['late_degradation']:+.4f}  "
            f"({s['wall_min']} min)"
        )
    with open(args.json, "w") as f:
        json.dump(rec, f, indent=2)
    log(f"wrote {args.json}")


if __name__ == "__main__":
    main()
