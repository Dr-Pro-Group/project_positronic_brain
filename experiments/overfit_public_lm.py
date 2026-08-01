#!/usr/bin/env python
"""
Push public-LM training into the **overfit regime**.

Design (opposite of under-trained 8–12k step benches)
----------------------------------------------------
* FIXED train slice on disk (no growing corpus).
* Many epochs: tokens_seen ≫ train_tokens so dense models can memorize.
* Report **train_bpc and val_bpc** every eval — gap is the signal.
* Full suite: lstm, rnn, cnn, gpt, brain, brain_wm (CNN = local floor).

Hypothesis
----------
LSTM/GPT drive train_bpc down hard and val plateaus or rises (overfit).
Brain / brain_wm may show a smaller train–val gap (or just train slower).

    # reuse TinyStories disk store; cap train to 1M tokens for faster epochs
    python experiments/overfit_public_lm.py \\
        --work-dir data/llm_tinystories \\
        --train-tokens-cap 1000000 \\
        --steps 60000 --eval-every 1000 \\
        --models lstm,cnn,gpt,brain,brain_wm \\
        --device mps --json runs/overfit_public_tinystories.json

Epochs ≈ steps * batch * seq_len / train_tokens_cap.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.disk_data import MemmapTokenStore
from positronic_brain.language import BrainLanguageModel, LMConfig
from positronic_brain.subword import SubwordTokenizer
from positronic_brain.utils import get_device

from experiments.matched_experiment import (
    CharCNN,
    CharLSTM,
    CharRNN,
    STANDARD_MODELS,
    count_params,
    match_cnn_config,
    match_lstm_hidden,
    match_rnn_hidden,
)
from experiments.public_lm_eval import CharGPT, match_gpt_config
from experiments.llm_public_benchmark import (
    estimate_bytes_per_token,
    eval_store,
    make_suite,
    log,
)
from positronic_brain.checkpoints import (
    default_run_dir,
    save_model_bundle,
    save_tokenizer,
    write_run_meta,
)


class CappedMemmapStore(MemmapTokenStore):
    """Train store restricted to the first ``cap`` tokens of the train split."""

    def __init__(self, path: str, split: str = "train", cap: Optional[int] = None):
        super().__init__(path, split=split)
        if split == "train" and cap is not None and cap > 0:
            self.hi = min(self.hi, self.lo + int(cap))
            self.n = self.hi - self.lo


def train_overfit(
    name: str,
    model: nn.Module,
    loss_fn,
    train: MemmapTokenStore,
    val: MemmapTokenStore,
    *,
    steps: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    eval_every: int,
    max_windows: int,
    device,
    bytes_per_token: float,
) -> Tuple[Dict, List[Dict]]:
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    # Cosine decay toward 0.1 * lr to keep late training stable (esp. brain).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)

    model.train()
    t0 = time.time()
    curve: List[Dict] = []
    best_val = float("inf")
    best_step = 0
    best_state = None
    best_gap = float("inf")
    run_ce = 0.0
    n_run = 0
    diverged_at = None

    tokens_per_step = batch_size * seq_len
    train_tokens = max(train.n_tokens, 1)
    epochs_total = steps * tokens_per_step / train_tokens

    log(
        f"  {name} overfit setup: train_tok={train.n_tokens:,}  "
        f"tokens/step={tokens_per_step}  steps={steps}  ≈{epochs_total:.1f} epochs  "
        f"params={count_params(model):,}"
    )

    for step in range(1, steps + 1):
        batch = train.sample_batch(seq_len, batch_size, device)
        loss = loss_fn(batch)
        if not math.isfinite(float(loss.item())):
            log(f"  {name} DIVERGED at step {step}")
            diverged_at = step
            break
        opt.zero_grad()
        loss.backward()
        grads_ok = True
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grads_ok = False
                break
        if not grads_ok:
            log(f"  {name} NONFINITE GRAD at step {step} — skip")
            opt.zero_grad(set_to_none=True)
            continue
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        run_ce += float(loss.item())
        n_run += 1

        if step % eval_every == 0 or step == steps:
            # True train probe (held-out windows from train slice, not running CE)
            tr = eval_store(
                model, loss_fn, train,
                seq_len=seq_len, batch_size=batch_size,
                max_windows=max_windows, device=device,
                bytes_per_token=bytes_per_token,
            )
            va = eval_store(
                model, loss_fn, val,
                seq_len=seq_len, batch_size=batch_size,
                max_windows=max_windows, device=device,
                bytes_per_token=bytes_per_token,
            )
            avg = run_ce / max(n_run, 1)
            run_ce, n_run = 0.0, 0
            gap = va["bpc"] - tr["bpc"]
            epochs = step * tokens_per_step / train_tokens
            row = {
                "step": step,
                "epochs": round(epochs, 3),
                "lr": opt.param_groups[0]["lr"],
                "train_ce_running": avg,
                "train_bpc": tr["bpc"],
                "train_ppl": tr["ppl"],
                "val_bpc": va["bpc"],
                "val_ppl": va["ppl"],
                "gap_bpc": gap,  # positive = val worse than train (overfit signature)
                "wall_s": round(time.time() - t0, 1),
            }
            curve.append(row)
            log(
                f"  {name} step {step}/{steps} ep={epochs:.1f}  "
                f"train_bpc={tr['bpc']:.4f} val_bpc={va['bpc']:.4f}  "
                f"gap={gap:+.4f}  run_ce={avg:.4f}"
            )
            if math.isfinite(va["bpc"]) and va["bpc"] < best_val:
                best_val = va["bpc"]
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if math.isfinite(gap):
                best_gap = min(best_gap, gap) if best_gap != float("inf") else gap

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final summary stats from curve
    final = curve[-1] if curve else {}
    max_gap = max((r["gap_bpc"] for r in curve), default=float("nan"))
    # Did val rise after best? (classic overfit)
    val_rise = float("nan")
    if curve and best_step:
        after = [r for r in curve if r["step"] > best_step]
        if after:
            val_rise = max(r["val_bpc"] for r in after) - best_val

    summary = {
        "params": count_params(model),
        "train_tokens": train.n_tokens,
        "steps": steps if diverged_at is None else diverged_at,
        "epochs_planned": epochs_total,
        "epochs_ran": final.get("epochs"),
        "best_val_bpc": None if best_val == float("inf") else best_val,
        "best_step": best_step if best_val != float("inf") else None,
        "final_train_bpc": final.get("train_bpc"),
        "final_val_bpc": final.get("val_bpc"),
        "final_gap_bpc": final.get("gap_bpc"),
        "max_gap_bpc": max_gap,
        "val_rise_after_best": val_rise,
        "diverged_at": diverged_at,
        "wall_min": round((time.time() - t0) / 60, 2),
        "overfit_signature": (
            bool(final.get("gap_bpc") is not None and final.get("gap_bpc", 0) > 0.15)
            or (isinstance(val_rise, float) and math.isfinite(val_rise) and val_rise > 0.05)
        ),
    }
    log(
        f"  {name} DONE  best_val={summary['best_val_bpc']} @ {summary['best_step']}  "
        f"final train/val/gap={summary['final_train_bpc']}/"
        f"{summary['final_val_bpc']}/{summary['final_gap_bpc']}  "
        f"overfit_sig={summary['overfit_signature']}  wall={summary['wall_min']}m"
    )
    return summary, curve


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True, help="prepared disk store (scale_train prepare)")
    p.add_argument(
        "--train-tokens-cap",
        type=int,
        default=1_000_000,
        help="freeze first N train tokens (smaller = more epochs sooner)",
    )
    p.add_argument("--models", default="lstm,cnn,gpt,brain,brain_wm")
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--steps", type=int, default=60_000, help="long run to force overfit")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--max-windows", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps")
    p.add_argument("--json", default="runs/overfit_public_lm.json")
    p.add_argument(
        "--checkpoint-dir",
        default="",
        help="save best weights per model (default: checkpoints/<json basename>)",
    )
    args = p.parse_args()

    device = get_device(args.device)
    done = os.path.join(args.work_dir, "prepare_done.json")
    if not os.path.isfile(done):
        raise SystemExit(f"missing {done}")
    with open(done) as f:
        prep = json.load(f)

    tok = SubwordTokenizer.load(prep["tokenizer_path"])
    train = CappedMemmapStore(prep["mmap_path"], split="train", cap=args.train_tokens_cap)
    val = MemmapTokenStore(prep["mmap_path"], split="val")
    bpt = estimate_bytes_per_token(tok, train)

    probe = BrainLanguageModel(
        tok.vocab_size,
        LMConfig(grid_size=args.grid_size, embed_dim=args.embed_dim, inner_steps=args.inner_steps, seed=args.seed),
        device=device,
    )
    target = count_params(probe)
    del probe

    want = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    catalog = make_suite(
        tok.vocab_size, target, device,
        grid_size=args.grid_size, embed_dim=args.embed_dim,
        inner_steps=args.inner_steps, seed=args.seed, seq_len=args.seq_len,
        want=want,
    )

    epochs_est = args.steps * args.batch_size * args.seq_len / max(train.n_tokens, 1)
    log(
        f"OVERFIT PUBLIC LM  work_dir={args.work_dir}  train_cap={train.n_tokens:,}  "
        f"val={val.n_tokens:,}  steps={args.steps}  ≈{epochs_est:.1f} epochs  "
        f"target_params={target:,}  device={device}"
    )

    rec: Dict = {
        "protocol": {
            "description": "Fixed public LM slice, many epochs — push to overfit; track train vs val bpc",
            "work_dir": args.work_dir,
            "dataset": prep.get("dataset"),
            "train_tokens_cap": args.train_tokens_cap,
            "train_tokens_used": train.n_tokens,
            "steps": args.steps,
            "epochs_est": epochs_est,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "metric": "train_bpc vs val_bpc gap; val_rise_after_best",
            "note": "Not an intelligence leaderboard — generalization stress test",
        },
        "params": {n: count_params(catalog[n][0]) for n in want},
        "results": {},
        "curves": {},
        "checkpoints": {},
    }

    run_name = os.path.splitext(os.path.basename(args.json))[0]
    ckpt_root = args.checkpoint_dir or default_run_dir(".", run_name)
    os.makedirs(ckpt_root, exist_ok=True)
    save_tokenizer(tok, os.path.join(ckpt_root, "tokenizer.json"))
    rec["protocol"]["checkpoint_dir"] = ckpt_root
    log(f"checkpoints → {ckpt_root}")

    order = [m for m in ("lstm", "rnn", "cnn", "gpt", "brain", "brain_wm") if m in want]
    for name in order:
        model, loss_fn = catalog[name]
        log(f"==== OVERFIT TRAIN {name} ====")
        # Slightly tighter clip for brain stability on long runs
        clip = args.grad_clip if name not in ("brain", "brain_wm") else min(args.grad_clip, 0.35)
        lr = args.lr if name not in ("brain", "brain_wm") else args.lr * 0.75
        summary, curve = train_overfit(
            name, model, loss_fn, train, val,
            steps=args.steps, seq_len=args.seq_len, batch_size=args.batch_size,
            lr=lr, grad_clip=clip, eval_every=args.eval_every,
            max_windows=args.max_windows, device=device, bytes_per_token=bpt,
        )
        rec["results"][name] = summary
        rec["curves"][name] = curve
        try:
            ckpt_path = os.path.join(ckpt_root, f"{name}.pt")
            save_model_bundle(
                name, model, ckpt_path,
                tokenizer=tok, metrics=summary, protocol=rec["protocol"],
            )
            rec["checkpoints"][name] = ckpt_path
            log(f"  saved checkpoint {ckpt_path}")
        except Exception as exc:
            log(f"  WARNING: checkpoint save failed for {name}: {exc}")
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rec, f, indent=2)
        del model
        catalog[name] = (None, None)  # type: ignore
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    write_run_meta(ckpt_root, {
        "run_name": run_name,
        "json": args.json,
        "protocol": rec["protocol"],
        "results": rec["results"],
        "checkpoints": rec.get("checkpoints", {}),
    })
    log("==== OVERFIT SUMMARY (final gap_bpc = val - train; higher ⇒ more overfit) ====")
    for name, s in rec["results"].items():
        log(
            f"  {name:<10} best_val={s.get('best_val_bpc')}  "
            f"final_train={s.get('final_train_bpc')}  final_val={s.get('final_val_bpc')}  "
            f"gap={s.get('final_gap_bpc')}  max_gap={s.get('max_gap_bpc')}  "
            f"val_rise={s.get('val_rise_after_best')}  overfit={s.get('overfit_signature')}"
        )
    with open(args.json, "w") as f:
        json.dump(rec, f, indent=2)
    log(f"wrote {args.json}")


if __name__ == "__main__":
    main()
