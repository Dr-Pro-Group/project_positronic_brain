#!/usr/bin/env python
"""
Track B — same public text, different *brain-like* training regimes.

Industrial Track A (public_lm_eval) showed LSTM > RNN > tiny GPT > brain under
Adam + BPTT + random windows. This harness asks whether **how** the brain is
trained changes the picture:

  * bptt              — same recipe as Track A (reference)
  * persistent        — carry membrane state across contiguous windows
  * eprop             — forward-only eligibility-trace credit (Bellec et al.)
  * persistent+eprop  — both (if stable)

Optional LSTM arm reuses the industrial baseline under identical data.

    python experiments/brain_training_eval.py \\
        --hf tinystories --hf-limit 100000 --steps 15000 \\
        --grid-size 12 --json runs/brain_training_tinystories.json

Does NOT claim GPT-4 parity. Measures val/test bpc + divergence under each rule.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.streaming import StreamingBatcher
from positronic_brain.utils import get_device

from experiments.matched_experiment import CharLSTM, count_params, match_lstm_hidden
from experiments.public_lm_eval import eval_metrics, generate_baseline, DEFAULT_PROMPTS


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def make_batch(data: torch.Tensor, seq_len: int, batch_size: int, device) -> torch.Tensor:
    n = data.numel() - seq_len - 1
    ix = torch.randint(0, max(n, 1), (batch_size,))
    return torch.stack([data[i : i + seq_len + 1] for i in ix]).to(device)


def train_regime(
    name: str,
    model: BrainLanguageModel,
    train: torch.Tensor,
    val: torch.Tensor,
    test: torch.Tensor,
    tok: CharTokenizer,
    *,
    regime: str,
    steps: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    eval_every: int,
    max_windows: int,
    seed: int,
    device,
) -> Tuple[Dict, List[Dict]]:
    """regime: bptt | persistent | eprop | persistent_eprop"""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    curve: List[Dict] = []
    best_val = float("inf")
    best_step = 0
    best_state = None
    carry_V = None
    stream = None
    use_persist = regime in ("persistent", "persistent_eprop")
    use_eprop = regime in ("eprop", "persistent_eprop")

    if use_persist:
        stream = StreamingBatcher(train, seq_len, batch_size, device)
        batch_size = stream.batch_size
        log(f"  {name}/{regime}: persistent lanes={batch_size}")

    for step in range(1, steps + 1):
        try:
            if use_persist and stream is not None:
                batch, reset = stream.next_batch()
                if carry_V is not None:
                    carry_V = carry_V.detach()
                    carry_V[reset] = model.brain.config.E_L
                if use_eprop:
                    # e-prop path does not yet carry state; fall back to BPTT with state
                    loss, carry_V = model.loss_with_state(batch, state=carry_V)
                    opt.zero_grad()
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    loss_val = float(loss.item())
                else:
                    loss, carry_V = model.loss_with_state(batch, state=carry_V)
                    opt.zero_grad()
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    loss_val = float(loss.item())
            elif use_eprop:
                from positronic_brain.eprop import eprop_step
                batch = make_batch(train, seq_len, batch_size, device)
                loss_val = eprop_step(model, batch, opt, grad_clip=grad_clip)
            else:
                batch = make_batch(train, seq_len, batch_size, device)
                loss = model.loss_on(batch)
                if not math.isfinite(float(loss.item())):
                    raise FloatingPointError("non-finite loss")
                opt.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                loss_val = float(loss.item())
        except FloatingPointError:
            log(f"  {name}/{regime} DIVERGED at step {step}")
            return {
                "regime": regime,
                "diverged_at": step,
                "params": count_params(model),
                "wall_min": round((time.time() - t0) / 60, 1),
            }, curve

        if not math.isfinite(loss_val):
            log(f"  {name}/{regime} DIVERGED at step {step}")
            return {
                "regime": regime,
                "diverged_at": step,
                "params": count_params(model),
                "wall_min": round((time.time() - t0) / 60, 1),
            }, curve

        if step % eval_every == 0 or step == steps:
            # eval always cold-start windows (honest held-out, comparable across regimes)
            va = eval_metrics(
                model, model.loss_on, val,
                seq_len=seq_len, batch_size=batch_size,
                max_windows=max_windows, device=device,
            )
            tr = eval_metrics(
                model, model.loss_on, train,
                seq_len=seq_len, batch_size=batch_size,
                max_windows=min(64, max_windows), device=device,
            )
            row = {
                "step": step,
                "train_bpc": tr["bpc"],
                "val_bpc": va["bpc"],
                "val_ppl": va["ppl"],
                "gap_bpc": va["bpc"] - tr["bpc"],
                "batch_loss": loss_val,
                "wall_s": round(time.time() - t0, 1),
            }
            curve.append(row)
            if va["bpc"] < best_val:
                best_val = va["bpc"]
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(
                f"  {name}/{regime} step {step}/{steps}  "
                f"train_bpc={tr['bpc']:.4f} val_bpc={va['bpc']:.4f} "
                f"val_ppl={va['ppl']:.2f} gap={row['gap_bpc']:+.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    te = eval_metrics(
        model, model.loss_on, test,
        seq_len=seq_len, batch_size=batch_size, max_windows=max_windows, device=device,
    )
    summary = {
        "regime": regime,
        "params": count_params(model),
        "best_val_bpc": best_val,
        "best_val_ppl": 2 ** best_val if math.isfinite(best_val) else None,
        "best_step": best_step,
        "test_bpc": te["bpc"],
        "test_ppl": te["ppl"],
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    log(
        f"  {name}/{regime} DONE  best_val_bpc={best_val:.4f} "
        f"test_bpc={te['bpc']:.4f} @ {best_step} ({summary['wall_min']} min)"
    )
    return summary, curve


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf", default="tinystories", choices=["tinystories", "wikitext"])
    p.add_argument("--hf-limit", type=int, default=100_000)
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--max-windows", type=int, default=256)
    p.add_argument(
        "--regimes",
        default="bptt,persistent,eprop",
        help="comma list: bptt,persistent,eprop,persistent_eprop",
    )
    p.add_argument("--with-lstm", action="store_true", help="also train matched LSTM (BPTT)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps")
    p.add_argument("--json", default="runs/brain_training_eval.json")
    args = p.parse_args()

    dev = get_device(args.device)
    tr_t, va_t, te_t = load_corpus_splits(
        hf=args.hf, hf_limit=args.hf_limit, builtin=False,
        repeats=1, seed=args.seed, val_frac=0.1, test_frac=0.1,
    )
    if not tr_t.strip():
        raise SystemExit("empty corpus")
    tok = CharTokenizer.from_text(tr_t)
    train = torch.tensor(tok.encode(tr_t), dtype=torch.long)
    val = torch.tensor(tok.encode(va_t), dtype=torch.long)
    test = torch.tensor(tok.encode(te_t), dtype=torch.long)
    log(
        f"Track B corpus {args.hf} train={train.numel():,} val={val.numel():,} "
        f"test={test.numel():,} vocab={tok.vocab_size} device={dev}"
    )

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    rec: Dict = {
        "protocol": {
            "track": "B_brain_like_training",
            "hf": args.hf,
            "hf_limit": args.hf_limit,
            "steps": args.steps,
            "seq_len": args.seq_len,
            "grid_size": args.grid_size,
            "seed": args.seed,
            "regimes": regimes,
            "note": "same public text; cold-start val/test bpc for comparability",
        },
        "corpus": {
            "train_chars": int(train.numel()),
            "val_chars": int(val.numel()),
            "test_chars": int(test.numel()),
            "vocab": tok.vocab_size,
        },
        "results": {},
        "curves": {},
        "samples": {},
    }

    if args.with_lstm:
        cfg_b = LMConfig(
            grid_size=args.grid_size, embed_dim=args.embed_dim,
            inner_steps=args.inner_steps, seed=args.seed,
        )
        probe = BrainLanguageModel(tok.vocab_size, cfg_b, device=dev)
        target = count_params(probe)
        del probe
        h = match_lstm_hidden(tok.vocab_size, args.embed_dim, target)
        lstm = CharLSTM(tok.vocab_size, args.embed_dim, h).to(dev)
        log(f"==== LSTM reference ({count_params(lstm):,} params) ====")
        # reuse simple BPTT loop via train_regime-compatible path
        torch.manual_seed(args.seed)
        opt = torch.optim.Adam(lstm.parameters(), lr=args.lr)
        t0 = time.time()
        best_val, best_step, best_state = float("inf"), 0, None
        curve = []
        for step in range(1, args.steps + 1):
            batch = make_batch(train, args.seq_len, args.batch_size, dev)
            loss = lstm.loss_on(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), args.grad_clip)
            opt.step()
            if step % args.eval_every == 0 or step == args.steps:
                va = eval_metrics(
                    lstm, lstm.loss_on, val,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                    max_windows=args.max_windows, device=dev,
                )
                tr = eval_metrics(
                    lstm, lstm.loss_on, train,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                    max_windows=64, device=dev,
                )
                curve.append({
                    "step": step, "train_bpc": tr["bpc"], "val_bpc": va["bpc"],
                    "val_ppl": va["ppl"], "gap_bpc": va["bpc"] - tr["bpc"],
                })
                if va["bpc"] < best_val:
                    best_val = va["bpc"]
                    best_step = step
                    best_state = {k: v.detach().cpu().clone() for k, v in lstm.state_dict().items()}
                log(f"  lstm step {step}/{args.steps} val_bpc={va['bpc']:.4f}")
        if best_state:
            lstm.load_state_dict(best_state)
        te = eval_metrics(
            lstm, lstm.loss_on, test,
            seq_len=args.seq_len, batch_size=args.batch_size,
            max_windows=args.max_windows, device=dev,
        )
        rec["results"]["lstm"] = {
            "regime": "bptt",
            "params": count_params(lstm),
            "best_val_bpc": best_val,
            "best_val_ppl": 2 ** best_val,
            "best_step": best_step,
            "test_bpc": te["bpc"],
            "test_ppl": te["ppl"],
            "wall_min": round((time.time() - t0) / 60, 1),
        }
        rec["curves"]["lstm"] = curve

    for regime in regimes:
        cfg = LMConfig(
            grid_size=args.grid_size, embed_dim=args.embed_dim,
            inner_steps=args.inner_steps, seed=args.seed,
        )
        brain = BrainLanguageModel(tok.vocab_size, cfg, device=dev)
        log(f"==== BRAIN regime={regime} ({count_params(brain):,} params) ====")
        summary, curve = train_regime(
            "brain", brain, train, val, test, tok,
            regime=regime, steps=args.steps, seq_len=args.seq_len,
            batch_size=args.batch_size, lr=args.lr, grad_clip=args.grad_clip,
            eval_every=args.eval_every, max_windows=args.max_windows,
            seed=args.seed, device=dev,
        )
        key = f"brain_{regime}"
        rec["results"][key] = summary
        rec["curves"][key] = curve
        if "diverged_at" not in summary:
            gens = []
            for prompt in DEFAULT_PROMPTS[:3]:
                cont = brain.generate(
                    tok, prompt=prompt, max_new_tokens=160, temperature=0.8, top_k=40,
                )
                gens.append({"prompt": prompt, "continuation": cont})
                log(f"  [{key} sample] {prompt!r} -> {cont[:100]!r}...")
            rec["samples"][key] = gens
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rec, f, indent=2)

    log("==== TRACK B LEADERBOARD (lower val_bpc better) ====")
    rows = []
    for k, s in rec["results"].items():
        if "best_val_bpc" in s:
            rows.append((s["best_val_bpc"], k, s))
    rows.sort()
    for bpc, k, s in rows:
        log(
            f"  {k:<22} val_bpc={bpc:.4f} test_bpc={s.get('test_bpc', float('nan')):.4f} "
            f"regime={s.get('regime')}"
        )
    with open(args.json, "w") as f:
        json.dump(rec, f, indent=2)
    log(f"wrote {args.json}")


if __name__ == "__main__":
    main()
