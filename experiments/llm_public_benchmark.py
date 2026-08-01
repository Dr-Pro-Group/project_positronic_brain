#!/usr/bin/env python
"""
Primary intelligence benchmark: public LLM data + standard LM metrics.

First principles (see docs/INTELLIGENCE_PROGRAM.md)
-------------------------------------------------
* Train on public natural-language corpora (FineWeb-Edu / TinyStories / …)
  prepared as disk memmap tokens — same family of data used for open LLMs.
* Score held-out **bpc / bits-per-byte / perplexity** and fixed-prompt samples.
* Full matched suite: LSTM, RNN, CNN (local floor), tiny GPT (LLM prior),
  brain, brain_wm. CNN is a *negative control* — if it wins open LM bpc, the
  protocol is broken, not "CNNs are intelligent."

This is the headline track. Synthetic hard tasks are secondary diagnostics only.

    # prepare once
    python experiments/scale_train.py prepare \\
        --dataset fineweb-edu --max-docs 15000 --max-chars 30000000 \\
        --vocab-size 4096 --work-dir data/llm_fineweb_edu

    # full suite benchmark
    python experiments/llm_public_benchmark.py \\
        --work-dir data/llm_fineweb_edu \\
        --models lstm,rnn,cnn,gpt,brain,brain_wm \\
        --steps 8000 --grid-size 12 --device mps \\
        --json runs/llm_bench_fineweb_g12.json
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
import torch.nn.functional as F

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
from experiments.public_lm_eval import CharGPT, generate_baseline, match_gpt_config
from positronic_brain.checkpoints import (
    default_run_dir,
    save_model_bundle,
    save_tokenizer,
    write_run_meta,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Fixed English prompts for sample quality (not scored, qualitative).
SAMPLE_PROMPTS = [
    "Once upon a time",
    "The scientific method",
    "In the beginning",
    "Artificial intelligence",
]


@torch.no_grad()
def eval_store(
    model,
    loss_fn,
    store: MemmapTokenStore,
    *,
    seq_len: int,
    batch_size: int,
    max_windows: int,
    device,
    bytes_per_token: Optional[float] = None,
) -> Dict[str, float]:
    """Held-out CE → bits/token, ppl, and bits-per-byte.

    Note on the ``bpc`` key: it is cross-entropy in bits per **token**, not per
    character.  Under a BPE tokenizer a token averages >2 bytes, so ``bpc`` is
    comparable *across models on the same corpus* (identical tokenizer) but NOT
    across corpora, and NOT against the character-level bpc reported by
    ``train_language.py``.  The key name is retained for compatibility with
    already-published run JSON.  **Use ``bpb`` for anything comparative** --- it
    is the tokenizer-independent bits-per-byte figure standard in the
    compression-based LM literature.
    """
    model.eval()
    tot = 0.0
    n = 0
    for batch in store.eval_batches(seq_len, batch_size, max_windows, device):
        loss = loss_fn(batch)
        nt = batch[:, 1:].numel()
        tot += float(loss.item()) * nt
        n += nt
    model.train()
    if n == 0:
        return {"ce": float("nan"), "bpc": float("nan"), "ppl": float("nan"), "bpb": float("nan"), "tokens": 0}
    ce = tot / n
    out = {
        "ce": ce,
        "bpc": ce / math.log(2),
        "ppl": math.exp(min(ce, 20.0)),
        "tokens": n,
    }
    # bits per byte if we know mean bytes/token from the tokenizer sample
    if bytes_per_token and bytes_per_token > 0:
        out["bpb"] = out["bpc"] / bytes_per_token
    else:
        out["bpb"] = float("nan")
    return out


def estimate_bytes_per_token(tok: SubwordTokenizer, store: MemmapTokenStore, n: int = 5000) -> float:
    """Rough UTF-8 bytes per token on a slice of the train memmap (for bpb)."""
    import numpy as np
    mm = store._mm
    lo, hi = store.lo, min(store.hi, store.lo + n)
    ids = mm[lo:hi].astype(np.int64).tolist()
    text = tok.decode(ids)
    if not ids:
        return 1.0
    return max(len(text.encode("utf-8")), 1) / len(ids)


def make_suite(
    vocab: int,
    target: int,
    device,
    *,
    grid_size: int,
    embed_dim: int,
    inner_steps: int,
    seed: int,
    seq_len: int,
    want: List[str],
) -> Dict[str, Tuple[nn.Module, callable]]:
    catalog = {}
    if "brain" in want or "brain_wm" in want or True:
        pass
    if "brain" in want:
        cfg = LMConfig(
            grid_size=grid_size, embed_dim=embed_dim, inner_steps=inner_steps, seed=seed,
            grad_checkpoint=True,
        )
        m = BrainLanguageModel(vocab, cfg, device=device)
        catalog["brain"] = (m, m.loss_on)
    if "brain_wm" in want:
        cfg = LMConfig(
            grid_size=grid_size, embed_dim=embed_dim, inner_steps=inner_steps, seed=seed,
            use_wm_attn=True, wm_slots=48,
            brain_overrides={"use_zone_attn": True, "zone_attn_dim": 16},
            grad_checkpoint=False,  # checkpoint disabled with WM state
        )
        m = BrainLanguageModel(vocab, cfg, device=device)
        catalog["brain_wm"] = (m, m.loss_on)
    if "lstm" in want:
        h = match_lstm_hidden(vocab, embed_dim, target)
        m = CharLSTM(vocab, embed_dim, h).to(device)
        catalog["lstm"] = (m, m.loss_on)
    if "rnn" in want:
        h = match_rnn_hidden(vocab, embed_dim, target)
        m = CharRNN(vocab, embed_dim, h).to(device)
        catalog["rnn"] = (m, m.loss_on)
    if "cnn" in want:
        emb, ch, nL = match_cnn_config(vocab, target)
        m = CharCNN(vocab, embed_dim=emb, channels=ch, n_layers=nL).to(device)
        catalog["cnn"] = (m, m.loss_on)
    if "gpt" in want:
        d, L, H = match_gpt_config(vocab, target, max_seq=max(seq_len, 256))
        m = CharGPT(vocab, d_model=d, n_layer=L, n_head=H, max_seq=max(seq_len, 256)).to(device)
        catalog["gpt"] = (m, m.loss_on)
    return catalog


def train_one(
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
    model.train()
    t0 = time.time()
    curve: List[Dict] = []
    best_bpc = float("inf")
    best_step = 0
    best_state = None
    run_ce = 0.0
    n_run = 0

    for step in range(1, steps + 1):
        batch = train.sample_batch(seq_len, batch_size, device)
        loss = loss_fn(batch)
        if not math.isfinite(float(loss.item())):
            log(f"  {name} DIVERGED at step {step}")
            # Keep best checkpoint metrics so a late NaN does not erase the run.
            if best_state is not None:
                model.load_state_dict(best_state)
            summary = {
                "diverged_at": step,
                "params": count_params(model),
                "best_val_bpc": None if best_bpc == float("inf") else best_bpc,
                "best_step": best_step if best_bpc != float("inf") else None,
                "wall_min": round((time.time() - t0) / 60, 2),
            }
            return summary, curve
        opt.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        run_ce += float(loss.item())
        n_run += 1

        if step % eval_every == 0 or step == steps:
            metrics = eval_store(
                model, loss_fn, val,
                seq_len=seq_len, batch_size=batch_size,
                max_windows=max_windows, device=device,
                bytes_per_token=bytes_per_token,
            )
            avg = run_ce / max(n_run, 1)
            run_ce, n_run = 0.0, 0
            row = {"step": step, "train_ce": avg, **metrics, "wall_s": round(time.time() - t0, 1)}
            curve.append(row)
            log(
                f"  {name} step {step}/{steps} train_ce={avg:.4f} "
                f"val_bpc={metrics['bpc']:.4f} val_ppl={metrics['ppl']:.2f} "
                f"val_bpb={metrics.get('bpb', float('nan')):.4f}"
            )
            if metrics["bpc"] < best_bpc:
                best_bpc = metrics["bpc"]
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "params": count_params(model),
        "best_val_bpc": best_bpc,
        "best_step": best_step,
        "wall_min": round((time.time() - t0) / 60, 2),
    }
    return summary, curve


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True, help="disk store from scale_train.py prepare")
    p.add_argument(
        "--models",
        default="lstm,rnn,cnn,gpt,brain,brain_wm",
        help="full intelligence suite (CNN = local floor, not IQ winner)",
    )
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--max-windows", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps")
    p.add_argument("--json", default="runs/llm_public_benchmark.json")
    p.add_argument("--gen-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument(
        "--checkpoint-dir",
        default="",
        help="if set, save best weights per model under this dir (for resume/finetune/DPO)",
    )
    p.add_argument(
        "--run-name",
        default="",
        help="checkpoint subfolder name (default: derived from --json basename)",
    )
    args = p.parse_args()

    device = get_device(args.device)
    done = os.path.join(args.work_dir, "prepare_done.json")
    if not os.path.isfile(done):
        raise SystemExit(
            f"missing {done} — run: python experiments/scale_train.py prepare "
            f"--dataset fineweb-edu|tinystories --work-dir {args.work_dir}"
        )
    with open(done) as f:
        prep = json.load(f)

    tok = SubwordTokenizer.load(prep["tokenizer_path"])
    train = MemmapTokenStore(prep["mmap_path"], split="train")
    val = MemmapTokenStore(prep["mmap_path"], split="val")
    test = MemmapTokenStore(prep["mmap_path"], split="test")
    bpt = estimate_bytes_per_token(tok, train)
    log(
        f"LLM public bench work_dir={args.work_dir} dataset={prep.get('dataset')} "
        f"vocab={tok.vocab_size} train_tok={train.n_tokens:,} "
        f"bytes/tok≈{bpt:.3f} device={device}"
    )

    # Param target = plain brain at this G (fair match; brain_wm may be heavier).
    probe = BrainLanguageModel(
        tok.vocab_size,
        LMConfig(grid_size=args.grid_size, embed_dim=args.embed_dim, inner_steps=args.inner_steps, seed=args.seed),
        device=device,
    )
    target = count_params(probe)
    del probe
    log(f"param_target (plain brain G={args.grid_size}) = {target:,}")

    want = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in want if m not in STANDARD_MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {STANDARD_MODELS}")

    catalog = make_suite(
        tok.vocab_size, target, device,
        grid_size=args.grid_size, embed_dim=args.embed_dim,
        inner_steps=args.inner_steps, seed=args.seed, seq_len=args.seq_len,
        want=want,
    )
    for name in want:
        m, _ = catalog[name]
        log(f"  {name}: params={count_params(m):,}")

    rec: Dict = {
        "protocol": {
            "description": "Primary intelligence track: public LLM data + bpc/ppl/bpb + samples",
            "work_dir": args.work_dir,
            "dataset": prep.get("dataset"),
            "tokenization": "subword_bpe",
            "vocab_size": tok.vocab_size,
            "param_target": target,
            "match": "plain brain G params (brain_wm may exceed)",
            "metrics": ["val_bpc", "test_bpc", "ppl", "bpb", "samples"],
            "note": (
                "CNN is local floor / negative control. Tiny GPT = from-scratch LLM prior, "
                "NOT pretrained GPT-4. Synthetic hard tasks are NOT the headline metric."
            ),
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "grid_size": args.grid_size,
            "seed": args.seed,
            "standard_models": list(STANDARD_MODELS),
        },
        "corpus": {
            "train_tokens": train.n_tokens,
            "val_tokens": val.n_tokens,
            "test_tokens": test.n_tokens,
            "bytes_per_token_est": bpt,
        },
        "params": {n: count_params(catalog[n][0]) for n in want},
        "results": {},
        "curves": {},
        "samples": {},
        "checkpoints": {},
    }

    # Checkpoint root: explicit --checkpoint-dir, else checkpoints/<run_name>
    run_name = args.run_name or os.path.splitext(os.path.basename(args.json))[0]
    ckpt_root = args.checkpoint_dir or default_run_dir(".", run_name)
    os.makedirs(ckpt_root, exist_ok=True)
    save_tokenizer(tok, os.path.join(ckpt_root, "tokenizer.json"))
    rec["protocol"]["checkpoint_dir"] = ckpt_root
    log(f"checkpoints → {ckpt_root}")

    # Train order: cheap baselines first, brain last (time).
    order = [m for m in ("lstm", "rnn", "cnn", "gpt", "brain", "brain_wm") if m in want]
    for name in order:
        model, loss_fn = catalog[name]
        log(f"==== TRAIN {name} (intelligence suite) ====")
        summary, curve = train_one(
            name, model, loss_fn, train, val,
            steps=args.steps, seq_len=args.seq_len, batch_size=args.batch_size,
            lr=args.lr, grad_clip=args.grad_clip, eval_every=args.eval_every,
            max_windows=args.max_windows, device=device, bytes_per_token=bpt,
        )
        rec["results"][name] = summary
        rec["curves"][name] = curve
        # Test/samples from best checkpoint when available (incl. late divergence).
        can_test = summary.get("best_val_bpc") is not None and math.isfinite(
            float(summary["best_val_bpc"])
        )
        if can_test:
            if "diverged_at" in summary:
                log(
                    f"  {name} using best ckpt @ step {summary.get('best_step')} "
                    f"(diverged_at={summary['diverged_at']}) for test/samples"
                )
            te = eval_store(
                model, loss_fn, test,
                seq_len=args.seq_len, batch_size=args.batch_size,
                max_windows=args.max_windows, device=device,
                bytes_per_token=bpt,
            )
            summary["test_bpc"] = te["bpc"]
            summary["test_ppl"] = te["ppl"]
            summary["test_bpb"] = te.get("bpb")
            rec["results"][name] = summary
            log(f"  {name} test_bpc={te['bpc']:.4f} test_ppl={te['ppl']:.2f} test_bpb={te.get('bpb', float('nan')):.4f}")

            # samples
            gens = []
            for prompt in SAMPLE_PROMPTS:
                try:
                    if name in ("brain", "brain_wm"):
                        cont = model.generate(
                            tok, prompt=prompt, max_new_tokens=args.gen_tokens,
                            temperature=args.temperature, top_k=40,
                        )
                    else:
                        cont = generate_baseline(
                            model, tok, prompt, max_new_tokens=args.gen_tokens,
                            temperature=args.temperature, top_k=40, device=device,
                        )
                except Exception as exc:
                    cont = f"<gen_error: {exc}>"
                gens.append({"prompt": prompt, "continuation": cont})
                log(f"  [{name}] {prompt!r} → {cont[:100]!r}...")
            rec["samples"][name] = gens

        # Persist weights for resume / fine-tune / DPO (even if diverged mid-run).
        try:
            ckpt_path = os.path.join(ckpt_root, f"{name}.pt")
            save_model_bundle(
                name, model, ckpt_path,
                tokenizer=tok,
                metrics=summary,
                protocol=rec["protocol"],
            )
            rec["checkpoints"][name] = ckpt_path
            log(f"  saved checkpoint {ckpt_path}")
        except Exception as exc:
            log(f"  WARNING: failed to save checkpoint for {name}: {exc}")

        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rec, f, indent=2)
        # free
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
        "checkpoints": rec["checkpoints"],
    })

    # Leaderboard
    log(
        "==== PUBLIC LLM INTELLIGENCE LEADERBOARD (lower better; "
        "'bpc' is bits/TOKEN under BPE — compare across corpora with bpb) ===="
    )
    rows = []
    for name, s in rec["results"].items():
        if "best_val_bpc" in s:
            rows.append((s["best_val_bpc"], name, s))
    rows.sort()
    for bpc, name, s in rows:
        log(
            f"  {name:<10} val_bpc={bpc:.4f}  test_bpc={s.get('test_bpc', float('nan')):.4f}  "
            f"test_ppl={s.get('test_ppl', float('nan')):.2f}  params={s['params']:,}"
        )
    # CNN control check
    if "cnn" in rec["results"] and "lstm" in rec["results"]:
        cb = rec["results"]["cnn"].get("best_val_bpc", float("inf"))
        lb = rec["results"]["lstm"].get("best_val_bpc", float("inf"))
        if cb < lb:
            log(
                "WARNING: CNN beat LSTM on open LM bpc — check data/protocol; "
                "do NOT interpret as intelligence ranking."
            )
        else:
            log("CNN control OK: local floor worse than LSTM on open LM bpc.")

    with open(args.json, "w") as f:
        json.dump(rec, f, indent=2)
    log(f"wrote {args.json}")


if __name__ == "__main__":
    main()
