#!/usr/bin/env python
"""
Hard-task intelligence probes — beyond next-char on easy English.

Track A/B showed LSTM > tiny GPT > brain on TinyStories/WikiText *bpc*.
Those tasks are short-context language modeling; they do not stress long
memory, binding, or composition. This harness trains and scores matched
models on **synthetic tasks** where those capacities are the objective:

  1. delayed_copy   — remember a string of length K over a blank delay D
  2. addition       — a+b=c with multi-digit numbers (char stream)
  3. associative    — study pairs, then query one key → value

Each task is generated online; train/val use different seeds. Metric is
**token accuracy on the answer span** (and CE), not open-ended fluency.

    python experiments/hard_tasks_eval.py \\
        --tasks delayed_copy,addition,associative \\
        --models lstm,cnn,gpt,brain,brain_wm --steps 8000 \\
        --json runs/hard_tasks_g12.json

Laptop/Mini scale. G=12 brain sets param target (~0.16–0.27M).
Standard suite always includes LSTM / CNN / tiny GPT / brain / brain_wm
(plus optional rnn). Do not drop baselines when adding brain variants.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
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


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------- tokenizer
# Fixed alphabet so every model sees the same vocab for synthetic tasks.
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz+-=,. ?|"
SPECIAL = {"pad": " ", "sep": "|", "blank": ".", "query": "?"}


def build_tokenizer() -> CharTokenizer:
    # CharTokenizer.from_text fits on a string containing every symbol once.
    return CharTokenizer.from_text(ALPHABET)


# ---------------------------------------------------------------- task samples
@dataclass
class Sample:
    text: str
    """Full string; model is trained with next-char CE on the whole string."""
    answer_start: int
    """Char index where the answer span begins (inclusive)."""
    answer_end: int
    """Char index where the answer span ends (exclusive)."""


def gen_delayed_copy(rng: random.Random, k: int = 8, delay: int = 16) -> Sample:
    """k content chars, delay blanks, then echo content. Answer = echoed span."""
    content = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(k))
    blank = SPECIAL["blank"] * delay
    # format: CONTENT...CONTENT
    text = content + blank + content
    ans_start = k + delay
    return Sample(text=text, answer_start=ans_start, answer_end=len(text))


def gen_addition(rng: random.Random, digits: int = 2) -> Sample:
    """'a+b=c' with a,b in [0, 10^digits). Answer = c digits (no leading zeros stripped)."""
    lo, hi = 0, 10 ** digits
    a = rng.randrange(lo, hi)
    b = rng.randrange(lo, hi)
    c = a + b
    # fixed-width operands and sum (sum may need digits+1)
    aw = f"{a:0{digits}d}"
    bw = f"{b:0{digits}d}"
    cw = f"{c:0{digits + 1}d}"
    prefix = f"{aw}+{bw}="
    text = prefix + cw
    return Sample(text=text, answer_start=len(prefix), answer_end=len(text))


def gen_associative(rng: random.Random, n_pairs: int = 3, key_len: int = 2, val_len: int = 2) -> Sample:
    """
    Study: k1=v1,k2=v2,... then query k_i?v_i
    Answer = value after '?'.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    pairs = []
    keys_used = set()
    for _ in range(n_pairs):
        while True:
            k = "".join(rng.choice(letters) for _ in range(key_len))
            if k not in keys_used:
                keys_used.add(k)
                break
        v = "".join(rng.choice(letters) for _ in range(val_len))
        pairs.append((k, v))
    study = ",".join(f"{k}={v}" for k, v in pairs)
    qi = rng.randrange(n_pairs)
    qk, qv = pairs[qi]
    prefix = f"{study}|{qk}?"
    text = prefix + qv
    return Sample(text=text, answer_start=len(prefix), answer_end=len(text))


TASKS: Dict[str, Callable[[random.Random], Sample]] = {
    "delayed_copy": lambda rng: gen_delayed_copy(rng, k=8, delay=16),
    "addition": lambda rng: gen_addition(rng, digits=2),
    "associative": lambda rng: gen_associative(rng, n_pairs=3),
}

# Harder variants for a second pass
TASKS_HARD: Dict[str, Callable[[random.Random], Sample]] = {
    "delayed_copy": lambda rng: gen_delayed_copy(rng, k=12, delay=32),
    "addition": lambda rng: gen_addition(rng, digits=3),
    "associative": lambda rng: gen_associative(rng, n_pairs=4, key_len=3, val_len=3),
}


def encode_sample(tok: CharTokenizer, sample: Sample, device) -> Tuple[torch.Tensor, slice]:
    ids = tok.encode(sample.text)
    # answer spans in char indices == token indices for char tokenizer
    ans = slice(sample.answer_start, sample.answer_end)
    return torch.tensor(ids, dtype=torch.long, device=device), ans


def batch_samples(
    tok: CharTokenizer,
    samples: Sequence[Sample],
    device,
) -> Tuple[torch.Tensor, List[slice]]:
    """Pad to max length in batch; return (B, T) and answer slices."""
    encoded = [tok.encode(s.text) for s in samples]
    T = max(len(e) for e in encoded)
    pad_id = tok.encode(SPECIAL["pad"])[0] if SPECIAL["pad"] in tok.stoi else 0
    # CharTokenizer uses dict stoi - check API
    if not hasattr(tok, "stoi"):
        # fallback: space might be missing — use 0
        pad_id = 0
    else:
        pad_id = tok.stoi.get(SPECIAL["pad"], 0)
    batch = torch.full((len(samples), T), pad_id, dtype=torch.long, device=device)
    spans = []
    for i, (e, s) in enumerate(zip(encoded, samples)):
        batch[i, : len(e)] = torch.tensor(e, dtype=torch.long, device=device)
        spans.append(slice(s.answer_start, s.answer_end))
    return batch, spans


# ------------------------------------------------------------------- models
def loss_on_ids(model, ids: torch.Tensor) -> torch.Tensor:
    """ids (B, T); next-char CE over all non-last positions."""
    if hasattr(model, "loss_on"):
        # BrainLanguageModel / CharLSTM expect (B, T) including target last
        return model.loss_on(ids)
    logits = model(ids[:, :-1])
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1),
    )


@torch.no_grad()
def answer_accuracy(
    model,
    tok: CharTokenizer,
    samples: Sequence[Sample],
    device,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Fraction of answer-span characters predicted correctly (teacher-forced)."""
    model.eval()
    correct = 0
    total = 0
    ce_sum = 0.0
    n_tok = 0
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        ids, spans = batch_samples(tok, chunk, device)
        # logits for predicting ids[:, 1:] from ids[:, :-1]
        if isinstance(model, BrainLanguageModel):
            logits, _ = model(ids[:, :-1])
            target = ids[:, 1:]
        elif isinstance(model, CharGPT):
            logits = model(ids[:, :-1])
            target = ids[:, 1:]
            if logits.size(1) != target.size(1):
                target = target[:, -logits.size(1) :]
        elif isinstance(model, CharCNN):
            logits = model(ids[:, :-1])
            target = ids[:, 1:]
        else:
            logits = model(ids[:, :-1])
            target = ids[:, 1:]
        # CE
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            reduction="sum",
        )
        ce_sum += float(ce.item())
        n_tok += target.numel()
        pred = logits.argmax(dim=-1)  # (B, T-1)
        for b, sp in enumerate(spans):
            # answer chars at positions [ans_start, ans_end) in original;
            # predictions for char at position j come from logits index j-1
            for pos in range(sp.start, sp.stop):
                if pos <= 0 or pos >= ids.size(1):
                    continue
                # pred index for target ids[:, pos] is pos-1
                pi = pos - 1
                if pi < 0 or pi >= pred.size(1):
                    continue
                total += 1
                if int(pred[b, pi].item()) == int(ids[b, pos].item()):
                    correct += 1
    model.train()
    return {
        "answer_acc": correct / max(total, 1),
        "answer_chars": total,
        "ce": ce_sum / max(n_tok, 1),
        "bpc": (ce_sum / max(n_tok, 1)) / math.log(2),
    }


def train_task(
    name: str,
    model,
    tok: CharTokenizer,
    task_fn: Callable[[random.Random], Sample],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    eval_every: int,
    n_eval: int,
    seed: int,
    device,
) -> Tuple[Dict, List[Dict]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 999)
    val_samples = [task_fn(eval_rng) for _ in range(n_eval)]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    curve = []
    best_acc = -1.0
    best_step = 0
    best_state = None

    for step in range(1, steps + 1):
        samples = [task_fn(rng) for _ in range(batch_size)]
        ids, _ = batch_samples(tok, samples, device)
        loss = loss_on_ids(model, ids)
        if not math.isfinite(float(loss.item())):
            log(f"  {name} DIVERGED at step {step}")
            return {"diverged_at": step, "params": count_params(model)}, curve
        opt.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step % eval_every == 0 or step == steps:
            metrics = answer_accuracy(model, tok, val_samples, device, batch_size=min(32, n_eval))
            row = {"step": step, **metrics, "wall_s": round(time.time() - t0, 1)}
            curve.append(row)
            if metrics["answer_acc"] > best_acc:
                best_acc = metrics["answer_acc"]
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(
                f"  {name} step {step}/{steps}  "
                f"ans_acc={metrics['answer_acc']:.3f}  bpc={metrics['bpc']:.3f}  "
                f"ce={metrics['ce']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    # fresh test set
    test_rng = random.Random(seed + 12345)
    test_samples = [task_fn(test_rng) for _ in range(n_eval)]
    test_m = answer_accuracy(model, tok, test_samples, device)
    summary = {
        "params": count_params(model),
        "best_answer_acc": best_acc,
        "best_step": best_step,
        "test_answer_acc": test_m["answer_acc"],
        "test_bpc": test_m["bpc"],
        "test_ce": test_m["ce"],
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    log(
        f"  {name} DONE  best_acc={best_acc:.3f} @ {best_step}  "
        f"test_acc={test_m['answer_acc']:.3f}  ({summary['wall_min']} min)"
    )
    return summary, curve


def make_model(
    kind: str,
    vocab: int,
    target_params: int,
    device,
    seed: int,
    seq_hint: int = 128,
    grid_size: int = 12,
):
    kind = kind.lower().strip()
    if kind == "brain":
        cfg = LMConfig(grid_size=grid_size, embed_dim=64, inner_steps=3, seed=seed)
        return BrainLanguageModel(vocab, cfg, device=device)
    if kind == "brain_wm":
        # Working-memory attention + zone gain routing (bio-flavoured attention).
        cfg = LMConfig(
            grid_size=grid_size, embed_dim=64, inner_steps=3, seed=seed,
            use_wm_attn=True, wm_slots=48,
            brain_overrides={"use_zone_attn": True, "zone_attn_dim": 16},
        )
        return BrainLanguageModel(vocab, cfg, device=device)
    if kind == "lstm":
        h = match_lstm_hidden(vocab, 64, target_params)
        return CharLSTM(vocab, 64, h).to(device)
    if kind == "rnn":
        h = match_rnn_hidden(vocab, 64, target_params)
        return CharRNN(vocab, 64, h).to(device)
    if kind == "cnn":
        emb, ch, n_layers = match_cnn_config(vocab, target_params)
        return CharCNN(vocab, embed_dim=emb, channels=ch, n_layers=n_layers).to(device)
    if kind == "gpt":
        d, L, H = match_gpt_config(vocab, target_params, max_seq=max(seq_hint, 256))
        return CharGPT(vocab, d_model=d, n_layer=L, n_head=H, max_seq=max(seq_hint, 256)).to(device)
    raise ValueError(f"unknown model {kind!r}; choose from {STANDARD_MODELS}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="delayed_copy,addition,associative")
    p.add_argument("--hard", action="store_true", help="use harder task variants")
    p.add_argument(
        "--models",
        default="lstm,cnn,gpt,brain,brain_wm",
        help="standard suite: lstm,cnn,gpt,brain,brain_wm (+ rnn via STANDARD_MODELS)",
    )
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--n-eval", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps")
    p.add_argument("--json", default="runs/hard_tasks_g12.json")
    p.add_argument(
        "--match-to",
        default="brain",
        help="which model sets the param budget: brain | brain_wm | or integer target",
    )
    p.add_argument(
        "--grid-size",
        type=int,
        default=12,
        help="brain / brain_wm cube side (default 12)",
    )
    args = p.parse_args()

    dev = get_device(args.device)
    tok = build_tokenizer()
    vocab = tok.vocab_size
    task_bank = TASKS_HARD if args.hard else TASKS
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Param target: match baselines to plain brain or heavier brain_wm (D6 fairness).
    match_key = str(args.match_to).strip().lower()
    if match_key.isdigit():
        target = int(match_key)
        match_src = f"explicit:{target}"
    elif match_key == "brain_wm":
        probe = BrainLanguageModel(
            vocab,
            LMConfig(
                grid_size=args.grid_size, embed_dim=64, inner_steps=3, seed=args.seed,
                use_wm_attn=True, wm_slots=48,
                brain_overrides={"use_zone_attn": True, "zone_attn_dim": 16},
            ),
            device=dev,
        )
        target = count_params(probe)
        match_src = "brain_wm"
        del probe
    else:
        probe = BrainLanguageModel(
            vocab,
            LMConfig(grid_size=args.grid_size, embed_dim=64, inner_steps=3, seed=args.seed),
            device=dev,
        )
        target = count_params(probe)
        match_src = "brain"
        del probe
    log(
        f"hard-tasks vocab={vocab} param_target={target:,} match_to={match_src} "
        f"G={args.grid_size} device={dev} hard={args.hard}"
    )

    rec = {
        "protocol": {
            "tasks": tasks,
            "hard": args.hard,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "grid_size": args.grid_size,
            "match_to": match_src,
            "metric": "teacher-forced answer-span character accuracy",
            "note": "Synthetic probes for memory/composition/binding — not open LM fluency",
        },
        "param_target": target,
        "results": {},
        "curves": {},
    }

    for task in tasks:
        if task not in task_bank:
            raise SystemExit(f"unknown task {task}; choose from {list(task_bank)}")
        task_fn = task_bank[task]
        # show one example
        ex = task_fn(random.Random(0))
        log(f"==== TASK {task} example: {ex.text!r} answer={ex.text[ex.answer_start:ex.answer_end]!r} ====")
        rec["results"][task] = {}
        rec["curves"][task] = {}
        for kind in models:
            model = make_model(
                kind, vocab, target, dev, args.seed, grid_size=args.grid_size
            )
            log(f"-- {kind} params={count_params(model):,}")
            summary, curve = train_task(
                f"{task}/{kind}",
                model,
                tok,
                task_fn,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                grad_clip=args.grad_clip,
                eval_every=args.eval_every,
                n_eval=args.n_eval,
                seed=args.seed + hash(task + kind) % 10000,
                device=dev,
            )
            rec["results"][task][kind] = summary
            rec["curves"][task][kind] = curve
            os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
            with open(args.json, "w") as f:
                json.dump(rec, f, indent=2)
            del model
            if dev.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    log("==== HARD-TASK LEADERBOARD (higher answer_acc better) ====")
    for task in tasks:
        log(f"  [{task}]")
        rows = []
        for kind, s in rec["results"][task].items():
            if "test_answer_acc" in s:
                rows.append((s["test_answer_acc"], kind, s))
        rows.sort(reverse=True)
        for acc, kind, s in rows:
            log(
                f"    {kind:<8} test_acc={acc:.3f}  best_acc={s.get('best_answer_acc', 0):.3f}  "
                f"@ {s.get('best_step')}  test_bpc={s.get('test_bpc', float('nan')):.3f}"
            )
    with open(args.json, "w") as f:
        json.dump(rec, f, indent=2)
    log(f"wrote {args.json}")


if __name__ == "__main__":
    main()
