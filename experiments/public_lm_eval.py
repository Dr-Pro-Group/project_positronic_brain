#!/usr/bin/env python
"""
Public LM qualification: TinyStories / WikiText + matched baselines + samples.

Protocol
--------
* Stream a public plain-text corpus (``--hf tinystories`` or ``wikitext``).
* Content-disjoint train/val/test (no built-in SODA unless ``--builtin``).
* Train brain + parameter-matched **LSTM, dense RNN, causal CNN, tiny GPT**
  under the same budget — from scratch, not pretrained GPT-4.
* Report held-out bits-per-char (bpc) and perplexity (standard char-LM metrics).
* Dump fixed-prompt generations from best-val checkpoints.

    python experiments/public_lm_eval.py \\
        --hf tinystories --hf-limit 100000 --steps 20000 \\
        --grid-size 12 --models lstm,rnn,cnn,gpt,brain \\
        --json runs/public_tinystories_g12.json

This is the honest "qualify text generation" track — public data, public metrics —
not a frontier LLM leaderboard. ``gpt`` here means a *tiny* GPT-style Transformer
with ~matched parameter count trained from scratch on the same data.
Standard suite: ``lstm,rnn,cnn,gpt,brain`` (+ optional ``brain_wm``).
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

from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.utils import get_device

from experiments.matched_experiment import (  # type: ignore
    CharCNN,
    CharLSTM,
    CharRNN,
    STANDARD_MODELS,
    count_params,
    match_cnn_config,
    match_lstm_hidden,
    match_rnn_hidden,
)


# --------------------------------------------------------------------------- tiny GPT
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, T, Dh
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(y)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        return x


class CharGPT(nn.Module):
    """Minimal GPT-2-style decoder-only Transformer for character LMs.

    From-scratch, laptop-scale: not pretrained weights, not ChatGPT.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        max_seq: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying (common GPT practice)
        self.head.weight = self.tok_emb.weight
        self.config = {
            "d_model": d_model,
            "n_layer": n_layer,
            "n_head": n_head,
            "max_seq": max_seq,
            "dropout": dropout,
        }

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        if T > self.max_seq:
            tokens = tokens[:, -self.max_seq :]
            T = tokens.shape[1]
        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def loss_on(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(next(self.parameters()).device)
        logits = self.forward(tokens[:, :-1])
        target = tokens[:, 1:]
        # align if max_seq truncated
        if logits.size(1) != target.size(1):
            target = target[:, -logits.size(1) :]
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size), target.reshape(-1)
        )


def match_gpt_config(
    vocab: int, target: int, max_seq: int = 256
) -> Tuple[int, int, int]:
    """Pick (d_model, n_layer, n_head) closest to ``target`` params (from below/near)."""
    best = (128, 4, 4)
    best_gap = float("inf")
    # small grid of GPT-mini shapes
    for n_layer in (2, 3, 4, 5, 6):
        for d_model in (64, 96, 128, 160, 192, 256):
            for n_head in (2, 4, 8):
                if d_model % n_head != 0:
                    continue
                probe = CharGPT(vocab, d_model=d_model, n_layer=n_layer,
                                n_head=n_head, max_seq=max_seq)
                n = count_params(probe)
                gap = abs(n - target)
                # prefer not massively over target
                if n > target * 1.15:
                    continue
                if gap < best_gap:
                    best_gap = gap
                    best = (d_model, n_layer, n_head)
    return best

# Fixed prompt bank for qualitative samples (story-oriented + one generic).
DEFAULT_PROMPTS = [
    "Once upon a time",
    "The little girl",
    "One day, a cat",
    "Tom and Lily",
    "In the forest",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_batch(data: torch.Tensor, seq_len: int, batch_size: int, device) -> torch.Tensor:
    n = data.numel() - seq_len - 1
    ix = torch.randint(0, max(n, 1), (batch_size,))
    return torch.stack([data[i : i + seq_len + 1] for i in ix]).to(device)


@torch.no_grad()
def eval_metrics(
    model: nn.Module,
    loss_on,
    data: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    max_windows: int,
    device,
) -> Dict[str, float]:
    """Held-out CE / bpc / ppl over fixed non-overlapping windows."""
    model.eval()
    if data.numel() <= seq_len + 1:
        return {"ce": float("nan"), "bpc": float("nan"), "ppl": float("nan"), "windows": 0}
    starts = list(range(0, data.numel() - seq_len - 1, seq_len))[:max_windows]
    tot = 0.0
    n = 0
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        batch = torch.stack([data[s : s + seq_len + 1] for s in chunk]).to(device)
        loss = loss_on(batch)
        # loss_on returns mean over tokens; reweight by token count
        ntok = batch[:, 1:].numel()
        tot += float(loss.item()) * ntok
        n += ntok
    ce = tot / max(n, 1)
    model.train()
    return {
        "ce": ce,
        "bpc": ce / math.log(2),
        "ppl": math.exp(min(ce, 20.0)),
        "windows": len(starts),
    }


@torch.no_grad()
def generate_baseline(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device,
) -> str:
    """Autoregressive sample for CharLSTM / CharRNN / CharCNN / CharGPT."""
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]

    if isinstance(model, (CharGPT, CharCNN)):
        # Full re-encode each step (Transformer / causal conv both support this).
        out_ids: List[int] = []
        ctx = list(ids)
        max_ctx = getattr(model, "max_seq", 4096)
        for _ in range(max_new_tokens):
            x = torch.tensor([ctx[-max_ctx:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1] / max(temperature, 1e-5)
            if 0 < top_k < logits.numel():
                vals, idx = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter_(0, idx, F.softmax(vals, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            out_ids.append(nxt)
            ctx.append(nxt)
        return tokenizer.decode(out_ids)

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    if isinstance(model, CharLSTM):
        e = model.embed(x)
        out, (h, c) = model.lstm(e)
        logits = model.head(out[:, -1])
    else:
        e = model.embed(x)
        out, h = model.rnn(e)
        logits = model.head(out[:, -1])
        c = None

    out_ids = []
    for _ in range(max_new_tokens):
        logits = logits[0] / max(temperature, 1e-5)
        if 0 < top_k < logits.numel():
            vals, idx = torch.topk(logits, top_k)
            probs = torch.zeros_like(logits).scatter_(0, idx, F.softmax(vals, dim=-1))
        else:
            probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        out_ids.append(nxt)
        tok = torch.tensor([[nxt]], device=device)
        if isinstance(model, CharLSTM):
            e = model.embed(tok)
            out, (h, c) = model.lstm(e, (h, c))
            logits = model.head(out[:, -1])
        else:
            e = model.embed(tok)
            out, h = model.rnn(e, h)
            logits = model.head(out[:, -1])
    return tokenizer.decode(out_ids)


def train_one(
    name: str,
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
    eval_every: int,
    max_windows: int,
    seed: int,
    device,
) -> Tuple[Dict, List[Dict]]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    curve: List[Dict] = []
    best_val_bpc = float("inf")
    best_state = None
    best_step = 0
    running = 0.0
    n_run = 0

    for step in range(1, steps + 1):
        batch = make_batch(train_data, seq_len, batch_size, device)
        loss = loss_on(batch)
        if not math.isfinite(float(loss.item())):
            log(f"  {name} DIVERGED at step {step}")
            return {
                "diverged_at": step,
                "params": count_params(model),
                "wall_min": round((time.time() - t0) / 60, 1),
            }, curve
        opt.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        running += float(loss.item())
        n_run += 1

        if step % eval_every == 0 or step == steps:
            tr = eval_metrics(
                model, loss_on, train_data,
                seq_len=seq_len, batch_size=batch_size,
                max_windows=min(max_windows, 64), device=device,
            )
            va = eval_metrics(
                model, loss_on, val_data,
                seq_len=seq_len, batch_size=batch_size,
                max_windows=max_windows, device=device,
            )
            row = {
                "step": step,
                "train_bpc": tr["bpc"],
                "val_bpc": va["bpc"],
                "train_ppl": tr["ppl"],
                "val_ppl": va["ppl"],
                "gap_bpc": va["bpc"] - tr["bpc"],
                "recent_loss": running / max(n_run, 1),
                "wall_s": round(time.time() - t0, 1),
            }
            curve.append(row)
            running = 0.0
            n_run = 0
            if va["bpc"] < best_val_bpc:
                best_val_bpc = va["bpc"]
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(
                f"  {name} step {step}/{steps}  "
                f"train_bpc={tr['bpc']:.4f}  val_bpc={va['bpc']:.4f}  "
                f"val_ppl={va['ppl']:.2f}  gap={row['gap_bpc']:+.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_tr = eval_metrics(
        model, loss_on, train_data,
        seq_len=seq_len, batch_size=batch_size, max_windows=min(max_windows, 64), device=device,
    )
    final_va = eval_metrics(
        model, loss_on, val_data,
        seq_len=seq_len, batch_size=batch_size, max_windows=max_windows, device=device,
    )
    summary = {
        "params": count_params(model),
        "best_val_bpc": best_val_bpc,
        "best_val_ppl": 2 ** best_val_bpc if math.isfinite(best_val_bpc) else None,
        "best_step": best_step,
        "final_train_bpc": final_tr["bpc"],
        "final_val_bpc": final_va["bpc"],
        "final_val_ppl": final_va["ppl"],
        "final_gap_bpc": final_va["bpc"] - final_tr["bpc"],
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    log(
        f"  {name} DONE  best_val_bpc={best_val_bpc:.4f} (ppl={summary['best_val_ppl']:.2f}) "
        f"@ {best_step}  ({summary['wall_min']} min)"
    )
    return summary, curve


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf", default="tinystories", choices=["tinystories", "wikitext"],
                   help="public plain-text corpus preset")
    p.add_argument("--hf-limit", type=int, default=100_000,
                   help="max streamed rows (TinyStories stories / WikiText lines)")
    p.add_argument("--builtin", action="store_true",
                   help="also append built-in dialogues (default: pure public text)")
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--max-windows", type=int, default=256)
    p.add_argument(
        "--models",
        default="lstm,rnn,cnn,gpt,brain",
        help="standard suite: lstm,rnn,cnn,gpt,brain (+ brain_wm). gpt = tiny from-scratch Transformer",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps")
    p.add_argument("--gen-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--json", default="runs/public_lm_eval.json")
    p.add_argument("--samples-json", default="",
                   help="optional path for generation samples (default: beside --json)")
    args = p.parse_args()

    dev = get_device(args.device)
    # Pure public text by default (no SODA/builtin leak into the "public LM" claim).
    tr_t, va_t, te_t = load_corpus_splits(
        hf=args.hf,
        hf_limit=args.hf_limit,
        builtin=args.builtin,
        repeats=1,
        seed=args.seed,
        val_frac=0.1,
        test_frac=0.1,
    )
    if not tr_t.strip():
        raise SystemExit(f"empty train corpus from --hf {args.hf}; check network / datasets install")
    if len(tr_t) < 50_000 and not args.builtin:
        raise SystemExit(
            f"train corpus only {len(tr_t):,} chars from --hf {args.hf} — too small "
            f"for a public LM claim (likely failed HF load + seed fallback). Aborting."
        )

    tok = CharTokenizer.from_text(tr_t)
    train = torch.tensor(tok.encode(tr_t), dtype=torch.long)
    val = torch.tensor(tok.encode(va_t), dtype=torch.long)
    test = torch.tensor(tok.encode(te_t), dtype=torch.long)
    log(
        f"corpus {args.hf} limit={args.hf_limit}  "
        f"train={train.numel():,} val={val.numel():,} test={test.numel():,}  "
        f"vocab={tok.vocab_size}  device={dev}  builtin={args.builtin}"
    )

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
    # Optional bio-attention brain variant (same G; extra WM / zone params).
    cfg_wm = LMConfig(
        grid_size=args.grid_size,
        embed_dim=args.embed_dim,
        inner_steps=args.inner_steps,
        seed=args.seed,
        use_wm_attn=True,
        wm_slots=48,
        brain_overrides={"use_zone_attn": True, "zone_attn_dim": 16},
    )
    brain_wm = BrainLanguageModel(tok.vocab_size, cfg_wm, device=dev)

    catalog = {
        "brain": (brain, brain.loss_on),
        "brain_wm": (brain_wm, brain_wm.loss_on),
        "lstm": (lstm, lstm.loss_on),
        "rnn": (rnn, rnn.loss_on),
        "cnn": (cnn, cnn.loss_on),
        "gpt": (gpt, gpt.loss_on),
    }
    want = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    for name in want:
        if name not in catalog:
            raise SystemExit(
                f"unknown model {name!r}; choose from {list(catalog)} "
                f"(STANDARD_MODELS={STANDARD_MODELS})"
            )
        extra = ""
        if name == "gpt":
            extra = f"  config={gpt.config}"
        if name == "cnn":
            extra = f"  config=emb{emb_cnn}/ch{ch_cnn}/L{n_cnn}"
        log(f"  {name}: params={count_params(catalog[name][0]):,}{extra}")

    rec: Dict = {
        "protocol": {
            "description": "public char-LM: held-out bpc/ppl + fixed-prompt samples",
            "hf": args.hf,
            "hf_limit": args.hf_limit,
            "builtin": args.builtin,
            "seq_len": args.seq_len,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "grid_size": args.grid_size,
            "seed": args.seed,
            "standard_models": list(STANDARD_MODELS),
            "note": (
                "gpt = tiny from-scratch decoder Transformer, NOT pretrained GPT-4; "
                "suite = LSTM / RNN / CNN / GPT / brain / brain_wm"
            ),
        },
        "corpus": {
            "train_chars": int(train.numel()),
            "val_chars": int(val.numel()),
            "test_chars": int(test.numel()),
            "vocab": tok.vocab_size,
        },
        "params": {n: count_params(catalog[n][0]) for n in want},
        "gpt_config": gpt.config if "gpt" in want else None,
        "cnn_config": (
            {"embed_dim": emb_cnn, "channels": ch_cnn, "n_layers": n_cnn}
            if "cnn" in want else None
        ),
        "results": {},
        "curves": {},
        "samples": {},
    }

    samples_path = args.samples_json or args.json.replace(".json", "_samples.json")

    for name in want:
        model, loss_on = catalog[name]
        log(f"==== TRAIN {name} ====")
        summary, curve = train_one(
            name, model, loss_on, train, val,
            steps=args.steps, seq_len=args.seq_len, batch_size=args.batch_size,
            lr=args.lr, grad_clip=args.grad_clip, eval_every=args.eval_every,
            max_windows=args.max_windows, seed=args.seed, device=dev,
        )
        rec["results"][name] = summary
        rec["curves"][name] = curve

        if "diverged_at" not in summary:
            # test split metrics at best-val weights
            te = eval_metrics(
                model, loss_on, test,
                seq_len=args.seq_len, batch_size=args.batch_size,
                max_windows=args.max_windows, device=dev,
            )
            summary["test_bpc"] = te["bpc"]
            summary["test_ppl"] = te["ppl"]
            rec["results"][name] = summary
            log(f"  {name} test_bpc={te['bpc']:.4f}  test_ppl={te['ppl']:.2f}")

            # generations (non-fatal: metrics already recorded; samples secondary)
            gens = []
            try:
                for prompt in DEFAULT_PROMPTS:
                    if name in ("brain", "brain_wm"):
                        cont = model.generate(
                            tok, prompt=prompt, max_new_tokens=args.gen_tokens,
                            temperature=args.temperature, top_k=40,
                        )
                    else:
                        cont = generate_baseline(
                            model, tok, prompt, max_new_tokens=args.gen_tokens,
                            temperature=args.temperature, top_k=40, device=dev,
                        )
                    gens.append({"prompt": prompt, "continuation": cont, "full": prompt + cont})
                    log(f"  [{name} sample] {prompt!r} -> {cont[:120]!r}...")
                rec["samples"][name] = gens
            except Exception as exc:  # noqa: BLE001 — keep suite metrics if samples fail
                log(f"  WARNING: {name} sample generation failed: {exc!r}")
                rec["samples"][name] = {"error": repr(exc)}

        # Persist after every model so a late crash cannot drop completed metrics.
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rec, f, indent=2)
        with open(samples_path, "w") as f:
            json.dump(rec.get("samples", {}), f, indent=2)

    # leaderboard
    log("==== PUBLIC LM LEADERBOARD (lower bpc better) ====")
    rows = []
    for name, s in rec["results"].items():
        if "best_val_bpc" in s:
            rows.append((s["best_val_bpc"], name, s))
    rows.sort()
    for bpc, name, s in rows:
        log(
            f"  {name:<8} val_bpc={bpc:.4f}  val_ppl={s.get('best_val_ppl', float('nan')):.2f}  "
            f"test_bpc={s.get('test_bpc', float('nan')):.4f}  params={s['params']:,}"
        )

    with open(args.json, "w") as f:
        json.dump(rec, f, indent=2)
    log(f"wrote {args.json}")
    log(f"wrote samples {samples_path}")


if __name__ == "__main__":
    main()
