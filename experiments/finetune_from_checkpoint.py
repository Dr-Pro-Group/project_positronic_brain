#!/usr/bin/env python
"""
Continue training (SFT / domain fine-tune) from a saved checkpoint.

Use after pretrain on public data:

    # fine-tune brain_wm on more TinyStories or domain shards
    python experiments/finetune_from_checkpoint.py \\
        --checkpoint checkpoints/llm_bench_tinystories_g12/brain_wm.pt \\
        --work-dir data/llm_tinystories \\
        --steps 5000 --lr 2e-4 \\
        --out-checkpoint checkpoints/ft_tinystories/brain_wm.pt \\
        --json runs/finetune_brain_wm.json

Supports brain / brain_wm and baselines (lstm, rnn, cnn, gpt).
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

from positronic_brain.checkpoints import load_baseline, load_brain, save_model_bundle
from positronic_brain.disk_data import MemmapTokenStore
from positronic_brain.language import BrainLanguageModel
from positronic_brain.utils import get_device
from experiments.llm_public_benchmark import eval_store, estimate_bytes_per_token, log
from experiments.matched_experiment import count_params


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--work-dir", required=True, help="prepared disk store for fine-tune data")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4, help="lower LR for fine-tune")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--max-windows", type=int, default=64)
    p.add_argument("--device", default="mps")
    p.add_argument("--out-checkpoint", required=True)
    p.add_argument("--json", default="runs/finetune.json")
    args = p.parse_args()

    device = get_device(args.device)
    # Peek kind
    try:
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kind = raw.get("kind", "brain")
    if kind in ("brain", "brain_wm") or "lm_config" in raw:
        model, tok, extra = load_brain(args.checkpoint, device=device)
        name = extra.get("name") or ("brain_wm" if model.config.use_wm_attn else "brain")
        loss_fn = model.loss_on
    else:
        model, tok, meta = load_baseline(args.checkpoint, device=device)
        name = meta["kind"]
        loss_fn = model.loss_on

    done = os.path.join(args.work_dir, "prepare_done.json")
    with open(done) as f:
        prep = json.load(f)
    if tok is None:
        from positronic_brain.subword import SubwordTokenizer
        tok = SubwordTokenizer.load(prep["tokenizer_path"])

    train = MemmapTokenStore(prep["mmap_path"], split="train")
    val = MemmapTokenStore(prep["mmap_path"], split="val")
    bpt = estimate_bytes_per_token(tok, train)
    log(f"finetune {name} from {args.checkpoint} on {args.work_dir} steps={args.steps}")

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()
    t0 = time.time()
    best_val = float("inf")
    best_state = None
    curve = []
    run_ce = 0.0
    n_run = 0

    for step in range(1, args.steps + 1):
        batch = train.sample_batch(args.seq_len, args.batch_size, device)
        loss = loss_fn(batch)
        if not math.isfinite(float(loss.item())):
            log(f"DIVERGED at {step}")
            break
        opt.zero_grad()
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        run_ce += float(loss.item())
        n_run += 1
        if step % args.eval_every == 0 or step == args.steps:
            va = eval_store(
                model, loss_fn, val,
                seq_len=args.seq_len, batch_size=args.batch_size,
                max_windows=args.max_windows, device=device, bytes_per_token=bpt,
            )
            avg = run_ce / max(n_run, 1)
            run_ce = n_run = 0
            curve.append({"step": step, "train_ce": avg, **va})
            log(f"  step {step}/{args.steps} train_ce={avg:.4f} val_bpc={va['bpc']:.4f}")
            if va["bpc"] < best_val:
                best_val = va["bpc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out_checkpoint) or ".", exist_ok=True)
    save_model_bundle(
        name, model, args.out_checkpoint, tokenizer=tok,
        metrics={"best_val_bpc": best_val, "steps": args.steps},
        protocol={"from_checkpoint": args.checkpoint, "work_dir": args.work_dir},
    )
    rec = {
        "from_checkpoint": args.checkpoint,
        "out_checkpoint": args.out_checkpoint,
        "name": name,
        "params": count_params(model),
        "best_val_bpc": best_val,
        "curve": curve,
        "wall_min": round((time.time() - t0) / 60, 2),
    }
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(rec, f, indent=2)
    log(f"wrote {args.out_checkpoint} and {args.json}")


if __name__ == "__main__":
    main()
