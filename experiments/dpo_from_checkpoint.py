#!/usr/bin/env python
"""
Preference / alignment fine-tune (DPO) from a saved brain checkpoint.

This is the practical RL-adjacent path today: no PPO yet, but real preference
optimisation on (prompt, chosen, rejected) triples with the brain's membrane
state as recurrent policy memory.

    python experiments/dpo_from_checkpoint.py \\
        --checkpoint checkpoints/llm_bench_wikitext_g12/brain_wm.pt \\
        --pairs data/preferences_example.jsonl \\
        --epochs 2 --beta 0.1 --lr 1e-4 \\
        --out-checkpoint checkpoints/dpo_brain_wm/brain_wm.pt

pairs JSONL lines: {"prompt": "...", "chosen": "...", "rejected": "..."}

See positronic_brain/preference.py for DPO math and BrainPolicy (PPO hook).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.checkpoints import load_brain, save_brain
from positronic_brain.preference import DPOConfig, dpo_finetune
from positronic_brain.utils import get_device


def load_pairs(path: str):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append((row["prompt"], row["chosen"], row["rejected"]))
    return pairs


def write_example_pairs(path: str) -> None:
    """Tiny synthetic pairs so the pipeline is runnable offline."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    examples = [
        {
            "prompt": "Once upon a time",
            "chosen": " there was a kind little girl who helped her friends.",
            "rejected": " asdf jkl qwerty zxcv nonsense garbage tokens.",
        },
        {
            "prompt": "The scientific method",
            "chosen": " starts with a question, a hypothesis, and careful tests.",
            "rejected": " is a purple banana that dances on the moon forever.",
        },
        {
            "prompt": "In the beginning",
            "chosen": " of the story, the forest was quiet and green.",
            "rejected": " 111222333!!!@@@### nonsense.",
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs", default="data/preferences_example.jsonl")
    p.add_argument("--write-example-pairs", action="store_true")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--device", default="mps")
    p.add_argument("--out-checkpoint", required=True)
    p.add_argument("--json", default="runs/dpo_finetune.json")
    args = p.parse_args()

    if args.write_example_pairs or not os.path.isfile(args.pairs):
        write_example_pairs(args.pairs)
        print(f"[dpo] wrote example pairs → {args.pairs}")

    device = get_device(args.device)
    model, tok, extra = load_brain(args.checkpoint, device=device)
    if tok is None:
        raise SystemExit("checkpoint has no tokenizer; save with tokenizer next time")

    pairs = load_pairs(args.pairs)
    print(f"[dpo] loaded {len(pairs)} pairs; finetuning {args.checkpoint}")
    cfg = DPOConfig(beta=args.beta, lr=args.lr, grad_clip=args.grad_clip)
    dpo_finetune(model, tok, pairs, config=cfg, epochs=args.epochs, log_every=1)

    os.makedirs(os.path.dirname(args.out_checkpoint) or ".", exist_ok=True)
    save_brain(
        model, args.out_checkpoint, tok,
        extra={
            "from_checkpoint": args.checkpoint,
            "pairs": args.pairs,
            "dpo": {"beta": args.beta, "lr": args.lr, "epochs": args.epochs},
            "parent": extra,
        },
    )
    rec = {
        "from_checkpoint": args.checkpoint,
        "out_checkpoint": args.out_checkpoint,
        "n_pairs": len(pairs),
        "epochs": args.epochs,
        "beta": args.beta,
        "lr": args.lr,
    }
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[dpo] wrote {args.out_checkpoint}")


if __name__ == "__main__":
    main()
