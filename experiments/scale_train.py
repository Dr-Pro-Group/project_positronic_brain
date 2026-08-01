#!/usr/bin/env python
"""
Scale-path trainer: public LLM data on disk + optional modular multi-area brain.

Phases covered
--------------
A. Stream FineWeb/C4/TinyStories → shards → BPE → memmap tokens (disk as data RAM)
B. Single large cube with grad_checkpoint + small batch
C. Modular areas: train one area at a time, offload others to disk

Examples
--------
  # prepare disk store (TinyStories by default — FineWeb needs bandwidth)
  python experiments/scale_train.py prepare \\
      --dataset tinystories --max-docs 5000 --max-chars 5000000 \\
      --work-dir data/scale_tinystories

  # train single brain on memmap tokens
  python experiments/scale_train.py train-single \\
      --work-dir data/scale_tinystories --grid-size 16 --steps 2000 \\
      --json runs/scale_single_g16.json

  # modular 3-area brain, train areas sequentially
  python experiments/scale_train.py train-modular \\
      --work-dir data/scale_tinystories --grid-size 12 --n-areas 3 \\
      --steps-per-area 1500 --json runs/scale_modular_g12.json

Run from repository root. Prefer Mini with ``--device mps``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.disk_data import (
    PUBLIC_LM_PRESETS,
    MemmapTokenStore,
    prepare_public_lm_disk,
)
from positronic_brain.language import BrainLanguageModel, LMConfig
from positronic_brain.modular import ModularBrainLM, ModularConfig
from positronic_brain.subword import SubwordTokenizer
from positronic_brain.utils import get_device


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def count_params(m: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def eval_bpc(
    model,
    store: MemmapTokenStore,
    *,
    seq_len: int,
    batch_size: int,
    max_windows: int,
    device,
) -> Dict[str, float]:
    model.eval()
    total_ce = 0.0
    n_tok = 0
    for batch in store.eval_batches(seq_len, batch_size, max_windows, device):
        loss = model.loss_on(batch)
        # loss is mean over tokens in batch
        n = batch[:, 1:].numel()
        total_ce += float(loss.item()) * n
        n_tok += n
    model.train()
    if n_tok == 0:
        return {"ce": float("nan"), "bpc": float("nan"), "ppl": float("nan"), "tokens": 0}
    ce = total_ce / n_tok
    return {
        "ce": ce,
        "bpc": ce / math.log(2),
        "ppl": math.exp(min(ce, 20.0)),
        "tokens": n_tok,
    }


def cmd_prepare(args: argparse.Namespace) -> None:
    out = prepare_public_lm_disk(
        args.dataset,
        args.work_dir,
        max_docs=args.max_docs,
        max_chars=args.max_chars,
        vocab_size=args.vocab_size,
        force=args.force,
    )
    log(f"prepared: {json.dumps({k: out[k] for k in out if k != 'mmap_meta'}, indent=2)}")


def cmd_train_single(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    done = os.path.join(args.work_dir, "prepare_done.json")
    with open(done) as f:
        prep = json.load(f)
    tok = SubwordTokenizer.load(prep["tokenizer_path"])
    train = MemmapTokenStore(prep["mmap_path"], split="train")
    val = MemmapTokenStore(prep["mmap_path"], split="val")
    log(
        f"single cube G={args.grid_size} vocab={tok.vocab_size} "
        f"train_tokens={train.n_tokens:,} device={device}"
    )

    cfg = LMConfig(
        grid_size=args.grid_size,
        embed_dim=args.embed_dim,
        inner_steps=args.inner_steps,
        seed=args.seed,
        grad_checkpoint=args.grad_checkpoint,
        readout_width=args.readout_width or None,
    )
    model = BrainLanguageModel(tok.vocab_size, cfg, device=device)
    n_params = count_params(model)
    log(f"params={n_params:,} neurons={model.num_neurons:,} grad_ckpt={args.grad_checkpoint}")

    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    t0 = time.time()
    curve: List[Dict] = []
    best_bpc = float("inf")
    best_step = 0
    best_state = None
    running = 0.0
    n_run = 0

    model.train()
    for step in range(1, args.steps + 1):
        batch = train.sample_batch(args.seq_len, args.batch_size, device)
        loss = model.loss_on(batch)
        if not math.isfinite(float(loss.item())):
            log(f"DIVERGED at step {step}")
            break
        opt.zero_grad()
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        running += float(loss.item())
        n_run += 1

        if step % args.eval_every == 0 or step == args.steps:
            metrics = eval_bpc(
                model, val,
                seq_len=args.seq_len, batch_size=args.batch_size,
                max_windows=args.max_windows, device=device,
            )
            avg = running / max(n_run, 1)
            running = 0.0
            n_run = 0
            row = {
                "step": step,
                "train_ce": avg,
                **metrics,
                "wall_s": round(time.time() - t0, 1),
            }
            curve.append(row)
            log(
                f"  step {step}/{args.steps} train_ce={avg:.4f} "
                f"val_bpc={metrics['bpc']:.4f} ppl={metrics['ppl']:.2f}"
            )
            if metrics["bpc"] < best_bpc:
                best_bpc = metrics["bpc"]
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            rec = {
                "mode": "single",
                "dataset": prep.get("dataset"),
                "grid_size": args.grid_size,
                "params": n_params,
                "neurons": model.num_neurons,
                "vocab_size": tok.vocab_size,
                "best_val_bpc": best_bpc,
                "best_step": best_step,
                "curve": curve,
                "protocol": vars(args),
            }
            os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
            with open(args.json, "w") as f:
                json.dump(rec, f, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)
    test = MemmapTokenStore(prep["mmap_path"], split="test")
    te = eval_bpc(
        model, test,
        seq_len=args.seq_len, batch_size=args.batch_size,
        max_windows=args.max_windows, device=device,
    )
    log(f"DONE best_val_bpc={best_bpc:.4f} @ {best_step}  test_bpc={te['bpc']:.4f}")
    with open(args.json) as f:
        rec = json.load(f)
    rec["test_bpc"] = te["bpc"]
    rec["test_ppl"] = te["ppl"]
    rec["wall_min"] = round((time.time() - t0) / 60, 2)
    with open(args.json, "w") as f:
        json.dump(rec, f, indent=2)
    if args.checkpoint:
        model.save(args.checkpoint, tokenizer=None)
        tok.save(args.checkpoint + ".tokenizer.json")
        log(f"saved {args.checkpoint}")


def cmd_train_modular(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    done = os.path.join(args.work_dir, "prepare_done.json")
    with open(done) as f:
        prep = json.load(f)
    tok = SubwordTokenizer.load(prep["tokenizer_path"])
    train = MemmapTokenStore(prep["mmap_path"], split="train")
    val = MemmapTokenStore(prep["mmap_path"], split="val")

    mcfg = ModularConfig.default_chain(
        grid_size=args.grid_size, n_areas=args.n_areas, seed=args.seed,
    )
    mcfg.embed_dim = args.embed_dim
    mcfg.inner_steps = args.inner_steps
    mcfg.train_one_area = True

    model = ModularBrainLM(tok.vocab_size, mcfg, device=device)
    area_dir = args.area_dir or os.path.join(args.work_dir, "areas")
    os.makedirs(area_dir, exist_ok=True)
    log(
        f"modular areas={model.area_order} total_neurons={model.total_neurons():,} "
        f"params={model.count_params():,} device={device}"
    )

    rec: Dict = {
        "mode": "modular",
        "dataset": prep.get("dataset"),
        "areas": model.area_order,
        "total_neurons": model.total_neurons(),
        "vocab_size": tok.vocab_size,
        "area_results": {},
        "protocol": vars(args),
    }
    t0 = time.time()

    # Sequential curriculum: train each area as active (others frozen).
    for area_name in model.area_order:
        model.set_active_area(area_name)
        n_train = model.count_params(trainable_only=True)
        log(f"==== AREA {area_name} trainable_params={n_train:,} ====")
        opt = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=args.lr
        )
        curve: List[Dict] = []
        best_bpc = float("inf")
        best_step = 0
        best_state = None
        running = 0.0
        n_run = 0
        model.train()
        for step in range(1, args.steps_per_area + 1):
            batch = train.sample_batch(args.seq_len, args.batch_size, device)
            loss = model.loss_on(batch)
            if not math.isfinite(float(loss.item())):
                log(f"  {area_name} DIVERGED at step {step}")
                break
            opt.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    args.grad_clip,
                )
            opt.step()
            running += float(loss.item())
            n_run += 1
            if step % args.eval_every == 0 or step == args.steps_per_area:
                metrics = eval_bpc(
                    model, val,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                    max_windows=args.max_windows, device=device,
                )
                avg = running / max(n_run, 1)
                running = 0.0
                n_run = 0
                curve.append({
                    "step": step, "train_ce": avg, **metrics,
                    "wall_s": round(time.time() - t0, 1),
                })
                log(
                    f"  [{area_name}] step {step}/{args.steps_per_area} "
                    f"train_ce={avg:.4f} val_bpc={metrics['bpc']:.4f}"
                )
                if metrics["bpc"] < best_bpc:
                    best_bpc = metrics["bpc"]
                    best_step = step
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

        if best_state is not None:
            model.load_state_dict(best_state)
        rec["area_results"][area_name] = {
            "best_val_bpc": best_bpc,
            "best_step": best_step,
            "trainable_params": n_train,
            "curve": curve,
        }
        # Disk offload pattern: save all areas each stage
        model.save_all_areas(area_dir)
        log(f"  saved areas → {area_dir}")
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rec, f, indent=2)

        if args.offload_inactive and area_name != model.area_order[-1]:
            # Demonstrate disk offload of completed early areas (reload all before next)
            # Keep all resident for correctness of multi-area forward; document
            # offload API via save_all_areas. Full delete+reload is optional.
            pass

    test = MemmapTokenStore(prep["mmap_path"], split="test")
    te = eval_bpc(
        model, test,
        seq_len=args.seq_len, batch_size=args.batch_size,
        max_windows=args.max_windows, device=device,
    )
    rec["test_bpc"] = te["bpc"]
    rec["test_ppl"] = te["ppl"]
    rec["wall_min"] = round((time.time() - t0) / 60, 2)
    rec["final_params"] = model.count_params()
    with open(args.json, "w") as f:
        json.dump(rec, f, indent=2)
    log(f"MODULAR DONE test_bpc={te['bpc']:.4f} wall={rec['wall_min']} min")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="stream public LM → disk shards → BPE → memmap")
    p_prep.add_argument(
        "--dataset", default="tinystories",
        help=f"preset {sorted(PUBLIC_LM_PRESETS)} or hub path",
    )
    p_prep.add_argument("--work-dir", default="data/scale_store")
    p_prep.add_argument("--max-docs", type=int, default=10_000)
    p_prep.add_argument("--max-chars", type=int, default=10_000_000)
    p_prep.add_argument("--vocab-size", type=int, default=4096)
    p_prep.add_argument("--force", action="store_true")

    def add_train_args(sp):
        sp.add_argument("--work-dir", default="data/scale_store")
        sp.add_argument("--device", default="mps")
        sp.add_argument("--steps", type=int, default=2000)
        sp.add_argument("--steps-per-area", type=int, default=1500)
        sp.add_argument("--seq-len", type=int, default=64)
        sp.add_argument("--batch-size", type=int, default=8)
        sp.add_argument("--lr", type=float, default=8e-4)
        sp.add_argument("--grad-clip", type=float, default=0.5)
        sp.add_argument("--eval-every", type=int, default=200)
        sp.add_argument("--max-windows", type=int, default=64)
        sp.add_argument("--grid-size", type=int, default=12)
        sp.add_argument("--embed-dim", type=int, default=64)
        sp.add_argument("--inner-steps", type=int, default=2)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--json", default="runs/scale_train.json")
        sp.add_argument("--checkpoint", default="")
        sp.add_argument("--grad-checkpoint", action="store_true")
        sp.add_argument("--readout-width", type=int, default=0)
        sp.add_argument("--n-areas", type=int, default=3)
        sp.add_argument("--area-dir", default="")
        sp.add_argument("--offload-inactive", action="store_true")

    p_s = sub.add_parser("train-single", help="single-cube LM on memmap tokens")
    add_train_args(p_s)
    p_m = sub.add_parser("train-modular", help="multi-area sequential train")
    add_train_args(p_m)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "train-single":
        cmd_train_single(args)
    elif args.cmd == "train-modular":
        cmd_train_modular(args)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
