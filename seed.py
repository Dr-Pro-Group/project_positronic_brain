"""
Seed a Positronic Brain from public multimodal datasets *before* live interaction.

Streams small, capped subsets of public Hugging Face datasets (image+caption and
speech+transcript), encodes them with the multimodal encoders, and trains the
brain online (self-supervised reconstruction + cross-modal recall). The result is
a checkpoint you can keep teaching live via ``interact.py``.

Examples
--------
    # ~1,000 samples: 700 image+text (Flickr30k) + 300 audio+text (MINDS-14)
    python seed.py --image-text 700 --audio-text 300 --out trained_models/seeded_brain.pt

    # offline smoke test (no downloads): the encoders fall back, but the public
    # datasets still need network — use a tiny cap or --image-text 0 --audio-text 0
    python seed.py --image-text 50 --audio-text 50 --grid-size 8

Then continue teaching it live:
    python interact.py --load trained_models/seeded_brain.pt --modalities image text audio
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from positronic_brain.online import LiveBrain, OnlineConfig
from positronic_brain import datasets as ds


def _modalities(image_text: int, audio_text: int):
    mods = set()
    if image_text > 0:
        mods.update(("image", "text"))
    if audio_text > 0:
        mods.update(("audio", "text"))
    return sorted(mods) or ["image", "text"]


def _seed_from(stream, live: LiveBrain, total: int, label: str, report_every: int):
    """Feed a raw-sample stream into the live brain; return (#seen, last_loss)."""
    seen, last_loss, t0 = 0, None, time.time()
    for sample in stream:
        try:
            state = live.perceive(**sample)
        except Exception as exc:  # encoder/IO hiccup on one sample: skip it
            print(f"  [{label}] skipped a sample ({exc})")
            continue
        if state.get("loss") is not None:
            last_loss = float(state["loss"])
        seen += 1
        if seen % report_every == 0 or seen == total:
            rate = seen / max(time.time() - t0, 1e-6)
            loss_s = f"{last_loss:.5f}" if last_loss is not None else "  n/a"
            print(f"  [{label}] {seen:>5}/{total}  loss={loss_s}  ({rate:4.1f}/s)")
    return seen, last_loss


def main() -> None:
    p = argparse.ArgumentParser(description="Seed a Positronic Brain from public multimodal data.")
    p.add_argument("--image-text", type=int, default=700, help="image+caption samples to stream")
    p.add_argument("--audio-text", type=int, default=300, help="speech+transcript samples to stream")
    p.add_argument("--grid-size", type=int, default=12, help="cube side (grid_size**3 neurons)")
    p.add_argument("--no-pretrained", action="store_true", help="force offline FallbackEncoder")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--buffer", type=int, default=1024, help="replay buffer capacity")
    p.add_argument("--final-consolidate", type=int, default=200,
                   help="extra gradient steps after streaming")
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="trained_models/seeded_brain.pt")
    args = p.parse_args()

    modalities = _modalities(args.image_text, args.audio_text)
    print(f"Building live brain (modalities={modalities}, grid_size={args.grid_size})...")
    online_cfg = OnlineConfig(
        lr=args.lr, train_every=args.train_every, train_steps=args.train_steps,
        batch_size=args.batch_size, buffer_capacity=args.buffer, seed=args.seed,
    )
    live = LiveBrain.create(
        modalities=modalities,
        grid_size=args.grid_size,
        prefer_pretrained=not args.no_pretrained,
        online_config=online_cfg,
        device=args.device,
        seed=args.seed,
    )
    enc_kinds = {m: type(e).__name__ for m, e in live.encoders.items()}
    print(f"Brain: {live.brain.num_neurons} neurons, device {live.brain.device}")
    print(f"Encoders: {enc_kinds}")

    total_seen = 0
    if args.image_text > 0:
        print(f"\nStreaming {args.image_text} image+text samples...")
        n, _ = _seed_from(ds.stream_image_text(args.image_text), live,
                          args.image_text, "img+txt", args.report_every)
        total_seen += n
    if args.audio_text > 0:
        print(f"\nStreaming {args.audio_text} audio+text samples...")
        n, _ = _seed_from(ds.stream_audio_text(args.audio_text), live,
                          args.audio_text, "aud+txt", args.report_every)
        total_seen += n

    if args.final_consolidate > 0 and total_seen > 0:
        print(f"\nFinal consolidation: {args.final_consolidate} gradient steps...")
        loss = live.learner.consolidate(steps=args.final_consolidate)
        print(f"  final reconstruction loss = {loss:.5f}")

    if total_seen == 0:
        print("\nNo samples were ingested (check network / dataset availability). Nothing saved.")
        return

    live.save(args.out)
    print(f"\nSeeded brain saved to {args.out}  ({total_seen} samples ingested).")
    print(f"Continue teaching it live:\n"
          f"  python interact.py --load {args.out} --modalities {' '.join(modalities)}")


if __name__ == "__main__":
    main()
