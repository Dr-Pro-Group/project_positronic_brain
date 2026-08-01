"""
Do the zones specialise for their stream, or is that just where the wires go?

The multi-stream brain injects each modality into a spatially distinct zone, and
it is tempting to read a zone's preference for its own stream as *emergent
functional specialisation*. Most of that preference is not emergent at all: a zone
sitting at an entry door responds to whatever comes through the door, and it would
do so in an untrained network with random weights. Reporting that number alone
would be measuring the wiring diagram and calling it a result.

This script separates the architectural prior from anything training adds, using
three controls:

  * an **untrained** network, measured identically, as the floor;
  * a split between **entry zones** (which receive a stream directly) and
    **non-entry zones** (which receive none) --- specialisation that appears in a
    non-entry zone cannot be explained by the input routing;
  * **held-out scenes** for every probe, and probes repeated over several
    initialisations, since a linear probe's accuracy depends on its own seed.

Scenes are counterbalanced: every stream in a scene carries the same latent
content class, so a zone can organise itself by *stream* or by *content*, and the
metrics can tell which.

    python experiments/zone_specialization.py --steps 600 --seeds 3

Writes a JSON record and prints a summary. Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain import specialization as spec
from positronic_brain.multimodal import MultiStreamBrain, StreamSpec
from positronic_brain.multimodal_data import batch_scenes, per_stream_samples, synthetic_scenes
from positronic_brain.utils import get_device

STREAMS = [("vision", "Visual"), ("audio", "Auditory"), ("text", "Association")]


def build(dim: int, grid: int, seed: int, device) -> MultiStreamBrain:
    return MultiStreamBrain(
        [StreamSpec(name, dim, zone) for name, zone in STREAMS],
        grid_size=grid, seed=seed, device=device)


def measure(model, samples, probe_repeats: int) -> Dict:
    """Selectivity, stream-decoding accuracy, and lesion deficits."""
    sel = spec.selectivity_index(model, samples)

    # zone_decoding_accuracy trains a fresh probe whose initialisation is not
    # pinned by the caller, so a single number is noisy. Repeat and report spread.
    accs = [spec.zone_decoding_accuracy(model, samples) for _ in range(probe_repeats)]

    entry_zones = set(model.routing.values())
    zone_names = model.brain.config.zone_names
    lesions = {}
    for zone in zone_names:
        deficits = spec.lesion_effect(model, samples, zone)
        lesions[zone] = {k: float(v) for k, v in deficits.items()}

    return {
        "selectivity": {z: {"prefers": s, "index": float(v)} for z, (s, v) in sel.items()},
        "decode_acc_mean": float(statistics.fmean(accs)),
        "decode_acc_std": float(statistics.pstdev(accs)) if len(accs) > 1 else 0.0,
        "decode_acc_runs": [float(a) for a in accs],
        "entry_zones": sorted(entry_zones),
        "non_entry_zones": [z for z in zone_names if z not in entry_zones],
        "lesion": lesions,
    }


def train(model, scenes, steps: int, batch: int, lr: float, seed: int, device,
          modality_dropout: float) -> List[float]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    model.train()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(scenes), size=batch)
        loss = model.loss(batch_scenes(scenes, idx, device=device),
                          modality_dropout=modality_dropout)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if step % max(steps // 6, 1) == 0:
            print(f"    step {step:>4}/{steps}  loss={statistics.fmean(losses[-50:]):.4f}",
                  flush=True)
    return losses


def mean_selectivity(block: Dict, zones: List[str]) -> float:
    vals = [block["selectivity"][z]["index"] for z in zones if z in block["selectivity"]]
    return statistics.fmean(vals) if vals else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--dim", type=int, default=64, help="per-stream embedding width")
    p.add_argument("--scenes", type=int, default=1200)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--n-content", type=int, default=6)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--probe-repeats", type=int, default=5)
    p.add_argument("--modality-dropout", type=float, default=0.5,
                   help="probability of masking a stream during training. This is "
                        "the knob that decides the outcome: masking forces the network "
                        "to reconstruct a missing stream from the others, which needs "
                        "shared cross-modal representation rather than zone-local ones. "
                        "Set 0.0 to train each stream without that pressure.")
    p.add_argument("--device", default="auto")
    p.add_argument("--json", default="runs/zone_specialization.json")
    args = p.parse_args()

    device = get_device(args.device)
    dims = {name: args.dim for name, _ in STREAMS}
    record = {"args": vars(args), "device": str(device), "seeds": []}

    for i in range(args.seeds):
        seed = args.seed0 + i
        print(f"\n=== seed {seed} ===", flush=True)
        scenes, _labels = synthetic_scenes(dims, args.scenes,
                                           n_content=args.n_content, seed=seed)
        n_val = int(len(scenes) * args.val_frac)
        train_scenes, val_scenes = scenes[n_val:], scenes[:n_val]
        val_samples = per_stream_samples(val_scenes, device=device)

        model = build(args.dim, args.grid_size, seed, device)
        print("  [untrained baseline]", flush=True)
        before = measure(model, val_samples, args.probe_repeats)

        losses = train(model, train_scenes, args.steps, args.batch_size,
                       args.lr, seed, device, args.modality_dropout)
        model.eval()
        after = measure(model, val_samples, args.probe_repeats)

        record["seeds"].append({
            "seed": seed, "loss_first": losses[0], "loss_last": statistics.fmean(losses[-50:]),
            "untrained": before, "trained": after,
        })
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=2)

    summarise(record)
    print(f"\n[saved] {args.json}")


def summarise(record: Dict) -> None:
    rows = record["seeds"]
    if not rows:
        return
    entry = rows[0]["trained"]["entry_zones"]
    non_entry = rows[0]["trained"]["non_entry_zones"]
    n_streams = len(STREAMS)

    def agg(phase: str, fn) -> str:
        vals = [fn(r[phase]) for r in rows]
        return f"{statistics.fmean(vals):.3f}" + (
            f" ± {statistics.pstdev(vals):.3f}" if len(vals) > 1 else "")

    print("\n" + "=" * 74)
    print(f"{'metric':<34}{'untrained':>18}{'trained':>18}")
    print("-" * 74)
    print(f"{'selectivity, entry zones':<34}"
          f"{agg('untrained', lambda b: mean_selectivity(b, entry)):>18}"
          f"{agg('trained', lambda b: mean_selectivity(b, entry)):>18}")
    print(f"{'selectivity, NON-entry zones':<34}"
          f"{agg('untrained', lambda b: mean_selectivity(b, non_entry)):>18}"
          f"{agg('trained', lambda b: mean_selectivity(b, non_entry)):>18}")
    print(f"{'stream-decoding accuracy':<34}"
          f"{agg('untrained', lambda b: b['decode_acc_mean']):>18}"
          f"{agg('trained', lambda b: b['decode_acc_mean']):>18}")
    print("=" * 74)
    print(f"entry zones     : {', '.join(entry)}")
    print(f"non-entry zones : {', '.join(non_entry)}")
    print(f"decoding chance : {1/n_streams:.3f}")
    print("\nSelectivity at an entry zone is largely architectural — the untrained\n"
          "column is the floor it has to beat. Movement in the NON-entry zones is\n"
          "the part that routing cannot explain.")


if __name__ == "__main__":
    main()
