"""
Does removing Dale's law actually destabilise training?

The matched-budget table reports that the no-Dale ablation is the least accurate
brain variant. An earlier run also showed it diverging to ``NaN`` on one seed, and
that divergence was used to argue that the E/I sign constraint buys *stability* as
well as accuracy. That earlier run was later invalidated for an unrelated reason
(a validation split that shared sentences with train), and the two corrected runs
that replaced it showed no divergence at all — leaving the stability claim resting
on a retracted experiment.

The two runs differ in more than the split, though: the invalidated one used
``seq_len=64`` and the corrected ones used ``seq_len=48``. Backpropagation through
a longer window compounds the recurrent Jacobian more times, so sequence length is
exactly the knob one would expect to control a runaway-excitation failure. This
script separates the two explanations by sweeping seeds at both sequence lengths
on the *corrected* corpus, and reports how often each condition diverges.

Divergence is recorded as an event, not silently averaged away: a ``NaN`` is a
result, and a mean taken over the seeds that happened to survive is a biased
estimate of accuracy.

    python experiments/dale_stability.py --seeds 8 --steps 150 --hf-chat soda

Writes a JSON record and prints a summary table. Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.utils import get_device


def make_batch(data: torch.Tensor, seq_len: int, batch_size: int, device) -> torch.Tensor:
    """Sample ``batch_size`` random contiguous chunks of length ``seq_len + 1``."""
    n = data.numel() - seq_len - 1
    ix = torch.randint(0, max(n, 1), (batch_size,))
    return torch.stack([data[i : i + seq_len + 1] for i in ix]).to(device)


def remove_dale(model: BrainLanguageModel) -> None:
    """Free every synapse's sign, keeping the graph and magnitudes untouched.

    Identical to the patch used by ``matched_experiment.py`` so the two scripts
    ablate the same thing in the same way.
    """
    brain = model.brain
    brain.signed_weights = types.MethodType(lambda self: self.edge_weight, brain)


def train_until_diverged(
    model: BrainLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    *,
    steps: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    seed: int,
    device,
    label: str,
) -> Dict:
    """Train for ``steps``, stopping early if the loss becomes non-finite.

    Returns the step at which divergence happened (``None`` if it never did) and
    the held-out perplexity, which is ``None`` for a diverged run.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()

    diverged_at: Optional[int] = None
    for step in range(1, steps + 1):
        loss = model.loss_on(make_batch(train_data, seq_len, batch_size, device))
        if not math.isfinite(float(loss.item())):
            diverged_at = step
            break
        opt.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    ppl = None
    if diverged_at is None:
        model.eval()
        torch.manual_seed(seed + 999)
        with torch.no_grad():
            total = sum(float(model.loss_on(make_batch(val_data, seq_len, batch_size, device)).item())
                        for _ in range(16))
        ce = total / 16
        ppl = math.exp(ce) if math.isfinite(ce) else None

    elapsed = time.time() - t0
    verdict = f"diverged at step {diverged_at}" if diverged_at else f"ppl={ppl:.2f}"
    print(f"  [{label}] {verdict}  ({elapsed:.0f}s)", flush=True)
    return {"diverged_at": diverged_at, "ppl": ppl, "seconds": round(elapsed, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=8, help="how many seeds, counting up from --seed0")
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--steps", type=int, default=150,
                   help="divergence is an early event; 150 steps is well past where it appears")
    p.add_argument("--seq-lens", default="48,64",
                   help="comma-separated sequence lengths to compare")
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--repeats", type=int, default=60)
    p.add_argument("--hf-chat", default="soda")
    p.add_argument("--hf-chat-limit", type=int, default=2000)
    p.add_argument("--control-seeds", type=int, default=3,
                   help="seeds for the Dale-intact control at each sequence length")
    p.add_argument("--device", default="auto")
    p.add_argument("--json", default="runs/dale_stability.json")
    args = p.parse_args()

    device = get_device(args.device)
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    seeds = [args.seed0 + i for i in range(args.seeds)]

    record: Dict = {"args": vars(args), "device": str(device), "runs": []}

    for seed in seeds:
        # The split itself is seeded, matching the matched-experiment protocol, so
        # the corpus is rebuilt per seed rather than shared across the sweep.
        train_text, val_text, _ = load_corpus_splits(
            hf_chat=args.hf_chat, hf_chat_limit=args.hf_chat_limit,
            builtin=True, repeats=args.repeats, seed=seed,
            val_frac=args.val_frac, test_frac=0.0)
        tok = CharTokenizer.from_text(train_text)
        train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
        val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)
        print(f"\n=== seed {seed}  vocab={tok.vocab_size} "
              f"train={train_data.numel()} val={val_data.numel()} ===", flush=True)

        for seq_len in seq_lens:
            conditions = [("no-Dale", True)]
            if seed - args.seed0 < args.control_seeds:
                conditions.append(("Dale intact", False))

            for name, ablate in conditions:
                cfg = LMConfig(grid_size=args.grid_size, embed_dim=args.embed_dim,
                               inner_steps=args.inner_steps, seed=seed)
                model = BrainLanguageModel(tok.vocab_size, cfg, device=device)
                if ablate:
                    remove_dale(model)
                out = train_until_diverged(
                    model, train_data, val_data, steps=args.steps, seq_len=seq_len,
                    batch_size=args.batch_size, lr=args.lr, grad_clip=args.grad_clip,
                    seed=seed, device=device, label=f"{name} seq{seq_len} s{seed}")
                record["runs"].append({"seed": seed, "seq_len": seq_len,
                                       "condition": name, **out})

                # Written after every run so a long sweep survives interruption.
                os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
                with open(args.json, "w") as fh:
                    json.dump(record, fh, indent=2)

    summarise(record, seq_lens)
    print(f"\n[saved] {args.json}")


def summarise(record: Dict, seq_lens: List[int]) -> None:
    print("\n" + "=" * 68)
    print(f"{'condition':<14}{'seq_len':>9}{'seeds':>7}{'diverged':>10}{'rate':>8}   surviving ppl")
    print("-" * 68)
    for condition in ("no-Dale", "Dale intact"):
        for seq_len in seq_lens:
            rows = [r for r in record["runs"]
                    if r["condition"] == condition and r["seq_len"] == seq_len]
            if not rows:
                continue
            n = len(rows)
            bad = sum(1 for r in rows if r["diverged_at"] is not None)
            ppls = [r["ppl"] for r in rows if r["ppl"] is not None]
            mean = f"{sum(ppls) / len(ppls):.2f}" if ppls else "—"
            spread = (f" ± {(sum((x - sum(ppls) / len(ppls)) ** 2 for x in ppls) / len(ppls)) ** 0.5:.2f}"
                      if len(ppls) > 1 else "")
            print(f"{condition:<14}{seq_len:>9}{n:>7}{bad:>10}{bad / n:>8.0%}   {mean}{spread}")
    print("=" * 68)
    print("A diverged run contributes no perplexity; means are over surviving seeds only.")


if __name__ == "__main__":
    main()
