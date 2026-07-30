#!/usr/bin/env python
"""
Multi-seed replications of the load-bearing non-density levers at G=16.

Seed 42 already lives in runs/controls.json and long_program_stage12.json.
This adds seeds 43/44 (or whatever --seeds says) so error bars exist for:

  * baseline          (shipped g_max=0.4)
  * g_max 0.691       (weight-init fix; beat every bio flag at seed 42)
  * --stp             (only quality-positive mechanism at seed 42)

Density multi-seed is owned by experiments/replicate.py.

    python experiments/lever_seeds.py --seeds 43,44 --steps 3000

Run from the repository root. Uses MPS by default.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.utils import get_device


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="43,44")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--json", default="runs/lever_seeds.json")
    args = p.parse_args()

    dev = get_device(args.device)
    tr_t, va_t, _ = load_corpus_splits(
        hf_chat="soda", hf_chat_limit=4000, builtin=True,
        repeats=60, seed=42, val_frac=0.1, test_frac=0.0,
    )
    tok = CharTokenizer.from_text(tr_t)
    tr = torch.tensor(tok.encode(tr_t), dtype=torch.long)
    va = torch.tensor(tok.encode(va_t), dtype=torch.long)
    log(f"corpus {tr.numel():,} chars vocab {tok.vocab_size} device={dev}")

    def batch(b=16, sl=48):
        ix = torch.randint(0, tr.numel() - sl - 1, (b,))
        return torch.stack([tr[i : i + sl + 1] for i in ix]).to(dev)

    @torch.no_grad()
    def ev(m, sl=48, b=16, mw=256):
        m.eval()
        st = list(range(0, va.numel() - sl - 1, sl))[:mw]
        tot = n = 0
        for i in range(0, len(st), b):
            c = torch.stack([va[s : s + sl + 1] for s in st[i : i + b]]).to(dev)
            lg, _ = m(c[:, :-1])
            tot += float(
                torch.nn.functional.cross_entropy(
                    lg.reshape(-1, m.vocab_size),
                    c[:, 1:].reshape(-1),
                    reduction="sum",
                ).item()
            )
            n += c[:, 1:].numel()
        return (tot / n) / math.log(2)

    def run(label, seed, overrides):
        cfg = LMConfig(
            grid_size=args.grid,
            embed_dim=64,
            inner_steps=3,
            seed=seed,
            brain_overrides=dict(overrides),
        )
        m = BrainLanguageModel(tok.vocab_size, cfg, device=dev)
        opt = torch.optim.Adam(m.parameters(), lr=8e-4)
        t0 = time.time()
        m.train()
        for s in range(args.steps):
            l = m.loss_on(batch())
            if not math.isfinite(float(l.item())):
                return {"label": label, "seed": seed, "diverged_at": s, **overrides}
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
            opt.step()
        r = {
            "label": label,
            "seed": seed,
            "grid": args.grid,
            "bpc": ev(m),
            "neurons": m.brain.num_neurons,
            "edges": m.brain.num_edges,
            "params": sum(q.numel() for q in m.parameters() if q.requires_grad),
            "min": round((time.time() - t0) / 60, 1),
            **overrides,
        }
        log(f"  {label:<18} s{seed} bpc {r['bpc']:.4f}  ({r['min']:.0f} min)")
        return r

    arms = [
        ("baseline", {}),
        ("g_max 0.691", {"g_max": 0.691}),
        ("stp", {"use_stp": True}),
    ]

    rec = {"args": vars(args), "runs": []}
    def add(r):
        rec["runs"].append(r)
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rec, f, indent=2)

    seeds = [int(s) for s in args.seeds.split(",")]
    log(f"LEVER SEEDS — seeds {seeds}, grid {args.grid}, steps {args.steps}")
    for sd in seeds:
        for label, ov in arms:
            add(run(label, sd, ov))
    log(f"LEVER SEEDS COMPLETE -> {args.json}")


if __name__ == "__main__":
    main()
