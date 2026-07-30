#!/usr/bin/env python
"""
Replicate the two load-bearing results across seeds.

Both were measured once. The pipeline is deterministic at fixed seed — the same
configuration reproduced to within 0.0003 bpc across independent invocations — so
what is unknown is not run-to-run drift but how much of each effect is seed
variance. A difference of 0.0735 bpc against a seed spread of 0.0206 is probably
real; measuring it three times says so rather than assuming it.

  density   grid12/k38 against grid16/k16 at matched recurrent parameters, the
            only clean positive architectural result in the investigation
  ladder    neuron count with the read-out held to a fixed width, whose 75.4%
            survival figure contradicts the 46% the manuscript published

Grid 24 is deliberately not replicated: at seven hours per run it would cost 42
hours by itself. The slope therefore rests on replicated endpoints (512 and 4,096)
with a single-seed extension to 13,824, and is reported that way.

    python experiments/replicate.py --seeds 43,44

Run from the repository root.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from positronic_brain.corpus import load_corpus_splits
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.utils import get_device

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="43,44", help="seeds to ADD; 42 is already measured")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--json", default="runs/replication.json")
    args = p.parse_args()

    dev = get_device("mps")
    tr_t, va_t, _ = load_corpus_splits(hf_chat="soda", hf_chat_limit=4000, builtin=True,
                                       repeats=60, seed=42, val_frac=0.1, test_frac=0.0)
    tok = CharTokenizer.from_text(tr_t)
    tr = torch.tensor(tok.encode(tr_t), dtype=torch.long)
    va = torch.tensor(tok.encode(va_t), dtype=torch.long)
    log(f"corpus {tr.numel():,} chars vocab {tok.vocab_size}")

    def batch(b=16, sl=48):
        ix = torch.randint(0, tr.numel()-sl-1, (b,))
        return torch.stack([tr[i:i+sl+1] for i in ix]).to(dev)

    @torch.no_grad()
    def ev(m, sl=48, b=16, mw=256):
        m.eval(); st = list(range(0, va.numel()-sl-1, sl))[:mw]; tot=n=0
        for i in range(0, len(st), b):
            c = torch.stack([va[s:s+sl+1] for s in st[i:i+b]]).to(dev)
            lg,_ = m(c[:,:-1])
            tot += float(torch.nn.functional.cross_entropy(
                lg.reshape(-1,m.vocab_size), c[:,1:].reshape(-1), reduction="sum").item())
            n += c[:,1:].numel()
        return (tot/n)/math.log(2)

    def run(label, grid, seed, k=16, width=None):
        cfg = LMConfig(grid_size=grid, embed_dim=64, inner_steps=3, seed=seed,
                       readout_width=width, brain_overrides={"k_max": k})
        m = BrainLanguageModel(tok.vocab_size, cfg, device=dev)
        opt = torch.optim.Adam(m.parameters(), lr=8e-4); t0=time.time(); m.train()
        for s in range(args.steps):
            l = m.loss_on(batch())
            if not math.isfinite(float(l.item())):
                return {"label":label,"seed":seed,"diverged_at":s}
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5); opt.step()
        r = {"label":label, "seed":seed, "grid":grid, "k_max":k,
             "readout":"fixed128" if width else "growing", "bpc":ev(m),
             "neurons":m.brain.num_neurons, "edges":m.brain.num_edges,
             "params":sum(q.numel() for q in m.parameters() if q.requires_grad),
             "min":round((time.time()-t0)/60,1)}
        log(f"  {label:<26} s{seed} bpc {r['bpc']:.4f}  ({r['min']:.0f} min)")
        return r

    rec = {"args": vars(args), "runs": []}
    def add(r):
        rec["runs"].append(r)
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump(rec, open(args.json, "w"), indent=2)

    seeds = [int(s) for s in args.seeds.split(",")]
    log(f"DENSITY CONTRAST — seeds {seeds}")
    for sd in seeds:
        add(run("volume grid16 k16", 16, sd, k=16))
        add(run("density grid12 k38", 12, sd, k=38))
    log(f"NEURON LADDER — seeds {seeds}, grids 8 and 16, both read-outs")
    for sd in seeds:
        for grid in (8, 16):
            for w in (None, 128):
                add(run(f"grid{grid} {'fixed' if w else 'growing'}", grid, sd, width=w))
    log(f"REPLICATION COMPLETE -> {args.json}")

if __name__ == "__main__":
    main()
