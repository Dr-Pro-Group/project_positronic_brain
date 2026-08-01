"""
Screen many organizations cheaply, then train only the ones that earn it.

Phase A established that the recurrent core is close to a passthrough: it expands
the dimensionality of its input by about 1.1x, holds roughly 3% of the memory
capacity a random sparse reservoir of the same size would, and does not improve on
either count as neurons are added. If that is the binding constraint, then no
amount of training fixes an organization that cannot expand, and training every
candidate to find out would waste hours per candidate.

So this screens first. Each configuration is measured on three quantities that take
seconds rather than hours:

  expansion   effective dimensionality of the recurrent state divided by the rank of
              the input driving it. This is what a reservoir is FOR; ~1.0 means the
              recurrence adds nothing.
  MC          Jaeger's linear memory capacity, as a fraction of the unit count.
  growth      one-step perturbation gain, for context on the dynamical regime.

Only configurations that move `expansion` are worth the training budget. The point
of the screen is to make that judgement on evidence instead of taste.

    python experiments/screen_organization.py --grid-size 12
    python experiments/screen_organization.py --grid-size 16 --json runs/screen16.json

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.model import BrainConfig, PositronicBrain


@torch.no_grad()
def _drive(brain, n: int, rank: int, gain: float, seed: int = 2):
    N = brain.num_neurons
    g = torch.Generator().manual_seed(seed)
    U = torch.rand(n, rank, generator=g) * 2 - 1
    W = torch.randn(rank, N, generator=g) * gain / np.sqrt(rank)
    V = torch.full((1, N), brain.config.E_L)
    brain.stp_begin(1)
    rs = []
    for t in range(n):
        V = brain.step(V, (U[t] @ W).unsqueeze(0))
        rs.append(brain.firing_rate(V).squeeze(0))
    brain.stp_end()
    return U, torch.stack(rs)


def expansion(brain, rank: int = 64, n: int = 400, gain: float = 3.0) -> float:
    """Effective state dimensionality per unit of input dimensionality."""
    _, R = _drive(brain, n, rank, gain)
    if not torch.isfinite(R).all():
        return float("nan")
    X = R[100:].double()
    X = X - X.mean(0)
    ev = torch.linalg.svdvals(X) ** 2
    if float(ev.sum()) <= 0:
        return float("nan")
    return float(ev.sum() ** 2 / (ev ** 2).sum()) / rank


def memory_capacity(brain, n: int = 4000, warmup: int = 200, max_delay: int = 30,
                    gain: float = 3.0, ridge: float = 1e-3, proj: int = 256) -> float:
    """Jaeger's linear memory capacity, measured on HELD-OUT data.

    Two details decide whether this number means anything. The read-out has one
    weight per unit, so with more units than timesteps it can reconstruct any target
    at all -- including a delayed input the network never retained -- and the score
    saturates at max_delay regardless of the dynamics. Fitting on one half and
    scoring on the other exposes that immediately; so does projecting the state onto
    a fixed random subspace so features stay well below samples. Both are applied
    here, and an earlier version of this screen had neither, which made every
    configuration look identical.
    """
    U, R = _drive(brain, n, 1, gain, seed=3)
    if not torch.isfinite(R).all():
        return float("nan")
    X = R[warmup:].double()
    if X.shape[1] > proj:                       # fixed random projection, seeded
        g = torch.Generator().manual_seed(7)
        P = torch.randn(X.shape[1], proj, generator=g).double() / np.sqrt(X.shape[1])
        X = X @ P
    X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
    u = U[warmup:, 0].double()

    total = 0.0
    for tau in range(1, max_delay + 1):
        if tau >= X.shape[0] // 2:
            break
        target, feats = u[:-tau], X[tau:]
        half = feats.shape[0] // 2
        ftr, fte = feats[:half], feats[half:]
        ttr, tte = target[:half], target[half:]
        A = ftr.T @ ftr + ridge * torch.eye(ftr.shape[1], dtype=torch.float64)
        try:
            w = torch.linalg.solve(A, ftr.T @ ttr)
        except Exception:
            continue
        vt = torch.var(tte)
        if vt > 0:                              # held-out r^2, floored at 0
            total += max(0.0, min(1.0, float(1 - torch.var(tte - fte @ w) / vt)))
    return total


@torch.no_grad()
def growth(brain, settle: int = 20, probes: int = 16, eps: float = 1e-3) -> float:
    torch.manual_seed(0)
    V = torch.full((1, brain.num_neurons), brain.config.E_L)
    I = torch.randn((1, brain.num_neurons)) * 0.3
    brain.stp_begin(1)
    for _ in range(settle):
        V = brain.step(V, I)
    if not torch.isfinite(V).all():
        brain.stp_end()
        return float("inf")
    rs = []
    for _ in range(probes):
        d = torch.randn_like(V)
        d = eps * d / d.norm()
        rs.append(((brain.step(V + d, I) - brain.step(V, I)).norm() / d.norm()).item())
    brain.stp_end()
    return float(np.mean(rs))


def candidates(grid: int) -> List[Dict]:
    """The hypotheses, each changing ONE thing against the shipped baseline where possible."""
    c: List[Dict] = [{"label": "baseline (as shipped)", "kw": {}}]

    # Diverse timescales — the direct prediction from Phase A: a population needs a
    # RANGE of integration windows, not one long one.
    for s in (0.2, 0.4, 0.6, 0.8, 1.0, 1.3):
        c.append({"label": f"tau spread {s}", "kw": {"tau_m_spread": s}})

    # Locality: how far a synapse may reach. A near-lattice mixes slowly; longer
    # reach shortens path length at the same edge count.
    for r in (1.5, 2.6, 4.0, 6.0, 9.0):
        c.append({"label": f"conn radius {r}", "kw": {"connection_radius": r}})

    # Fan-in: more inputs per neuron at the same population size.
    for k in (4, 8, 16, 32, 64):
        c.append({"label": f"k_max {k}", "kw": {"k_max": k}})

    # Wiring-distance bias: how strongly distance shapes which synapses exist.
    for s in (0.75, 1.75, 4.0, 12.0):
        c.append({"label": f"decay sigma {s}", "kw": {"decay_sigma": s}})

    # Topology and the biological constraints themselves.
    c += [
        {"label": "random wiring", "kw": {"spatial_wiring": False}},
        {"label": "no Dale", "kw": {"use_dale": False}},
        {"label": "no conductance", "kw": {"use_conductance": False}},
        {"label": "E/I 50-50", "kw": {"frac_inhibitory": 0.5}},
        {"label": "inh_scale 1 (no I boost)", "kw": {"inh_scale": 1.0}},
        {"label": "laminar", "kw": {"use_laminar": True}},
        {"label": "divnorm", "kw": {"use_divnorm": True}},
        {"label": "stp", "kw": {"use_stp": True}},
        {"label": "adaptation", "kw": {"use_adaptation": True}},
        {"label": "delays", "kw": {"use_delays": True}},
        {"label": "oscillation", "kw": {"use_oscillation": True}},
    ]

    # Combinations of whatever the single-factor rows suggest should compose.
    c += [
        {"label": "spread 0.8 + radius 4", "kw": {"tau_m_spread": 0.8, "connection_radius": 4.0}},
        {"label": "spread 0.8 + k_max 32", "kw": {"tau_m_spread": 0.8, "k_max": 32}},
        {"label": "spread 0.8 + stp", "kw": {"tau_m_spread": 0.8, "use_stp": True}},
        {"label": "spread 0.8 + adaptation", "kw": {"tau_m_spread": 0.8, "use_adaptation": True}},
        {"label": "spread 0.8 + delays", "kw": {"tau_m_spread": 0.8, "use_delays": True}},
        {"label": "spread 1.0 + radius 4 + k32",
         "kw": {"tau_m_spread": 1.0, "connection_radius": 4.0, "k_max": 32}},
    ]
    return c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--rank", type=int, default=64,
                   help="input rank; the LM injects a learned embedding of width 64")
    p.add_argument("--json", default="runs/screen_organization.json")
    args = p.parse_args()

    cands = candidates(args.grid_size)
    print(f"screening {len(cands)} organizations at grid {args.grid_size} "
          f"({args.grid_size ** 3} units)\n")
    print("expansion ~1.0 means the recurrence adds nothing; a reservoir should exceed it.")
    print("MC/N reference for a random sparse reservoir is ~0.6.\n")
    print(f"{'configuration':<30}{'growth':>9}{'MC':>8}{'MC/N':>8}{'expansion':>11}")
    print("-" * 66)

    rows = []
    for cand in cands:
        cfg = BrainConfig(grid_size=args.grid_size, **cand["kw"])
        try:
            b = PositronicBrain(cfg, device="cpu")
            g, mc, ex = growth(b), memory_capacity(b), expansion(b, rank=args.rank)
        except Exception as exc:                       # a config that cannot even build
            print(f"{cand['label']:<30}   failed: {type(exc).__name__}")
            rows.append({**cand, "error": f"{type(exc).__name__}: {exc}"})
            continue
        n = b.num_neurons
        rows.append({"label": cand["label"], "kw": cand["kw"], "neurons": n,
                     "growth": g, "MC": mc, "MC_over_N": mc / n, "expansion": ex})
        fmt = lambda v, w, d=2: (" " * (w - 3) + "nan") if v != v else f"{v:>{w}.{d}f}"
        print(f"{cand['label']:<30}{fmt(g,9,3)}{fmt(mc,8,1)}"
              f"{'    nan' if mc != mc else f'{mc/n:>7.1%}'}{fmt(ex,11,3)}", flush=True)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump({"args": vars(args), "rows": rows}, fh, indent=2)

    ok = [r for r in rows if "error" not in r and r["expansion"] == r["expansion"]]
    base = next((r for r in ok if r["label"].startswith("baseline")), None)
    if base and ok:
        ok.sort(key=lambda r: -r["expansion"])
        print("\n" + "=" * 66)
        print(f"baseline expansion {base['expansion']:.3f}, MC/N {base['MC_over_N']:.1%}")
        print("best by dimensional expansion:")
        for r in ok[:8]:
            print(f"  {r['label']:<32} expansion {r['expansion']:.3f} "
                  f"({r['expansion']/base['expansion']:.2f}x baseline)  MC/N {r['MC_over_N']:.1%}")
        print("=" * 66)
    print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
