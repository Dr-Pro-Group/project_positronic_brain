"""
Where does this network sit in the space of recurrent dynamics, and what does that cost it?

Reservoir-computing theory gives recurrent networks a small set of standard,
comparable measurements, and this project has never taken them. Three are computed
here for any configuration:

  growth factor   the average one-step amplification of a small perturbation. Below
                  1 the network contracts and forgets; above 1 it expands and blows
                  up; useful computation lives near 1 -- the "edge of chaos". Recent
                  echo-state analysis puts the empirical optimum near 0.9, close
                  enough to criticality to hold a long history while still
                  satisfying the echo-state property.

  memory capacity Jaeger's MC: drive the network with i.i.d. noise, then ask how
                  well a linear read-out of the current state reconstructs the input
                  from tau steps ago, summed over tau. It is bounded above by the
                  number of units, and random sparse reservoirs typically reach
                  ~0.6N. Reporting MC/N says directly how much of the population is
                  being used as memory.

  participation   the fraction of units whose firing rate is meaningfully modulated
                  by the input. A unit that never moves contributes nothing and,
                  because it also feeds the recurrence, transmits nothing.

Two knobs dominate all three: the recurrent weight scale (which sets the loop gain)
and the input gain (which, per the same literature, buys higher-order capacity at
the cost of linear memory -- so a large input drive actively erases history).

    python experiments/dynamical_regime.py --sweep default
    python experiments/dynamical_regime.py --sweep gain --grid-size 12

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.model import BrainConfig, PositronicBrain
from positronic_brain.utils import get_device


@torch.no_grad()
def growth_factor(brain, steps_settle: int = 20, probes: int = 24, eps: float = 1e-3,
                 drive: float = 0.3, device="cpu") -> float:
    """Mean one-step amplification ||dV_{t+1}|| / ||dV_t|| at a driven operating point."""
    torch.manual_seed(0)
    V = torch.full((1, brain.num_neurons), brain.config.E_L, device=device)
    I = torch.randn((1, brain.num_neurons), device=device) * drive
    brain.stp_begin(1)
    for _ in range(steps_settle):
        V = brain.step(V, I)
    ratios = []
    for _ in range(probes):
        d = torch.randn_like(V)
        d = eps * d / d.norm()
        ratios.append(((brain.step(V + d, I) - brain.step(V, I)).norm() / d.norm()).item())
    brain.stp_end()
    return float(np.mean(ratios))


@torch.no_grad()
def memory_capacity(brain, n_steps: int = 1200, warmup: int = 200, max_delay: int = 30,
                    input_gain: float = 3.0, ridge: float = 1e-6, device="cpu") -> Dict:
    """Jaeger's linear memory capacity, MC = sum_tau corr^2(u_{t-tau}, readout(x_t)).

    The input is a single i.i.d. stream broadcast through a fixed random projection,
    standing in for the token current the language model injects. Read-outs are
    fitted by ridge regression on the collected states, which is the standard
    protocol; MC is bounded by the number of units.
    """
    N = brain.num_neurons
    g = torch.Generator(device="cpu").manual_seed(0)
    u = (torch.rand(n_steps, generator=g) * 2 - 1)                    # uniform[-1, 1]
    w_in = torch.randn(N, generator=g).to(device) * input_gain

    V = torch.full((1, N), brain.config.E_L, device=device)
    brain.stp_begin(1)
    states = []
    for t in range(n_steps):
        V = brain.step(V, (u[t].to(device) * w_in).unsqueeze(0))
        states.append(brain.firing_rate(V).squeeze(0).clone())
    brain.stp_end()

    X = torch.stack(states[warmup:]).double()                          # (T, N)
    X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=device)], 1)
    U = u[warmup:].double().to(device)

    # One ridge solve reused for every delay.
    XtX = X.T @ X + ridge * torch.eye(X.shape[1], dtype=torch.float64, device=device)
    try:
        chol = torch.linalg.cholesky(XtX)
    except Exception:
        return {"MC": float("nan"), "per_delay": [], "MC_over_N": float("nan")}

    per_delay = []
    for tau in range(1, max_delay + 1):
        if tau >= X.shape[0] - 10:
            break
        target = U[:-tau]
        feats = X[tau:]
        w = torch.cholesky_solve((feats.T @ target).unsqueeze(1), chol).squeeze(1)
        pred = feats @ w
        vt = torch.var(target)
        c = 0.0 if vt <= 0 else float(1.0 - torch.var(target - pred) / vt)
        per_delay.append(max(0.0, min(1.0, c)))
    return {"MC": float(sum(per_delay)), "per_delay": per_delay,
            "MC_over_N": float(sum(per_delay)) / N}


@torch.no_grad()
def participation(brain, n_steps: int = 300, input_gain: float = 3.0,
                  thresh: float = 1e-3, device="cpu") -> float:
    """Fraction of units whose firing rate varies meaningfully under random drive."""
    N = brain.num_neurons
    g = torch.Generator(device="cpu").manual_seed(1)
    u = (torch.rand(n_steps, generator=g) * 2 - 1)
    w_in = torch.randn(N, generator=g).to(device) * input_gain
    V = torch.full((1, N), brain.config.E_L, device=device)
    brain.stp_begin(1)
    rs = []
    for t in range(n_steps):
        V = brain.step(V, (u[t].to(device) * w_in).unsqueeze(0))
        rs.append(brain.firing_rate(V).squeeze(0))
    brain.stp_end()
    return float((torch.stack(rs).std(dim=0) > thresh).float().mean())


def probe(label: str, cfg: BrainConfig, input_gain: float, device) -> Dict:
    brain = PositronicBrain(cfg, device=device)
    gf = growth_factor(brain, device=device)
    mc = memory_capacity(brain, input_gain=input_gain, device=device)
    part = participation(brain, input_gain=input_gain, device=device)
    row = {"label": label, "neurons": brain.num_neurons, "g_max": cfg.g_max,
           "tau_m": cfg.tau_m, "input_gain": input_gain,
           "growth": gf, "MC": mc["MC"], "MC_over_N": mc["MC_over_N"],
           "participation": part, "per_delay": mc["per_delay"]}
    print(f"  {label:<34} growth {gf:6.3f}   MC {mc['MC']:7.2f} "
          f"({mc['MC_over_N']:5.1%} of N)   participating {part:5.1%}", flush=True)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--sweep", choices=["default", "gmax", "gain", "both"], default="both")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", default="runs/dynamical_regime.json")
    args = p.parse_args()

    device = get_device(args.device)
    base = BrainConfig(grid_size=args.grid_size)
    rows: List[Dict] = []

    print(f"grid {args.grid_size} ({args.grid_size**3} units); "
          f"defaults g_max={base.g_max} tau_m={base.tau_m}\n")
    print("Reference points: growth ~0.9 is the echo-state optimum; MC/N ~0.6 is what a")
    print("random sparse reservoir typically reaches.\n")

    print("baseline:")
    rows.append(probe("as shipped", BrainConfig(grid_size=args.grid_size), 3.0, device))

    if args.sweep in ("gmax", "both"):
        print("\nrecurrent weight scale (loop gain):")
        for m in (1, 2, 4, 8, 16, 32):
            rows.append(probe(f"g_max x{m}",
                              BrainConfig(grid_size=args.grid_size, g_max=base.g_max * m),
                              3.0, device))

    if args.sweep in ("gain", "both"):
        print("\ninput gain (theory: large gain suppresses linear memory):")
        for gain in (0.1, 0.3, 1.0, 3.0, 9.0):
            rows.append(probe(f"input_gain {gain}",
                              BrainConfig(grid_size=args.grid_size), gain, device))

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump({"args": vars(args), "rows": rows}, fh, indent=2)
    print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
