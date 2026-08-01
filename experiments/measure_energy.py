"""
Energy / sparsity measurement for the trained Positronic Brain language model.

This script computes *honest, hardware-agnostic proxies* for the energy argument
that the brain-inspired literature (SpikeGPT, SpikingBrain, etc.) rests on:

  1. CONNECTIVITY SPARSITY — how many synapses the 3D distance-biased graph
     actually has (E) versus a dense all-to-all layer (N^2). This is a real,
     architectural property of our model and the main source of its compute
     savings: synaptic operations scale O(E) = O(N*k), not O(N^2).

  2. ACTIVATION SPARSITY — the distribution of neuron firing rates during real
     text generation. Event-driven hardware only pays for neurons whose rate
     crosses a threshold, so we report the fraction of "active" neurons at
     several thresholds. (Our model uses continuous rates, not binary spikes, so
     this is an *upper-bound proxy* for what a spiking version could save.)

  3. EFFECTIVE SYNAPTIC OPERATIONS (SynOps) per generated character, compared to
     a dense ANN of the same neuron count. SynOps is the standard energy proxy
     in the SNN literature.

It does NOT claim Joules — that requires specific neuromorphic hardware. It
reports the *structural* and *activity* sparsity that determine energy on such
hardware, so the numbers are defensible and reproducible.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positronic_brain.language import BrainLanguageModel


def measure(lm_path: str, prompt: str, n_chars: int, device: str) -> dict:
    model, tok = BrainLanguageModel.load(lm_path, device=device)
    if tok is None:
        raise SystemExit("checkpoint has no tokenizer; cannot generate")
    model.eval()
    brain = model.brain

    N = int(brain.num_neurons)
    E = int(brain.edge_index.shape[1])
    inner = int(model.config.inner_steps)

    # 1. Connectivity sparsity ------------------------------------------------
    dense = N * N
    conn_density = E / dense
    fan_in = E / N

    # 2 & 3. Activation sparsity + SynOps during real generation --------------
    rates_collected: list[np.ndarray] = []
    active_synops = 0.0          # edges whose presyn neuron is "active"
    total_steps = 0

    src = brain.edge_index[0]    # (E,) presynaptic neuron index per edge

    V = model.init_state(1)
    ids = tok.encode(prompt) or [0]
    with torch.no_grad():
        # warm up on the prompt
        for tid in ids[:-1]:
            V, _ = model.step_token(V, torch.tensor([tid], device=model.device))
        last = ids[-1]
        for _ in range(n_chars):
            # replicate step_token but capture per-inner-step firing rates
            I_ext = model._token_current(torch.tensor([last], device=model.device))
            for _ in range(inner):
                V = brain.step(V, I_ext)
                r = brain.firing_rate(V)[0]                  # (N,)
                rates_collected.append(r.detach().cpu().numpy())
                # SynOps with an event threshold of 0.5 (presyn "fires")
                active_mask = (r > 0.5)
                active_synops += float(active_mask[src].sum().item())
                total_steps += 1
            logits = model.head(brain.firing_rate(V))[0]
            last = int(torch.argmax(logits).item())

    R = np.stack(rates_collected, axis=0)        # (steps, N)
    mean_rate = float(R.mean())
    # activation sparsity at several thresholds
    act = {f"frac_active>{thr}": float((R > thr).mean()) for thr in (0.1, 0.3, 0.5, 0.7)}

    # SynOps comparison: dense ANN does N*N MACs per step; we do E,
    # and an event-driven version only does active_synops.
    dense_synops_per_step = float(N * N)
    sparse_synops_per_step = float(E)
    event_synops_per_step = active_synops / max(total_steps, 1)

    return {
        "neurons_N": N,
        "synapses_E": E,
        "fan_in_avg": fan_in,
        "dense_NxN": dense,
        "connectivity_density": conn_density,
        "connectivity_savings_x": dense / E,
        "mean_firing_rate": mean_rate,
        **act,
        "dense_synops_per_step": dense_synops_per_step,
        "sparse_synops_per_step": sparse_synops_per_step,
        "event_synops_per_step": event_synops_per_step,
        "synops_savings_vs_dense_x": dense_synops_per_step / max(event_synops_per_step, 1.0),
        "inner_steps": inner,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lm-path", default="trained_models/brain_lm.pt")
    ap.add_argument("--prompt", default="User: hello\nBrain:")
    ap.add_argument("--n-chars", type=int, default=120)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    m = measure(args.lm_path, args.prompt, args.n_chars, args.device)

    print("\n=== Positronic Brain — energy / sparsity report ===")
    print(f"neurons (N)                 : {m['neurons_N']:,}")
    print(f"synapses (E)                : {m['synapses_E']:,}")
    print(f"avg fan-in (E/N)            : {m['fan_in_avg']:.1f}")
    print(f"dense equivalent (N*N)      : {m['dense_NxN']:,}")
    print(f"connectivity density (E/N^2): {m['connectivity_density']:.3e}")
    print(f"  -> connectivity savings   : {m['connectivity_savings_x']:.0f}x fewer synapses than dense")
    print("---")
    print(f"mean firing rate            : {m['mean_firing_rate']:.3f}")
    for k in ("frac_active>0.1", "frac_active>0.3", "frac_active>0.5", "frac_active>0.7"):
        print(f"{k:28s}: {m[k]:.3f}")
    print("---")
    print(f"dense SynOps / step         : {m['dense_synops_per_step']:,.0f}")
    print(f"sparse SynOps / step (E)    : {m['sparse_synops_per_step']:,.0f}")
    print(f"event SynOps / step (active): {m['event_synops_per_step']:,.0f}")
    print(f"  -> SynOps savings vs dense: {m['synops_savings_vs_dense_x']:.0f}x")
    print("===================================================\n")


if __name__ == "__main__":
    main()
