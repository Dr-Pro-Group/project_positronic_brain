"""
Diagnostics that test the Positronic Brain language model's own claims.

These are deliberately small, dependency-free probes used by the trainer and the
tests to keep the project's narrative honest about what the recurrent dynamics
actually compute:

* :func:`saturation_fraction` — what fraction of neurons sit in the flat tails of
  the sigmoid firing-rate (zero-gradient, effectively dead at that step).
* :func:`settling_residual` — how much the membrane potential is still moving at
  the *last* reverberation step. Near-zero ⇒ the state has settled (an attractor
  picture); large ⇒ ``inner_steps`` is just a short transient filter.
* :func:`memory_horizon` — how many characters a single-character perturbation
  survives in the membrane state. This is the empirical answer to the "the living
  state *is* the context" claim.
* :func:`grad_norms_by_group` — per-parameter-group gradient norms, to reveal
  whether the recurrent core (``edge_weight``) is actually learning or whether the
  Linear read-in/read-out is doing all the work.
"""

from __future__ import annotations

from typing import Dict, List

import torch


@torch.no_grad()
def saturation_fraction(rates: torch.Tensor, lo: float = 0.01, hi: float = 0.99) -> float:
    """Fraction of firing rates in the flat (near-0 / near-1) sigmoid tails."""
    sat = (rates < lo) | (rates > hi)
    return float(sat.float().mean().item())


@torch.no_grad()
def settling_residual(model, token_id: int) -> Dict[str, float]:
    """Mean |ΔV| at the last reverberation step for one injected character.

    Returns the absolute last-step change and its size relative to the mean |V|,
    so a value near 0 means the dynamics have settled and a value near 1 means the
    final step is still moving the state as much as the state itself.
    """
    device = model.device
    V = model.init_state(1)
    I_ext = model._token_current(torch.tensor([int(token_id)], device=device))
    V_prev = V
    for _ in range(model.config.inner_steps):
        V_prev = V
        V = model.brain.step(V, I_ext)
    dv = (V - V_prev).abs().mean().item()
    scale = V.abs().mean().item() + 1e-8
    return {"last_step_dV": dv, "relative": dv / scale}


@torch.no_grad()
def memory_horizon(model, token_ids: List[int], perturb_id: int) -> Dict[str, float]:
    """Half-life (in characters) of a single-character perturbation in V.

    Two copies of the brain are driven by the same sequence except for the first
    character; we track ‖V_a − V_b‖ as the sequence proceeds and report the
    position at which that divergence has fallen to half of its initial value. A
    small number quantifies how shallow the "membrane state carries context"
    memory really is.
    """
    device = model.device
    if len(token_ids) < 2:
        return {"half_life_chars": 0.0, "n": float(len(token_ids))}
    Va = model.init_state(1)
    Vb = model.init_state(1)
    first_a = int(token_ids[0])
    first_b = int(perturb_id if perturb_id != first_a else (first_a + 1) % model.vocab_size)
    Va, _ = model.step_token(Va, torch.tensor([first_a], device=device))
    Vb, _ = model.step_token(Vb, torch.tensor([first_b], device=device))
    div0 = (Va - Vb).norm().item() + 1e-8
    half_life = float(len(token_ids))
    found = False
    for k, tid in enumerate(token_ids[1:], start=1):
        t = torch.tensor([int(tid)], device=device)
        Va, _ = model.step_token(Va, t)
        Vb, _ = model.step_token(Vb, t)
        div = (Va - Vb).norm().item()
        if not found and div <= 0.5 * div0:
            half_life = float(k)
            found = True
    return {"half_life_chars": half_life, "initial_divergence": div0}


@torch.no_grad()
def grad_norms_by_group(model) -> Dict[str, float]:
    """L2 norm of the accumulated gradient for each named parameter group.

    Call right after ``loss.backward()`` (before ``opt.step()``). Groups the
    recurrent core (``brain.edge_weight`` / ``brain.neuron_bias``) separately from
    the language read-in/read-out so a hollow recurrent core is visible.
    """
    groups = {
        "head": [], "token_in": [], "embed": [],
        "edge_weight": [], "neuron_bias": [], "other": [],
    }
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().norm().item())
        if "head" in name:
            groups["head"].append(g)
        elif "token_in" in name:
            groups["token_in"].append(g)
        elif "embed" in name:
            groups["embed"].append(g)
        elif "edge_weight" in name:
            groups["edge_weight"].append(g)
        elif "neuron_bias" in name:
            groups["neuron_bias"].append(g)
        else:
            groups["other"].append(g)
    return {k: (sum(x * x for x in v) ** 0.5 if v else 0.0) for k, v in groups.items()}
