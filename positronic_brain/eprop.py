"""
e-prop: a forward-only, biologically-plausible alternative to BPTT.

Backpropagation-through-time is the *least* biological part of the Positronic
Brain pipeline: it requires storing the whole unrolled activation history and
propagating gradients backward through time, which no synapse can do. **e-prop**
(Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass, *Nat. Commun.* 2020)
replaces the backward-through-time pass with two biologically-local factors:

* an **eligibility trace** ``e_ij`` per synapse — a quantity each synapse can
  compute *forward in time* from its own pre- and post-synaptic activity, which
  remembers how that synapse has recently influenced the post-synaptic neuron; and
* a **learning signal** ``L_i`` per neuron — broadcast from the readout (here the
  language head), the local error projected back through the output weights only
  (no propagation through time).

The weight update is the product ``Δw_ij ∝ Σ_t L_i^t · e_ij^t`` — exactly the
neoHebbian three-factor form (pre × post × top-down signal). For the conductance
rate neuron used here the eligibility must include the **driving force**
``(E_rev − V_i)`` (the conductance term), without which the trace does not track
the true sensitivity — so we keep it explicitly.

This is an *approximation* of BPTT (it drops the multi-inner-step and cross-time
second-order terms), so it is validated by a gradient-agreement test: the e-prop
estimate of ``∂loss/∂edge_weight`` must point in the same direction as the true
BPTT gradient (positive cosine) on a small brain. It is offered as a fidelity /
online-learning path, not a quality booster.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def eprop_edge_gradient(model, tokens: torch.Tensor) -> torch.Tensor:
    """Forward-only e-prop estimate of ``∂loss/∂edge_weight`` (shape ``(E,)``).

    ``loss`` is the same next-character cross-entropy as :meth:`BrainLanguageModel.loss_on`.
    No autograd graph is built: every quantity is accumulated forward in time.
    """
    brain = model.brain
    cfg = brain.config
    device = model.device
    tokens = tokens.to(device)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    B, T = tokens.shape
    src, dst = brain.edge_index[0], brain.edge_index[1]
    E = brain.num_edges

    rho = 1.0 - brain.alpha                 # membrane retention per step
    sign = brain.edge_sign                  # (E,) fixed Dale sign
    w_eff = brain.signed_weights()          # (E,)
    head_W = model.head.weight              # (vocab, N)

    elig = torch.zeros((B, E), device=device)   # filtered, driving-force-weighted presyn
    grad = torch.zeros((E,), device=device)
    V = model.init_state(B)
    norm = 1.0 / (B * max(T - 1, 1))            # match cross_entropy mean reduction

    for t in range(T - 1):
        I_ext = model._token_current(tokens[:, t])
        # Reverberate exactly like the model (inner_steps), keeping the final r/V.
        V = brain.integrate(V, I_ext, model.config.inner_steps)
        r = brain.firing_rate(V)                          # (B, N)
        r_pre = r[:, src]                                 # (B, E)
        # Driving force per edge depends on the presynaptic neuron's reversal
        # potential (excitatory vs inhibitory), i.e. the conductance term.
        E_rev = torch.where(sign > 0, cfg.E_E, cfg.E_I)   # (E,)
        drive = (E_rev.unsqueeze(0) - V[:, dst])          # (B, E)
        # Eligibility trace: low-pass of the (driving-force-weighted) presyn drive.
        elig = rho * elig + drive * r_pre                 # (B, E)
        # Post-synaptic pseudo-derivative dr_i/dV_i = gamma * r_i * (1 - r_i).
        psi = cfg.gamma * r * (1.0 - r)                   # (B, N)
        psi_dst = psi[:, dst]                             # (B, E)

        # Learning signal: project the output error back through the head weights
        # (symmetric feedback). d = softmax(logits) - onehot(target).
        logits = model.head(r)                            # (B, vocab)
        probs = F.softmax(logits, dim=-1)
        target = tokens[:, t + 1]
        d = probs
        d.scatter_add_(1, target.unsqueeze(1), -torch.ones_like(target, dtype=d.dtype).unsqueeze(1))
        L = d @ head_W                                    # (B, N) learning signal per neuron
        L_dst = L[:, dst]                                 # (B, E)

        # Three-factor accumulation; sign maps d/dw_eff -> d/d|edge_weight|.
        grad += sign * (L_dst * psi_dst * elig).sum(0) * norm

    return grad


def eprop_step(model, tokens: torch.Tensor, optimizer, grad_clip: float = 0.5) -> float:
    """One e-prop optimisation step.

    The recurrent core (``edge_weight``) is updated by the forward-only e-prop
    estimate; the read-in/read-out/embedding (``token_in``/``head``/``embed``)
    receive their *exact* local gradients via a per-character autograd pass whose
    membrane state is detached between characters (so no gradient flows through
    time — keeping the whole step local in time). Returns the mean CE loss.
    """
    brain = model.brain
    device = model.device
    tokens = tokens.to(device)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    B, T = tokens.shape

    # e-prop gradient for the recurrent synapses.
    edge_grad = eprop_edge_gradient(model, tokens)

    # Exact local (in-time) gradients for the feed-forward read-in/read-out.
    optimizer.zero_grad()
    model.brain.stp_begin(B)
    V = model.init_state(B).detach()
    total, n = 0.0, 0
    for t in range(T - 1):
        I_ext = model._token_current(tokens[:, t])
        V = brain.integrate(V, I_ext, model.config.inner_steps)
        logits = model.head(brain.firing_rate(V))
        loss = F.cross_entropy(logits, tokens[:, t + 1])
        loss.backward()
        V = V.detach()
        total += float(loss.item()); n += 1
    model.brain.stp_end()

    # Inject the e-prop estimate as edge_weight's gradient and update everything.
    if brain.edge_weight.grad is None:
        brain.edge_weight.grad = edge_grad.clone()
    else:
        brain.edge_weight.grad += edge_grad
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return total / max(n, 1)
