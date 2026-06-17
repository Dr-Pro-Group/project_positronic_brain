"""Gradient-agreement test gating the e-prop local learning rule.

e-prop is an approximation of BPTT, so its only hard requirement is that the
forward-only estimate of d(loss)/d(edge_weight) points in the SAME direction as
the true BPTT gradient. We assert a positive cosine similarity on a small brain.
"""

import torch

from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig
from positronic_brain.eprop import eprop_edge_gradient, eprop_step


def _setup():
    torch.manual_seed(0)
    text = "User: hi\nBrain: hello there friend\n" * 20
    tok = CharTokenizer.from_text(text)
    cfg = LMConfig(grid_size=4, embed_dim=16, inner_steps=2)
    model = BrainLanguageModel(tok.vocab_size, cfg, device="cpu")
    data = torch.tensor(tok.encode(text))
    batch = data[:48].unsqueeze(0)
    return model, batch


def test_eprop_gradient_agrees_with_bptt():
    model, batch = _setup()

    # True BPTT gradient for edge_weight.
    model.zero_grad()
    loss = model.loss_on(batch)
    loss.backward()
    bptt = model.brain.edge_weight.grad.detach().clone()

    # Forward-only e-prop estimate.
    eprop = eprop_edge_gradient(model, batch)

    cos = torch.nn.functional.cosine_similarity(
        bptt.flatten(), eprop.flatten(), dim=0).item()
    # e-prop drops higher-order/multi-step terms, so we only require clearly
    # positive alignment (the gate the roadmap specifies).
    assert cos > 0.2, f"e-prop/BPTT cosine too low: {cos:.3f}"


def test_eprop_step_reduces_loss():
    model, batch = _setup()
    opt = torch.optim.Adam([p for p in model.parameters()], lr=5e-3)
    first = float(model.loss_on(batch).item())
    for _ in range(25):
        eprop_step(model, batch, opt)
    last = float(model.loss_on(batch).item())
    assert last < first
