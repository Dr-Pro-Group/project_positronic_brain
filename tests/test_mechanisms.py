"""Tests for the Phase-2 fidelity mechanisms and numerical-stability guards.

These also serve as the regression suite the roadmap calls for: every off-by-
default mechanism must be byte-identical to baseline when off, finite when on, and
the network must stay finite across configurations.
"""

import math

import torch

from positronic_brain.model import BrainConfig, PositronicBrain
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig


def _rates(cfg_kwargs, drive=2.0):
    brain = PositronicBrain(BrainConfig(grid_size=6, seed=0, **cfg_kwargs), device="cpu")
    ext = torch.full((brain.config.num_zones,), drive)
    return torch.tensor(brain.run_with_inputs(ext.numpy())["rates"])


def test_oscillation_off_identical_on_changes():
    base = _rates({})
    off = _rates({"use_oscillation": False})
    on = _rates({"use_oscillation": True})
    assert torch.allclose(base, off)
    assert not torch.allclose(base, on)
    assert torch.isfinite(on).all()


def test_dendrites_off_identical_on_changes_and_finite():
    base = _rates({})
    off = _rates({"use_dendrites": False})
    on = _rates({"use_dendrites": True})
    assert torch.allclose(base, off)
    assert not torch.allclose(base, on)
    assert torch.isfinite(on).all()


def test_dendrites_gradients_flow():
    brain = PositronicBrain(BrainConfig(grid_size=6, seed=0, use_dendrites=True), device="cpu")
    out = brain(torch.ones(1, brain.config.num_zones))
    out.sum().backward()
    assert brain.edge_weight.grad is not None
    assert torch.isfinite(brain.edge_weight.grad).all()


def test_sparse_penalty_tracks_and_is_positive():
    text = "User: hi\nBrain: hello\n" * 10
    tok = CharTokenizer.from_text(text)
    model = BrainLanguageModel(tok.vocab_size, LMConfig(grid_size=6, embed_dim=16,
                                                        inner_steps=2), device="cpu")
    model.track_activity = True
    batch = torch.tensor(tok.encode(text))[:40].unsqueeze(0)
    model.loss_on(batch)
    pen = model.sparse_penalty(0.02)
    assert pen.item() >= 0.0
    assert model._last_activity is not None


def test_all_mechanisms_together_are_finite_and_train():
    # Stack every mechanism at once: the network must stay finite and trainable.
    text = "User: hi\nBrain: hello there\n" * 20
    tok = CharTokenizer.from_text(text)
    cfg = LMConfig(grid_size=6, embed_dim=16, inner_steps=2, brain_overrides={
        "use_divnorm": True, "use_stp": True, "use_homeostasis": True,
        "use_oscillation": True, "use_dendrites": True,
    })
    model = BrainLanguageModel(tok.vocab_size, cfg, device="cpu")
    data = torch.tensor(tok.encode(text))
    batch = data[:48].unsqueeze(0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = float(model.loss_on(batch).item())
    assert math.isfinite(first)
    for _ in range(20):
        loss = model.loss_on(batch)
        opt.zero_grad(); loss.backward(); opt.step()
        assert torch.isfinite(loss).all()
    last = float(model.loss_on(batch).item())
    assert math.isfinite(last) and last < first


def test_numerical_stability_grid12_finite():
    # A grid-12 brain under strong drive must produce finite logits (no NaN/inf).
    tok = CharTokenizer.from_text("abcdefghij " * 5)
    model = BrainLanguageModel(tok.vocab_size, LMConfig(grid_size=12, embed_dim=32,
                                                        inner_steps=3), device="cpu")
    batch = torch.tensor(tok.encode("abcdefghij " * 3))[:30].unsqueeze(0)
    logits, V = model(batch)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(V).all()
