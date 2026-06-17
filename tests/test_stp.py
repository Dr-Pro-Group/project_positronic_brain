"""Tests for short-term synaptic plasticity (Tsodyks-Markram)."""

import torch

from positronic_brain.model import BrainConfig, PositronicBrain
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig


def test_stp_off_is_identical_to_baseline():
    torch.manual_seed(0)
    b_off = PositronicBrain(BrainConfig(grid_size=6, seed=0, use_stp=False), device="cpu")
    b_def = PositronicBrain(BrainConfig(grid_size=6, seed=0), device="cpu")
    ext = torch.full((b_off.config.num_zones,), 2.0)
    r_off = torch.tensor(b_off.run_with_inputs(ext.numpy())["rates"])
    r_def = torch.tensor(b_def.run_with_inputs(ext.numpy())["rates"])
    assert torch.allclose(r_off, r_def)


def test_stp_state_lifecycle():
    brain = PositronicBrain(BrainConfig(grid_size=5, seed=0, use_stp=True), device="cpu")
    assert brain._stp_u is None
    brain.stp_begin(3)
    assert brain._stp_u.shape == (3, brain.num_edges)
    # rest values: u == U, x == 1
    assert torch.allclose(brain._stp_u, torch.full_like(brain._stp_u, brain.config.stp_U))
    assert torch.allclose(brain._stp_x, torch.ones_like(brain._stp_x))
    brain.stp_end()
    assert brain._stp_u is None


def test_stp_depresses_under_sustained_drive():
    # Sustained presynaptic activity should DEPLETE resources (x falls below 1):
    # the hallmark of short-term depression.
    brain = PositronicBrain(BrainConfig(grid_size=6, seed=0, use_stp=True), device="cpu")
    brain.stp_begin(1)
    V = torch.full((1, brain.num_neurons), 0.6)  # above threshold -> active
    I_ext = torch.zeros((1, brain.num_neurons))
    for _ in range(20):
        V = brain.step(V, I_ext)
    assert float(brain._stp_x.detach().mean()) < 1.0
    brain.stp_end()


def test_stp_changes_language_model_and_trains():
    text = "User: hi\nBrain: hello\n" * 30
    tok = CharTokenizer.from_text(text)
    cfg = LMConfig(grid_size=6, embed_dim=16, inner_steps=2, brain_overrides={"use_stp": True})
    model = BrainLanguageModel(tok.vocab_size, cfg, device="cpu")
    data = torch.tensor(tok.encode(text))
    batch = data[:60].unsqueeze(0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    first = float(model.loss_on(batch).item())
    for _ in range(30):
        loss = model.loss_on(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    last = float(model.loss_on(batch).item())
    assert last < first
    # STP state must be cleared after each forward (no leak between calls).
    assert model.brain._stp_u is None
