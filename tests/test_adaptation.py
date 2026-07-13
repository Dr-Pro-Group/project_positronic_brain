"""Tests for spike-frequency adaptation (I_M / I_AHP; --adaptation).

Adaptation is a slow, per-neuron hyperpolarizing current that low-pass-filters a
neuron's own firing rate and subtracts it from the membrane update, so a neuron
that has been firing progressively throttles itself. Off by default and
byte-identical to baseline when off.
"""

import torch

from positronic_brain.model import BrainConfig, PositronicBrain
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig


def test_adaptation_off_is_byte_identical_to_baseline():
    # The off==baseline contract is EXACT, not approximate: with the flag off the
    # adaptation state is never even allocated and step() skips the block, so the
    # dynamics must be bit-identical. Assert torch.equal, not allclose.
    b_off = PositronicBrain(BrainConfig(grid_size=6, seed=0, use_adaptation=False), device="cpu")
    b_def = PositronicBrain(BrainConfig(grid_size=6, seed=0), device="cpu")
    ext = torch.full((b_off.config.num_zones,), 2.0)
    r_off = torch.tensor(b_off.run_with_inputs(ext.numpy())["rates"])
    r_def = torch.tensor(b_def.run_with_inputs(ext.numpy())["rates"])
    assert torch.equal(r_off, r_def)


def test_adaptation_state_lifecycle():
    brain = PositronicBrain(BrainConfig(grid_size=5, seed=0, use_adaptation=True), device="cpu")
    assert brain._adapt_a is None
    brain.stp_begin(3)
    assert brain._adapt_a.shape == (3, brain.num_neurons)
    assert torch.count_nonzero(brain._adapt_a) == 0     # starts at rest (zeros)
    brain.stp_end()
    assert brain._adapt_a is None


def test_adaptation_throttles_sustained_firing():
    # The behavioural hallmark: under constant supra-threshold drive, adaptation
    # must LOWER the sustained firing rate relative to the non-adapting baseline
    # (the neuron inhibits itself as it keeps firing). Same seed/geometry/drive so
    # the ONLY difference is the adaptation current.
    cfg_on = BrainConfig(grid_size=6, seed=0, use_adaptation=True, adapt_gain=0.8, adapt_tau=15.0)
    cfg_off = BrainConfig(grid_size=6, seed=0, use_adaptation=False)
    b_on = PositronicBrain(cfg_on, device="cpu")
    b_off = PositronicBrain(cfg_off, device="cpu")
    V_on = torch.full((1, b_on.num_neurons), b_on.config.E_L)
    V_off = V_on.clone()
    drive = torch.full((1, b_on.num_neurons), 0.6)      # constant external drive
    b_on.stp_begin(1)
    b_off.stp_begin(1)
    for _ in range(40):
        V_on = b_on.step(V_on, drive)
        V_off = b_off.step(V_off, drive)
    r_on = float(b_on.firing_rate(V_on).mean().detach())
    r_off = float(b_off.firing_rate(V_off).mean().detach())
    b_on.stp_end()
    b_off.stp_end()
    assert r_on < r_off


def test_adaptation_changes_language_model_and_trains():
    text = "User: hi\nBrain: hello\n" * 30
    tok = CharTokenizer.from_text(text)
    cfg = LMConfig(grid_size=6, embed_dim=16, inner_steps=2,
                   brain_overrides={"use_adaptation": True})
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
    # Adaptation state must be cleared after each forward (no leak between calls).
    assert model.brain._adapt_a is None
