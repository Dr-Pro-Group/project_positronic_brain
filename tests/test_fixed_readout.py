"""Tests for the fixed-width read-out (LMConfig.readout_width).

The default read-out is ``Linear(N, vocab)``, whose parameter count grows with the
brain, so a scaling curve conflates "more neurons" with "a bigger read-out". Setting
``readout_width`` inserts a frozen random projection in front of the head, holding
the trainable read-out fixed at every brain size. It is a measurement control, so
what matters is that the head really does stop growing and the projection really is
never trained.
"""

import torch

from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig


def _head_params(model) -> int:
    return sum(p.numel() for p in model.head.parameters())


def test_default_readout_grows_with_the_brain():
    small = BrainLanguageModel(50, LMConfig(grid_size=6), device="cpu")
    large = BrainLanguageModel(50, LMConfig(grid_size=10), device="cpu")
    assert _head_params(large) > _head_params(small)
    assert small._readout_proj is None
    assert small.head.in_features == small.brain.num_neurons


def test_fixed_readout_is_constant_across_brain_sizes():
    sizes = {g: BrainLanguageModel(50, LMConfig(grid_size=g, readout_width=64),
                                   device="cpu")
             for g in (6, 8, 10)}
    counts = {g: _head_params(m) for g, m in sizes.items()}
    assert len(set(counts.values())) == 1, counts
    for m in sizes.values():
        assert m.head.in_features == 64


def test_projection_is_frozen_and_covers_every_neuron():
    m = BrainLanguageModel(50, LMConfig(grid_size=6, readout_width=32), device="cpu")
    proj = m._readout_proj
    assert proj.shape == (m.brain.num_neurons, 32)
    # A buffer, not a Parameter: saved with the model but never optimised.
    assert not any(p is proj for p in m.parameters())
    assert not proj.requires_grad
    # Every neuron projects somewhere, so the head still sees the whole population.
    assert bool((proj.abs().sum(dim=1) > 0).all())


def test_fixed_readout_still_trains():
    text = "User: hi\nBrain: hello\n" * 30
    tok = CharTokenizer.from_text(text)
    m = BrainLanguageModel(tok.vocab_size,
                           LMConfig(grid_size=6, embed_dim=16, inner_steps=2,
                                    readout_width=48),
                           device="cpu")
    batch = torch.tensor(tok.encode(text))[:60].unsqueeze(0)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    first = float(m.loss_on(batch).item())
    for _ in range(30):
        loss = m.loss_on(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(m.loss_on(batch).item()) < first

    # The frozen projection must be untouched by training.
    assert m._readout_proj.grad is None


def test_projection_survives_a_save_load_round_trip(tmp_path):
    cfg = LMConfig(grid_size=6, embed_dim=16, readout_width=32, seed=3)
    m = BrainLanguageModel(40, cfg, device="cpu")
    path = tmp_path / "fixed_readout.pt"
    torch.save(m.state_dict(), path)

    clone = BrainLanguageModel(40, cfg, device="cpu")
    clone.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(clone._readout_proj, m._readout_proj)
