"""Tests for the laminar microcircuit (canonical layer motif + even inhibition)."""

import numpy as np
import torch

from positronic_brain.model import BrainConfig, PositronicBrain
from positronic_brain.connectivity import laminar_bands, canonical_laminar_motif
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig


def test_laminar_bands_partition_by_depth():
    b = laminar_bands(6, bands=3)
    assert b.min() == 0 and b.max() == 2
    assert set(np.unique(b)) == {0, 1, 2}


def test_motif_shape_and_feedforward_bias():
    M = canonical_laminar_motif(3)
    assert M.shape == (3, 3)
    # L4 (band 1) -> L2/3 (band 0) should be the strongest feedforward entry.
    assert M[1, 0] == M.max()


def test_laminar_off_identical_on_changes_topology():
    base = PositronicBrain(BrainConfig(grid_size=6, seed=0), device="cpu")
    off = PositronicBrain(BrainConfig(grid_size=6, seed=0, use_laminar=False), device="cpu")
    on = PositronicBrain(BrainConfig(grid_size=6, seed=0, use_laminar=True), device="cpu")
    assert torch.equal(base.edge_index, off.edge_index)          # off == baseline graph
    # Laminar reshapes which local neighbours connect (same edge count, k_max).
    assert on.edge_index.shape[1] == base.edge_index.shape[1]
    assert not torch.equal(on.edge_index, base.edge_index)
    assert hasattr(on, "laminar_band")


def test_laminar_inhibition_is_more_even_than_random():
    # Every neuron should have at least one inhibitory presynaptic neighbour under
    # the laminar even-placement, fixing the zero-inhibition gaps random placement
    # leaves. Measure the fraction of targets receiving any inhibitory input.
    def zero_inhib_fraction(use_laminar):
        b = PositronicBrain(BrainConfig(grid_size=8, seed=0, use_laminar=use_laminar),
                            device="cpu")
        src, dst = b.edge_index
        inh_src = b.is_inhibitory[src]
        N = b.num_neurons
        got_inhib = torch.zeros(N, dtype=torch.bool)
        got_inhib[dst[inh_src]] = True
        return float((~got_inhib).float().mean())
    assert zero_inhib_fraction(True) <= zero_inhib_fraction(False)


def test_laminar_trains_and_is_finite():
    text = "User: hi\nBrain: hello there\n" * 20
    tok = CharTokenizer.from_text(text)
    cfg = LMConfig(grid_size=6, embed_dim=16, inner_steps=2,
                   brain_overrides={"use_laminar": True})
    model = BrainLanguageModel(tok.vocab_size, cfg, device="cpu")
    data = torch.tensor(tok.encode(text))
    batch = data[:48].unsqueeze(0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = float(model.loss_on(batch).item())
    for _ in range(20):
        loss = model.loss_on(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    last = float(model.loss_on(batch).item())
    assert np.isfinite(last) and last < first
