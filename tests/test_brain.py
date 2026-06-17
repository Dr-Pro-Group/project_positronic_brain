"""
Pytest suite for the Positronic Brain v3.

Covers: zone assignment, sparse graph structure, Dale's law, model shapes and
determinism, resting-state behaviour, learnability, save/load round-trip, and
scaling to a larger grid.

Run from the project root with the venv active:

    source .venv/bin/activate
    pytest -q
"""

from __future__ import annotations

import numpy as np
import torch

from positronic_brain import (
    BrainConfig,
    PositronicBrain,
    assign_zones,
    build_sparse_graph,
    init_edge_weights,
)
from positronic_brain.zones import DEFAULT_ZONES, num_zones


# --------------------------------------------------------------------- zones
def test_zone_assignment_covers_all_neurons():
    G = 5
    zones = assign_zones(G)
    assert zones.shape == (G ** 3,)
    assert zones.min() >= 0
    assert zones.max() < num_zones()
    # Every zone should claim at least one neuron at this grid size.
    assert len(np.unique(zones)) == num_zones()


def test_zone_assignment_deterministic():
    a = assign_zones(4)
    b = assign_zones(4)
    assert np.array_equal(a, b)


# -------------------------------------------------------------- connectivity
def test_sparse_graph_shapes_and_radius():
    edge_index, edge_dist, pos = build_sparse_graph(grid_size=4, k_max=16, seed=0)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_dist.shape[0]
    assert pos.shape == (64, 3)
    # No self-connections by default.
    assert not np.any(edge_index[0] == edge_index[1])
    # All edges within the connection radius.
    assert edge_dist.max() <= 2.6 + 1e-5


def test_fan_in_cap_respected():
    k_max = 8
    edge_index, _, _ = build_sparse_graph(grid_size=5, k_max=k_max, seed=1)
    dst = edge_index[1]
    counts = np.bincount(dst, minlength=125)
    assert counts.max() <= k_max


def test_dale_law_sign_per_source():
    edge_index, edge_dist, pos = build_sparse_graph(grid_size=4, seed=2)
    N = pos.shape[0]
    w, is_inh = init_edge_weights(edge_index, edge_dist, N, frac_inhibitory=0.25, seed=2)
    src = edge_index[0]
    # Inhibitory sources -> all outgoing weights <= 0; excitatory -> >= 0.
    assert np.all(w[is_inh[src]] <= 0.0)
    assert np.all(w[~is_inh[src]] >= 0.0)
    # Roughly the requested inhibitory fraction.
    assert abs(is_inh.mean() - 0.25) < 0.1


# --------------------------------------------------------------------- model
def _brain(**kw):
    return PositronicBrain(BrainConfig(grid_size=4, **kw), device="cpu")


def test_forward_shapes():
    b = _brain()
    Z = b.config.num_zones
    out = b(torch.zeros(3, Z))
    assert out.shape == (3, 1)
    assert torch.all((out >= 0) & (out <= 1))


def test_single_sample_input():
    b = _brain()
    Z = b.config.num_zones
    out = b(torch.zeros(Z))  # 1-D input is auto-batched
    assert out.shape == (1, 1)


def test_determinism_same_seed():
    b1 = _brain(seed=7)
    b2 = _brain(seed=7)
    x = torch.ones(2, b1.config.num_zones)
    with torch.no_grad():
        o1, o2 = b1(x), b2(x)
    # Same seed -> identical graph + identical untrained params.
    assert torch.allclose(o1, o2)


def test_resting_state_low_firing():
    b = _brain()
    Z = b.config.num_zones
    res = b.run_with_inputs(np.zeros(Z))
    # With no external drive, membrane stays near E_L=0 so firing rate is low.
    assert res["rates"].mean() < 0.25


def test_input_increases_activity():
    b = _brain()
    Z = b.config.num_zones
    rest = b.run_with_inputs(np.zeros(Z))["rates"].mean()
    driven = b.run_with_inputs(np.ones(Z))["rates"].mean()
    assert driven > rest


def test_signed_weights_match_dale():
    b = _brain()
    w = b.signed_weights().detach().numpy()
    src = b.edge_index[0].numpy()
    is_inh = b.is_inhibitory.numpy()
    assert np.all(w[is_inh[src]] <= 0.0)
    assert np.all(w[~is_inh[src]] >= 0.0)


# ---------------------------------------------------------------- learning
def test_synapses_are_learnable():
    b = _brain(seed=3)
    # edge_weight must be a leaf parameter that requires grad.
    assert b.edge_weight.requires_grad
    Z = b.config.num_zones
    x = torch.rand(8, Z)
    y = (x[:, 0] > 0.5).float().unsqueeze(1)
    opt = torch.optim.Adam(b.parameters(), lr=0.05)
    loss_fn = torch.nn.BCELoss()

    losses = []
    for _ in range(40):
        opt.zero_grad()
        out = b(x)
        loss = loss_fn(out, y)
        loss.backward()
        # Gradient must flow into the synaptic weights.
        assert b.edge_weight.grad is not None
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]  # learning reduces loss


# ------------------------------------------------------------- persistence
def test_save_load_roundtrip(tmp_path):
    b = _brain(seed=11)
    x = torch.rand(4, b.config.num_zones)
    with torch.no_grad():
        before = b(x)
    path = tmp_path / "brain.pt"
    b.save(str(path))
    b2 = PositronicBrain.load(str(path), device="cpu")
    with torch.no_grad():
        after = b2(x)
    assert torch.allclose(before, after, atol=1e-6)


# ----------------------------------------------------------------- scaling
def test_scaling_larger_grid():
    b = PositronicBrain(BrainConfig(grid_size=6, k_max=20), device="cpu")
    assert b.num_neurons == 216
    # Sparse: far fewer edges than the dense N^2 = 46656.
    assert b.num_edges <= 216 * 20
    out = b(torch.zeros(2, b.config.num_zones))
    assert out.shape == (2, 1)
