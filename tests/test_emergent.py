"""
Pytest suite for the input-anchored emergent-zone research stub
(:mod:`positronic_brain.emergent`).

Covers: construction and port seeding, current injection, the plastic recurrent
step (Oja-normalised Hebbian update), the emergent-zone read-out, and the
experimental structural-plasticity rewrite (prune + Dale-consistent grow), plus
its gating flag. These verify the *runnable* parts of the stub; the stable
plasticity<->BPTT schedule remains the open research item (see paper Sec. 7.4).

Run from the project root with the venv active:  pytest -q
"""

from __future__ import annotations

import torch

from positronic_brain.emergent import (
    EmergentZoneBrain,
    EmergentZoneConfig,
    PlasticityConfig,
    PortSpec,
)


def _tiny_brain(rule="hebbian", grow=0):
    cfg = EmergentZoneConfig(
        grid_size=6,                       # 216 neurons: small & fast
        ports=[
            PortSpec("chiasm", "vision", 16, center=(0.2, 0.5, 0.5), radius=2.0),
            PortSpec("auditory_nerve", "audio", 12, center=(0.8, 0.5, 0.5), radius=2.0),
        ],
        plasticity=PlasticityConfig(rule=rule, grow_per_step=grow, prune_thresh=1e-3),
        seed=0,
    )
    return EmergentZoneBrain(cfg, device="cpu")


def test_construction_and_ports():
    ez = _tiny_brain()
    assert ez.brain.num_neurons == 216
    assert ez._ports == ["chiasm", "auditory_nerve"]
    # each port seeded a non-empty entry patch with a matching projection
    for name in ez._ports:
        idx = getattr(ez, f"_portidx_{name}")
        assert idx.numel() >= 1
        assert name in ez.port_proj


def test_inject_shapes_and_locality():
    ez = _tiny_brain()
    I = ez.inject(chiasm=torch.randn(16))
    assert I.shape == (1, ez.brain.num_neurons)
    # current lands only on the chiasm's entry patch, nowhere else
    idx = getattr(ez, "_portidx_chiasm")
    mask = torch.ones(ez.brain.num_neurons, dtype=torch.bool)
    mask[idx] = False
    assert torch.allclose(I[0][mask], torch.zeros(int(mask.sum())))
    assert I[0][idx].abs().sum() > 0


def test_plastic_step_stable_and_nonneg():
    ez = _tiny_brain(rule="hebbian")
    V = ez.init_state(1)
    w0 = ez.brain.edge_weight.data.clone()
    for _ in range(8):
        V = ez.step_with_plasticity(V, ez.inject(chiasm=torch.randn(16)),
                                    driven_port="chiasm")
    assert torch.isfinite(V).all()
    w = ez.brain.edge_weight.data
    assert torch.isfinite(w).all()
    assert (w >= 0).all()                 # Dale magnitudes stay non-negative
    assert not torch.allclose(w, w0)      # Hebbian actually changed weights


def test_emergent_zones_readout():
    ez = _tiny_brain()
    V = ez.init_state(1)
    for _ in range(5):
        V = ez.step_with_plasticity(V, ez.inject(chiasm=torch.randn(16)),
                                    driven_port="chiasm")
    for _ in range(5):
        V = ez.step_with_plasticity(V, ez.inject(auditory_nerve=torch.randn(12)),
                                    driven_port="auditory_nerve")
    zones = ez.emergent_zones()
    assert zones.shape == (ez.brain.num_neurons,)
    assert int(zones.min()) >= 0 and int(zones.max()) < len(ez._ports)


def test_emergent_zones_requires_drive():
    ez = _tiny_brain()
    try:
        ez.emergent_zones()
        assert False, "expected RuntimeError before any port drive"
    except RuntimeError:
        pass


def test_structural_step_gated():
    ez = _tiny_brain(rule="hebbian")          # not "structural"
    try:
        ez.structural_step()
        assert False, "structural_step must be gated behind rule='structural'"
    except RuntimeError:
        pass


def test_structural_step_prune_grow_keeps_graph_valid():
    ez = _tiny_brain(rule="structural", grow=20)
    # warm activity so growth has something to sample from
    V = ez.init_state(1)
    for _ in range(5):
        V = ez.step_with_plasticity(V, ez.inject(chiasm=torch.randn(16)),
                                    driven_port="chiasm")
    E_before = ez.brain.edge_index.shape[1]
    stats = ez.structural_step()
    assert set(stats) == {"pruned", "grown", "edges"}
    assert stats["edges"] == ez.brain.edge_index.shape[1]

    b = ez.brain
    E = b.edge_index.shape[1]
    # graph stays internally consistent after the rewrite
    assert b.edge_sign.shape == (E,)
    assert b.edge_weight.shape == (E,)
    assert int(b.edge_index.max()) < b.num_neurons
    # Dale's law preserved: every edge's sign matches its presynaptic neuron
    src = b.edge_index[0]
    expected = torch.where(b.is_inhibitory[src], -1.0, 1.0)
    assert torch.equal(b.edge_sign, expected)
    # the rewritten brain still runs and stays finite
    V2 = b.step(ez.init_state(1), torch.zeros(1, b.num_neurons))
    assert torch.isfinite(V2).all()
    assert E_before > 0 and E > 0
