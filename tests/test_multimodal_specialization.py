"""Tests for the multi-stream brain and the specialization-measurement toolkit."""

import numpy as np
import torch

from positronic_brain.multimodal import MultiStreamBrain, StreamSpec
from positronic_brain.multimodal_data import (
    synthetic_scenes, batch_scenes, per_stream_samples,
)
from positronic_brain import specialization as spec

DIMS = {"vision": 16, "audio": 16, "text": 16}


def _model():
    streams = [
        StreamSpec("vision", 16, "Visual"),
        StreamSpec("audio", 16, "Auditory"),
        StreamSpec("text", 16, "Association"),
    ]
    return MultiStreamBrain(streams, grid_size=8, inner_steps=3, seed=0, device="cpu")


def _train(m, scenes, steps=60, lr=5e-3):
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        idx = rng.integers(0, len(scenes), size=16)
        loss = m.loss(batch_scenes(scenes, idx))
        opt.zero_grad(); loss.backward(); opt.step()
    return opt


def test_builds_with_distinct_entry_doors():
    m = _model()
    assert set(m.routing) == {"vision", "audio", "text"}
    assert m.routing["vision"] == "Visual"
    vi = getattr(m.brain, "_zoneidx_vision")
    ai = getattr(m.brain, "_zoneidx_audio")
    assert vi.numel() > 0 and ai.numel() > 0
    # different streams target different (disjoint) zone-neuron sets
    assert set(vi.tolist()).isdisjoint(set(ai.tolist()))


def test_loss_decreases():
    torch.manual_seed(0)
    m = _model()
    scenes, _ = synthetic_scenes(DIMS, n=200, seed=1)
    rng = np.random.default_rng(1)
    first = float(m.loss(batch_scenes(scenes, rng.integers(0, 200, 16))).item())
    _train(m, scenes, steps=60)
    last = float(m.loss(batch_scenes(scenes, rng.integers(0, 200, 16))).item())
    assert last < first


def test_streams_decodable_above_chance():
    # The entry-door routing alone should make the streams linearly separable in
    # neural activity (decoding well above the 1/3 chance level).
    torch.manual_seed(0)
    m = _model()
    scenes, _ = synthetic_scenes(DIMS, n=240, seed=2)
    _train(m, scenes, steps=40)
    samples = per_stream_samples(scenes[:120])
    acc = spec.zone_decoding_accuracy(m, samples)
    assert acc > 0.6


def test_selectivity_and_lesion():
    m = _model()
    scenes, _ = synthetic_scenes(DIMS, n=60, seed=3)
    samples = per_stream_samples(scenes)
    sel = spec.selectivity_index(m, samples)
    assert set(sel) == set(m.brain.config.zone_names)
    for zone, (stream, val) in sel.items():
        assert stream in DIMS
        assert -1e-6 <= val <= 1 + 1e-6
    eff = spec.lesion_effect(m, batch_scenes(scenes, range(8)), "Visual")
    assert set(eff) == set(DIMS)


def test_rsa_self_is_one():
    m = _model()
    scenes, _ = synthetic_scenes(DIMS, n=20, seed=5)
    rates = m.neuron_rates(batch_scenes(scenes, range(20)))
    rdm = spec.representational_dissimilarity(rates)
    assert abs(spec.rsa(rdm, rdm) - 1.0) < 1e-4


def test_reroute_changes_entry_door():
    m = _model()
    before = getattr(m.brain, "_zoneidx_vision").clone()
    m.reroute("vision", "Memory")
    assert m.routing["vision"] == "Memory"
    after = getattr(m.brain, "_zoneidx_vision")
    assert not (before.numel() == after.numel() and torch.equal(before, after))
    scenes, _ = synthetic_scenes(DIMS, n=8, seed=4)
    out = m.perceive(batch_scenes(scenes, range(8)))
    assert out["vision"].shape == (8, 16)
