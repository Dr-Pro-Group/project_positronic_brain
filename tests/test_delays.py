"""Tests for distance-dependent axonal conduction delays (--delays).

Each synapse transmits the presynaptic firing rate from ``round(edge_dist /
delay_velocity)`` integration steps ago rather than the current one, so the
network's 3D geometry determines *when* a signal lands and not only *where* it
goes. Off by default and byte-identical to baseline when off.
"""

import torch

from positronic_brain.model import BrainConfig, PositronicBrain
from positronic_brain.language import BrainLanguageModel, CharTokenizer, LMConfig


def _drive(brain, steps, batch=2, seed=0, amp=0.1):
    """Run ``steps`` integration steps under a fixed random drive."""
    torch.manual_seed(seed)
    V = torch.zeros((batch, brain.num_neurons))
    I = torch.randn((batch, brain.num_neurons)) * amp
    brain.stp_begin(batch)
    states = []
    for _ in range(steps):
        V = brain.step(V, I)
        states.append(V)
    brain.stp_end()
    return states


def test_delays_off_is_byte_identical_to_baseline():
    # The off==baseline contract is EXACT: with the flag off no delay line is
    # allocated and step() takes the original r[:, src] path, so the dynamics must
    # be bit-identical. Assert torch.equal, not allclose.
    b_off = PositronicBrain(BrainConfig(grid_size=6, seed=0, use_delays=False), device="cpu")
    b_def = PositronicBrain(BrainConfig(grid_size=6, seed=0), device="cpu")
    ext = torch.full((b_off.config.num_zones,), 2.0)
    r_off = torch.tensor(b_off.run_with_inputs(ext.numpy())["rates"])
    r_def = torch.tensor(b_def.run_with_inputs(ext.numpy())["rates"])
    assert torch.equal(r_off, r_def)

    for a, b in zip(_drive(b_off, 6), _drive(b_def, 6)):
        assert torch.equal(a, b)


def test_delays_change_the_dynamics():
    base = _drive(PositronicBrain(BrainConfig(grid_size=6, seed=0), device="cpu"), 6)
    lagged = _drive(PositronicBrain(BrainConfig(grid_size=6, seed=0, use_delays=True),
                                    device="cpu"), 6)
    assert not torch.equal(base[-1], lagged[-1])


def test_delay_line_lifecycle():
    brain = PositronicBrain(BrainConfig(grid_size=5, seed=0, use_delays=True), device="cpu")
    assert brain._rate_hist is None
    brain.stp_begin(3)
    assert brain._rate_hist == []                    # starts empty: nothing in flight
    V = torch.zeros((3, brain.num_neurons))
    for step in range(1, 4):
        V = brain.step(V, torch.zeros_like(V))
        # Frames accumulate until the longest axon is covered, then stop.
        assert len(brain._rate_hist) == min(step, brain._max_delay)
    brain.stp_end()
    assert brain._rate_hist is None


def test_delay_line_holds_only_what_the_longest_axon_needs():
    # The buffer is bounded by the longest delay the geometry actually produced,
    # which is normally well below the configured ceiling — holding delay_max
    # frames regardless would make every step needlessly expensive.
    cfg = BrainConfig(grid_size=5, seed=0, use_delays=True, delay_max=8)
    brain = PositronicBrain(cfg, device="cpu")
    assert brain._max_delay == int(brain.edge_delay.max())
    assert brain._max_delay <= cfg.delay_max

    brain.stp_begin(1)
    V = torch.zeros((1, brain.num_neurons))
    for _ in range(10):
        V = brain.step(V, torch.zeros_like(V))
    assert len(brain._rate_hist) == brain._max_delay
    brain.stp_end()


def test_delay_max_clamps_long_axons():
    tight = PositronicBrain(
        BrainConfig(grid_size=8, seed=0, use_delays=True, delay_velocity=0.2, delay_max=3),
        device="cpu")
    assert int(tight.edge_delay.max()) == 3          # clamped by the ceiling
    assert tight._max_delay == 3


def test_delays_derive_from_distance_and_respect_bounds():
    cfg = BrainConfig(grid_size=8, seed=0, use_delays=True, delay_velocity=1.0, delay_max=8)
    brain = PositronicBrain(cfg, device="cpu")
    d = brain.edge_delay
    assert d.shape[0] == brain.num_edges
    assert int(d.min()) >= 1 and int(d.max()) <= cfg.delay_max
    assert int(d.max()) > int(d.min())               # geometry produces a real spread

    # Slower conduction must not shorten any edge's delay.
    slow = PositronicBrain(
        BrainConfig(grid_size=8, seed=0, use_delays=True, delay_velocity=0.5, delay_max=8),
        device="cpu")
    assert torch.all(slow.edge_delay >= d)


def test_signal_cannot_arrive_before_its_delay():
    # The causal signature of a delay line: after a single step nothing has
    # traversed any axon, so no synaptic weight can yet have influenced the loss.
    # After k+1 steps, exactly the edges with delay <= k can have.
    brain = PositronicBrain(BrainConfig(grid_size=6, seed=0, use_delays=True), device="cpu")
    reachable = {}
    for unroll in (1, 2, 3):
        brain.zero_grad()
        _drive(brain, unroll)[-1].pow(2).mean().backward()
        reachable[unroll] = int((brain.edge_weight.grad != 0).sum())

    assert reachable[1] == 0
    expected_after_two = int((brain.edge_delay == 1).sum())
    assert reachable[2] == expected_after_two
    assert reachable[3] > reachable[2]


def test_delays_join_the_stateful_guard():
    # The delay line is mutated inside step(), so a gradient-checkpoint recompute
    # would replay it against end-of-forward state. It must be declared stateful.
    assert PositronicBrain(BrainConfig(grid_size=5, use_delays=True),
                           device="cpu").step_mutates_state()
    assert not PositronicBrain(BrainConfig(grid_size=5), device="cpu").step_mutates_state()


def test_delays_survive_detach_across_windows():
    brain = PositronicBrain(BrainConfig(grid_size=5, seed=0, use_delays=True), device="cpu")
    brain.stp_begin(1)
    V = torch.zeros((1, brain.num_neurons))
    for _ in range(3):
        V = brain.step(V, torch.zeros_like(V))
    brain.stp_detach()
    assert len(brain._rate_hist) == brain._max_delay         # history is kept…
    assert all(not r.requires_grad for r in brain._rate_hist)   # …but its graph is not
    brain.stp_end()


def test_delays_change_language_model_and_trains():
    text = "User: hi\nBrain: hello\n" * 30
    tok = CharTokenizer.from_text(text)
    cfg = LMConfig(grid_size=6, embed_dim=16, inner_steps=2,
                   brain_overrides={"use_delays": True})
    model = BrainLanguageModel(tok.vocab_size, cfg, device="cpu")
    data = torch.tensor(tok.encode(text))
    batch = data[:60].unsqueeze(0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    first = float(model.loss_on(batch).item())
    for _ in range(30):
        loss = model.loss_on(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(model.loss_on(batch).item()) < first
    # The delay line must not leak between forward passes.
    assert model.brain._rate_hist is None


def test_delay_modes_control_for_geometry():
    """`uniform` and `shuffled` isolate lag from distance.

    Without them, any improvement from --delays reads as evidence that the spatial
    embedding matters; measured, it is the lag that matters and the geometry
    contributes nothing. The controls have to preserve the right things to say so:
    `shuffled` keeps the exact latency histogram and only moves it between edges,
    and `uniform` collapses the histogram to a single value.
    """
    base = BrainConfig(grid_size=8, seed=0, use_delays=True)
    dist = PositronicBrain(base, device="cpu")
    shuf = PositronicBrain(BrainConfig(grid_size=8, seed=0, use_delays=True,
                                       delay_mode="shuffled"), device="cpu")
    uni = PositronicBrain(BrainConfig(grid_size=8, seed=0, use_delays=True,
                                      delay_mode="uniform"), device="cpu")

    # Shuffled preserves the distribution exactly, but not the assignment.
    assert torch.equal(torch.bincount(shuf.edge_delay), torch.bincount(dist.edge_delay))
    assert not torch.equal(shuf.edge_delay, dist.edge_delay)

    # Uniform collapses to one latency everywhere.
    assert int(uni.edge_delay.unique().numel()) == 1

    # Only the distance mode should correlate with actual edge length.
    import numpy as np
    src, dst = dist.edge_index[0], dist.edge_index[1]
    d = (dist.positions[src] - dist.positions[dst]).norm(dim=1).numpy()
    r_dist = float(np.corrcoef(d, dist.edge_delay.numpy())[0, 1])
    r_shuf = float(np.corrcoef(d, shuf.edge_delay.numpy())[0, 1])
    # Rounding to integers caps the achievable correlation (the latencies take only
    # a couple of distinct values), so the contract is a strong association in the
    # distance mode and essentially none once the assignment is scrambled.
    assert r_dist > 0.8
    assert abs(r_shuf) < 0.1
    assert r_dist > 10 * abs(r_shuf)


def test_unknown_delay_mode_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="delay_mode"):
        PositronicBrain(BrainConfig(grid_size=5, use_delays=True, delay_mode="nope"),
                        device="cpu")
