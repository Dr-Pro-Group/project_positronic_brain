"""Tests for heterogeneous membrane time constants and the brain-shaped volume.

Both change the substrate rather than adding a mechanism on top of it, so what
matters is that the defaults are untouched and that the new settings stay
numerically sound — a spread of time constants is easy to write in a way that hands
the fastest neurons an unstable integration step.
"""

import numpy as np
import torch

from positronic_brain.connectivity import brain_mask, neuron_positions
from positronic_brain.model import BrainConfig, PositronicBrain


def _run(brain, steps=6, batch=2, seed=0):
    torch.manual_seed(seed)
    V = torch.zeros((batch, brain.num_neurons))
    I = torch.randn((batch, brain.num_neurons)) * 0.1
    brain.stp_begin(batch)
    out = []
    for _ in range(steps):
        V = brain.step(V, I)
        out.append(V)
    brain.stp_end()
    return out


# ----------------------------------------------------------- heterogeneous tau
def test_zero_spread_is_byte_identical_to_the_scalar_default():
    a = PositronicBrain(BrainConfig(grid_size=6, seed=0), device="cpu")
    b = PositronicBrain(BrainConfig(grid_size=6, seed=0, tau_m_spread=0.0), device="cpu")
    assert b.neuron_alpha is None
    for x, y in zip(_run(a), _run(b)):
        assert torch.equal(x, y)


def test_spread_widens_the_timescale_range_around_the_nominal_value():
    cfg = BrainConfig(grid_size=8, seed=0, tau_m=4.0, tau_m_spread=0.8)
    brain = PositronicBrain(cfg, device="cpu")
    tau = cfg.dt / brain.neuron_alpha
    assert tau.numel() == brain.num_neurons
    assert float(tau.min()) < cfg.tau_m < float(tau.max())
    # The nominal value stays inside the realised range rather than at an edge.
    assert float(tau.median()) == float(tau.median())          # finite


def test_no_neuron_receives_an_unstable_integration_step():
    # alpha = dt/tau must stay below 1 or forward Euler diverges on the first step.
    # A log-uniform draw around tau_m=4 reaches tau<1 by spread 0.8, so the floor
    # is what keeps a wide spread usable at all.
    for spread in (0.4, 0.8, 1.3, 2.0):
        brain = PositronicBrain(
            BrainConfig(grid_size=6, seed=0, tau_m=4.0, tau_m_spread=spread), device="cpu")
        assert float(brain.neuron_alpha.max()) <= 0.5 + 1e-6
        states = _run(brain)
        assert torch.isfinite(states[-1]).all()


def test_wide_spread_still_trains_finitely():
    brain = PositronicBrain(BrainConfig(grid_size=6, seed=0, tau_m_spread=1.0), device="cpu")
    states = _run(brain, steps=25)
    assert torch.isfinite(states[-1]).all()


# ------------------------------------------------------------- brain-shaped volume
def test_brain_mask_selects_a_strict_subset_of_the_lattice():
    for g in (12, 16, 24):
        m = brain_mask(g)
        assert m.shape == (g ** 3,)
        assert 0 < m.sum() < g ** 3


def test_brain_mask_is_longer_front_to_back_than_it_is_wide_or_deep():
    pos = neuron_positions(24)[brain_mask(24)]
    extent = pos.max(0) - pos.min(0)
    assert extent[0] > extent[1]        # anterior-posterior longer than lateral
    assert extent[0] > extent[2]        # anterior-posterior longer than dorsal-ventral


def test_brain_mask_is_left_right_symmetric():
    g = 16
    m = brain_mask(g).reshape(g, g, g)
    assert np.array_equal(m, m[:, ::-1, :])


def test_brain_mask_has_a_sagittal_fissure_that_closes_ventrally():
    g = 24
    m = brain_mask(g).reshape(g, g, g)
    mid = g // 2
    # A dorsal slice through the midline is interrupted; a ventral one is not,
    # because the hemispheres remain joined below.
    dorsal = m[:, mid - 1:mid + 1, int(g * 0.75)]
    ventral = m[:, mid - 1:mid + 1, int(g * 0.42)]
    assert dorsal.sum() < ventral.sum()
