"""Tests for divisive normalization (Carandini & Heeger gain control)."""

import torch

from positronic_brain.model import BrainConfig, PositronicBrain


def _brain(use_divnorm):
    cfg = BrainConfig(grid_size=6, seed=0, use_divnorm=use_divnorm)
    return PositronicBrain(cfg, device="cpu")


def test_divnorm_off_is_identical_to_baseline():
    # Flag off must be byte-identical to a model that never knew about divnorm.
    torch.manual_seed(0)
    b_off = _brain(False)
    b_def = PositronicBrain(BrainConfig(grid_size=6, seed=0), device="cpu")
    ext = torch.ones(1, b_off.config.num_zones)
    out_off = b_off.run_with_inputs(ext.numpy()[0])["rates"]
    out_def = b_def.run_with_inputs(ext.numpy()[0])["rates"]
    assert torch.allclose(torch.tensor(out_off), torch.tensor(out_def))


def test_divnorm_changes_dynamics():
    b_off = _brain(False)
    b_on = _brain(True)
    ext = torch.full((b_off.config.num_zones,), 3.0)
    r_off = torch.tensor(b_off.run_with_inputs(ext.numpy())["rates"])
    r_on = torch.tensor(b_on.run_with_inputs(ext.numpy())["rates"])
    assert not torch.allclose(r_off, r_on)


def test_divnorm_bounds_excitation_under_strong_drive():
    # Under strong external drive, divisive normalization should keep the mean
    # firing rate no higher than the un-normalised network (gain control).
    b_off = _brain(False)
    b_on = _brain(True)
    ext = torch.full((b_off.config.num_zones,), 8.0)
    mean_off = float(torch.tensor(b_off.run_with_inputs(ext.numpy())["rates"]).mean())
    mean_on = float(torch.tensor(b_on.run_with_inputs(ext.numpy())["rates"]).mean())
    assert mean_on <= mean_off + 1e-4


def test_divnorm_gradients_flow():
    b_on = _brain(True)
    ext = torch.ones(1, b_on.config.num_zones, requires_grad=False)
    out = b_on(ext)
    out.sum().backward()
    assert b_on.edge_weight.grad is not None
    assert torch.isfinite(b_on.edge_weight.grad).all()
