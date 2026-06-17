"""Tests for homeostatic intrinsic-gain control."""

import torch

from positronic_brain.model import BrainConfig, PositronicBrain


def test_homeostasis_off_has_no_buffers_and_matches_baseline():
    torch.manual_seed(0)
    b_off = PositronicBrain(BrainConfig(grid_size=6, seed=0, use_homeostasis=False),
                            device="cpu")
    assert not hasattr(b_off, "homeo_gain")
    b_def = PositronicBrain(BrainConfig(grid_size=6, seed=0), device="cpu")
    ext = torch.full((b_off.config.num_zones,), 2.0)
    r_off = torch.tensor(b_off.run_with_inputs(ext.numpy())["rates"])
    r_def = torch.tensor(b_def.run_with_inputs(ext.numpy())["rates"])
    assert torch.allclose(r_off, r_def)


def test_homeostasis_raises_gain_for_silent_neurons():
    # With activity far below target, the controller should RAISE the gain.
    cfg = BrainConfig(grid_size=6, seed=0, use_homeostasis=True,
                      homeo_target_rate=0.2, homeo_tau=5.0, homeo_lr=0.1)
    brain = PositronicBrain(cfg, device="cpu")
    brain.train()
    g0 = brain.homeo_gain.clone()
    V = torch.full((2, brain.num_neurons), -2.0)  # strongly hyperpolarised -> ~silent
    I_ext = torch.zeros((2, brain.num_neurons))
    for _ in range(50):
        V = brain.step(V, I_ext)
    assert float(brain.homeo_gain.mean()) > float(g0.mean())


def test_homeostasis_frozen_in_eval():
    cfg = BrainConfig(grid_size=5, seed=0, use_homeostasis=True, homeo_tau=2.0)
    brain = PositronicBrain(cfg, device="cpu")
    brain.eval()
    g0 = brain.homeo_gain.clone()
    V = torch.full((1, brain.num_neurons), 0.6)
    for _ in range(10):
        V = brain.step(V, torch.zeros((1, brain.num_neurons)))
    assert torch.allclose(brain.homeo_gain, g0)  # no drift in eval


def test_homeostasis_checkpoint_roundtrip(tmp_path):
    cfg = BrainConfig(grid_size=5, seed=0, use_homeostasis=True)
    brain = PositronicBrain(cfg, device="cpu")
    p = tmp_path / "h.pt"
    brain.save(str(p))
    loaded = PositronicBrain.load(str(p), device="cpu")
    assert hasattr(loaded, "homeo_gain")
