"""
Tests for the multimodal sensory pathway, encoders, and online learning.

All tests run fully offline using the deterministic :class:`FallbackEncoder`
(no model downloads, no network), so they validate the *architecture* and the
*learning machinery* deterministically.
"""

import numpy as np
import torch

from positronic_brain.model import PositronicBrain, BrainConfig
from positronic_brain.encoders import (
    FallbackEncoder,
    build_encoders,
    embedding_dims,
    l2_normalize,
)
from positronic_brain.online import (
    ReplayBuffer,
    OnlineLearner,
    OnlineConfig,
    LiveBrain,
)


# --------------------------------------------------------------------- encoders
def test_fallback_encoder_deterministic_and_normalised():
    enc = FallbackEncoder("text", embedding_dim=64)
    a = enc.embed("hello world")
    b = enc.embed("hello world")
    assert a.shape == (64,)
    assert np.allclose(a, b)                       # deterministic
    assert abs(np.linalg.norm(a) - 1.0) < 1e-4     # L2-normalised


def test_fallback_encoder_content_sensitive():
    enc = FallbackEncoder("text", embedding_dim=128)
    a = enc.embed("the cat sat on the mat")
    b = enc.embed("a completely different sentence")
    # different content -> different embedding
    assert np.dot(a, b) < 0.999


def test_fallback_image_encoder_handles_array():
    enc = FallbackEncoder("image", embedding_dim=512)
    img = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype("uint8")
    v = enc.embed(img)
    assert v.shape == (512,)
    assert np.isfinite(v).all()


def test_build_encoders_offline_fallback():
    encs = build_encoders(["image", "text", "audio"], prefer_pretrained=False)
    dims = embedding_dims(encs)
    assert set(encs) == {"image", "text", "audio"}
    assert all(d > 0 for d in dims.values())


# ----------------------------------------------------------------- sensory path
def test_brain_multimodal_shapes():
    cfg = BrainConfig(
        grid_size=6,
        sensory_embedding_dims={"image": 32, "text": 40},
        modality_to_zone={"image": "Visual", "text": "Association"},
        seed=1,
    )
    brain = PositronicBrain(cfg, device="cpu")
    sens = {"image": np.random.randn(32).astype("f"),
            "text": np.random.randn(40).astype("f")}
    state = brain.run_multimodal(sens)
    assert state["rates"].shape == (1, cfg.num_neurons)
    assert state["recon"]["image"].shape == (1, 32)
    assert state["recon"]["text"].shape == (1, 40)


def test_brain_backward_compatible_zone_scalar():
    # A brain with no modalities behaves exactly like the classic v3 brain.
    brain = PositronicBrain(BrainConfig(grid_size=4, seed=2), device="cpu")
    out = brain.run_with_inputs(np.zeros(brain.config.num_zones, dtype="f"))
    assert out["output"].shape == (1,)


def test_sensory_gradients_flow():
    cfg = BrainConfig(
        grid_size=5,
        sensory_embedding_dims={"image": 16, "text": 16},
        modality_to_zone={"image": "Visual", "text": "Association"},
        seed=3,
    )
    brain = PositronicBrain(cfg, device="cpu")
    brain.train()
    t = {"image": torch.randn(2, 16), "text": torch.randn(2, 16)}
    _, recon = brain(None, sensory=t, return_recon=True)
    loss = sum(((recon[m] - t[m]) ** 2).mean() for m in t)
    loss.backward()
    g = brain.sensory_encode["image"].weight.grad
    assert g is not None and torch.isfinite(g).all()


# --------------------------------------------------------------- replay buffer
def test_replay_buffer_capacity_and_sample():
    buf = ReplayBuffer(capacity=3, seed=0)
    for i in range(5):
        buf.add({"text": np.full(4, i, dtype="f")})
    assert len(buf) == 3                       # bounded
    batch = buf.sample(2)
    assert len(batch) == 2


# --------------------------------------------------------------- online learner
def _toy_dataset(n=8, dim=24, seed=0):
    rng = np.random.default_rng(seed)
    img_enc = FallbackEncoder("image", embedding_dim=dim)
    txt_enc = FallbackEncoder("text", embedding_dim=dim)
    samples = []
    for i in range(n):
        img = (rng.random((12, 12, 3)) * 255).astype("uint8")
        txt = f"sample number {i} description {i * 7 % 5}"
        samples.append({"image": img_enc.embed(img), "text": txt_enc.embed(txt)})
    return samples


def test_online_learning_reduces_reconstruction_loss():
    cfg = BrainConfig(
        grid_size=6,
        sensory_embedding_dims={"image": 24, "text": 24},
        modality_to_zone={"image": "Visual", "text": "Association"},
        seed=5,
    )
    brain = PositronicBrain(cfg, device="cpu")
    learner = OnlineLearner(
        brain,
        OnlineConfig(lr=1e-2, batch_size=8, modality_dropout=0.0, seed=0),
    )
    data = _toy_dataset(n=8, dim=24, seed=1)
    for s in data:
        learner.buffer.add(s)

    first = learner.consolidate(steps=1, batch_size=8)
    for _ in range(60):
        learner.consolidate(steps=1, batch_size=8)
    last = learner.consolidate(steps=1, batch_size=8)
    assert last < first * 0.9                  # loss meaningfully decreased


def test_cross_modal_recall_better_than_chance():
    """After training with modality dropout, image-only drive should reconstruct
    the *paired* text embedding better than an unrelated text embedding."""
    dim = 24
    cfg = BrainConfig(
        grid_size=6,
        sensory_embedding_dims={"image": dim, "text": dim},
        modality_to_zone={"image": "Visual", "text": "Association"},
        seed=7,
    )
    brain = PositronicBrain(cfg, device="cpu")
    learner = OnlineLearner(
        brain,
        OnlineConfig(lr=1e-2, batch_size=8, modality_dropout=0.5, seed=0),
    )
    data = _toy_dataset(n=6, dim=dim, seed=2)
    for s in data:
        learner.buffer.add(s)
    for _ in range(250):
        learner.consolidate(steps=1, batch_size=6)

    # Drive with image only; compare recalled text to true paired vs others.
    paired_sims, cross_sims = [], []
    for i, s in enumerate(data):
        recon = brain.run_multimodal({"image": s["image"]})["recon"]["text"][0]
        recon = l2_normalize(recon)
        paired_sims.append(float(np.dot(recon, l2_normalize(s["text"]))))
        others = [data[j]["text"] for j in range(len(data)) if j != i]
        cross_sims.append(float(np.mean([np.dot(recon, l2_normalize(o)) for o in others])))
    assert np.mean(paired_sims) > np.mean(cross_sims)


# ------------------------------------------------------------------- LiveBrain
def test_livebrain_perceive_and_learn_offline():
    live = LiveBrain.create(
        modalities=["image", "text"],
        grid_size=6,
        prefer_pretrained=False,
        online_config=OnlineConfig(lr=1e-2, train_every=1, train_steps=1, batch_size=4),
        device="cpu",
        seed=11,
    )
    img = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype("uint8")
    state = live.perceive(image=img, text="a small grey square")
    assert "recon" in state and "image" in state["recon"]
    assert state["output"].shape == (1,)
    # a handful of perceptions should populate the buffer and run consolidation
    for k in range(6):
        live.perceive(image=img, text=f"observation {k}")
    assert len(live.learner.buffer) >= 1


def test_multimodal_save_load_roundtrip(tmp_path):
    cfg = BrainConfig(
        grid_size=5,
        sensory_embedding_dims={"image": 16, "text": 16},
        modality_to_zone={"image": "Visual", "text": "Association"},
        seed=9,
    )
    brain = PositronicBrain(cfg, device="cpu")
    sens = {"image": np.random.randn(16).astype("f"), "text": np.random.randn(16).astype("f")}
    before = brain.run_multimodal(sens)["recon"]["text"]
    path = tmp_path / "mm_brain.pt"
    brain.save(str(path))
    reloaded = PositronicBrain.load(str(path), device="cpu")
    after = reloaded.run_multimodal(sens)["recon"]["text"]
    assert np.allclose(before, after, atol=1e-5)
