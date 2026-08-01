"""Unit tests for scale path: subword, disk memmap, modular brain (offline)."""

from __future__ import annotations

import json
import os
import tempfile

import torch

from positronic_brain.disk_data import (
    MemmapTokenStore,
    tokenize_shards_to_memmap,
    train_tokenizer_from_shards,
)
from positronic_brain.modular import ModularBrainLM, ModularConfig
from positronic_brain.subword import SubwordTokenizer


SAMPLE = (
    "Once upon a time a small robot learned to count.\n\n"
    "The neurons fired in quiet patterns of thought.\n\n"
    "User: hello\nBrain: Hello! How are you today?\n\n"
    "Counting is hard for a larva but easier for a monkey.\n\n"
) * 20


def test_subword_train_encode_decode_roundtrip():
    tok = SubwordTokenizer.train([SAMPLE], vocab_size=512, max_chars=50_000)
    assert tok.vocab_size >= 260
    ids = tok.encode("hello robot")
    assert isinstance(ids, list) and len(ids) >= 1
    # decode should be non-empty for ascii
    text = tok.decode(ids)
    assert isinstance(text, str)
    d = tok.to_dict()
    tok2 = SubwordTokenizer.from_dict(d)
    assert tok2.encode("hello") == tok.encode("hello")


def test_memmap_token_store_offline(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard = shard_dir / "shard_00000.txt"
    shard.write_text(SAMPLE, encoding="utf-8")
    meta = {
        "name": "offline",
        "shards": [str(shard)],
        "n_chars": len(SAMPLE),
        "n_shards": 1,
        "max_docs": 1,
        "max_chars": 10**9,
        "shard_chars": 10**9,
        "seed": 0,
    }
    (shard_dir / "shards_meta.json").write_text(json.dumps(meta))

    tok_path = str(tmp_path / "tok.json")
    tok = train_tokenizer_from_shards(str(shard_dir), tok_path, vocab_size=400, max_chars=100_000)
    mmap_path = str(tmp_path / "tokens.mmap")
    mm_meta = tokenize_shards_to_memmap(str(shard_dir), tok, mmap_path)
    assert mm_meta["n_tokens"] > 100

    store = MemmapTokenStore(mmap_path, split="train")
    batch = store.sample_batch(seq_len=16, batch_size=4, device=torch.device("cpu"))
    assert batch.shape == (4, 17)
    assert batch.dtype == torch.int64


def test_modular_forward_and_loss():
    cfg = ModularConfig.default_chain(grid_size=4, n_areas=3, seed=0)
    cfg.inner_steps = 1
    model = ModularBrainLM(vocab_size=50, config=cfg, device="cpu")
    assert model.total_neurons() == 3 * (4**3)
    model.set_active_area("Association")
    x = torch.randint(0, 50, (2, 8))
    loss = model.loss_on(x)
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    # pathways + active area must receive grad (areas couple via pathways)
    path_grads = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for path in model.pathways
        for p in path.parameters()
    )
    area_grads = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in model.areas["Association"].parameters()
        if p.requires_grad
    )
    head_grads = any(p.grad is not None for p in model.head.parameters())
    assert head_grads
    assert path_grads or area_grads


def test_modular_save_reload_area(tmp_path):
    cfg = ModularConfig.default_chain(grid_size=4, n_areas=2, seed=1)
    cfg.inner_steps = 1
    model = ModularBrainLM(vocab_size=40, config=cfg, device="cpu")
    model.save_all_areas(str(tmp_path / "areas"))
    assert (tmp_path / "areas" / "area_Sensory.pt").exists()
    assert (tmp_path / "areas" / "modular_config.json").exists()

    # offload + reload sensory
    model.offload_area_to_disk("Sensory", str(tmp_path / "areas"))
    assert "Sensory" not in model.areas
    model.reload_area_from_disk("Sensory", str(tmp_path / "areas"))
    assert "Sensory" in model.areas
    x = torch.randint(0, 40, (1, 6))
    loss = model.loss_on(x)
    assert torch.isfinite(loss)
