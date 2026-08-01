"""
Unified checkpoints for brain + baseline models.

Purpose
-------
Every serious training run should leave a **reloadable artifact** so we can:

1. **Resume** interrupted training
2. **Fine-tune** on new public data / domain text
3. **Align** with preference / DPO (and later RL-style loops)
4. **Scale curricula**: smaller pretrain → larger grid / more data

Layout (default)
----------------
    checkpoints/<run_name>/
        meta.json                 # protocol, metrics, paths
        tokenizer.json            # CharTokenizer or SubwordTokenizer
        brain.pt / brain_wm.pt    # BrainLanguageModel bundles
        lstm.pt / gpt.pt / ...    # baseline bundles

Brain bundles use the same schema as :meth:`BrainLanguageModel.save`.
Baseline bundles add ``kind`` + architecture hyperparams for rebuild.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .language import BrainLanguageModel, CharTokenizer, LMConfig
from .subword import SubwordTokenizer
from .utils import get_device


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_tokenizer(tok: Any, path: str) -> None:
    """Save CharTokenizer or SubwordTokenizer to JSON."""
    d = tok.to_dict()
    if "type" not in d:
        # CharTokenizer historically has no type field
        d = {"type": "char", **d} if not isinstance(tok, SubwordTokenizer) else d
    if isinstance(tok, SubwordTokenizer):
        d["type"] = "subword_bpe"
    else:
        d.setdefault("type", "char")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def load_tokenizer(path: str):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    t = d.get("type", "char")
    if t in ("subword_bpe", "subword"):
        return SubwordTokenizer.from_dict(d)
    return CharTokenizer.from_dict(d)


def save_brain(
    model: BrainLanguageModel,
    path: str,
    tokenizer=None,
    *,
    extra: Optional[Dict] = None,
) -> str:
    """Save brain LM (compatible with BrainLanguageModel.load + extras)."""
    ensure_dir(os.path.dirname(path) or ".")
    payload = {
        "kind": "brain",
        "lm_config": model.config.to_dict(),
        "vocab_size": model.vocab_size,
        "tokenizer": tokenizer.to_dict() if tokenizer is not None else None,
        "tokenizer_type": (
            "subword_bpe" if isinstance(tokenizer, SubwordTokenizer)
            else ("char" if tokenizer is not None else None)
        ),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "extra": extra or {},
    }
    if tokenizer is not None and isinstance(tokenizer, SubwordTokenizer):
        payload["tokenizer"]["type"] = "subword_bpe"
    torch.save(payload, path)
    return path


def load_brain(
    path: str,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[BrainLanguageModel, Optional[Any], Dict]:
    """Load brain; returns (model, tokenizer_or_None, extra)."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LMConfig.from_dict(ckpt.get("lm_config", {}))
    model = BrainLanguageModel(ckpt["vocab_size"], cfg, device=device)
    model.load_state_dict(ckpt["state_dict"])
    tok = None
    if ckpt.get("tokenizer"):
        td = ckpt["tokenizer"]
        if ckpt.get("tokenizer_type") == "subword_bpe" or td.get("type") == "subword_bpe":
            tok = SubwordTokenizer.from_dict(td)
        else:
            tok = CharTokenizer.from_dict(td)
    return model, tok, ckpt.get("extra") or {}


def save_baseline(
    model: nn.Module,
    path: str,
    *,
    kind: str,
    arch: Dict,
    vocab_size: int,
    tokenizer=None,
    extra: Optional[Dict] = None,
) -> str:
    """Save LSTM/RNN/CNN/GPT baseline with rebuild recipe in ``arch``."""
    ensure_dir(os.path.dirname(path) or ".")
    payload = {
        "kind": kind,
        "arch": arch,
        "vocab_size": vocab_size,
        "tokenizer": tokenizer.to_dict() if tokenizer is not None else None,
        "tokenizer_type": (
            "subword_bpe" if isinstance(tokenizer, SubwordTokenizer)
            else ("char" if tokenizer is not None else None)
        ),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "extra": extra or {},
    }
    if tokenizer is not None and isinstance(tokenizer, SubwordTokenizer):
        payload["tokenizer"]["type"] = "subword_bpe"
    torch.save(payload, path)
    return path


def load_baseline(path: str, device: Union[str, torch.device] = "cpu"):
    """Rebuild baseline from checkpoint. Returns (model, tokenizer, meta)."""
    from experiments.matched_experiment import CharCNN, CharLSTM, CharRNN
    from experiments.public_lm_eval import CharGPT

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    kind = ckpt["kind"]
    arch = ckpt["arch"]
    V = ckpt["vocab_size"]
    dev = get_device(device) if isinstance(device, str) else device

    if kind == "lstm":
        model = CharLSTM(V, arch["embed_dim"], arch["hidden"]).to(dev)
    elif kind == "rnn":
        model = CharRNN(V, arch["embed_dim"], arch["hidden"]).to(dev)
    elif kind == "cnn":
        model = CharCNN(
            V,
            embed_dim=arch["embed_dim"],
            channels=arch["channels"],
            n_layers=arch["n_layers"],
        ).to(dev)
    elif kind == "gpt":
        model = CharGPT(
            V,
            d_model=arch["d_model"],
            n_layer=arch["n_layer"],
            n_head=arch["n_head"],
            max_seq=arch.get("max_seq", 256),
        ).to(dev)
    else:
        raise ValueError(f"unknown baseline kind {kind!r}")
    model.load_state_dict(ckpt["state_dict"])
    tok = None
    if ckpt.get("tokenizer"):
        td = ckpt["tokenizer"]
        if ckpt.get("tokenizer_type") == "subword_bpe" or td.get("type") == "subword_bpe":
            tok = SubwordTokenizer.from_dict(td)
        else:
            tok = CharTokenizer.from_dict(td)
    return model, tok, {"kind": kind, "arch": arch, "extra": ckpt.get("extra") or {}}


def write_run_meta(run_dir: str, meta: Dict) -> str:
    ensure_dir(run_dir)
    path = os.path.join(run_dir, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def default_run_dir(root: str, run_name: str) -> str:
    return ensure_dir(os.path.join(root, "checkpoints", run_name))


def arch_from_baseline(kind: str, model: nn.Module) -> Dict:
    """Extract rebuild hyperparams from a constructed baseline module."""
    kind = kind.lower()
    if kind == "lstm":
        return {
            "embed_dim": model.embed.embedding_dim,
            "hidden": model.lstm.hidden_size,
        }
    if kind == "rnn":
        return {
            "embed_dim": model.embed.embedding_dim,
            "hidden": model.rnn.hidden_size,
        }
    if kind == "cnn":
        ch = model.head.in_features
        emb = model.embed.embedding_dim
        n_layers = sum(1 for m in model.net if m.__class__.__name__ == "CausalConv1d")
        return {"embed_dim": emb, "channels": ch, "n_layers": max(n_layers, 1)}
    if kind == "gpt":
        return dict(model.config)
    raise ValueError(f"not a baseline kind: {kind}")


def _vocab_size_of(model: nn.Module) -> int:
    if hasattr(model, "vocab_size"):
        return int(model.vocab_size)
    if hasattr(model, "embed"):
        return int(model.embed.num_embeddings)
    if hasattr(model, "tok_emb"):
        return int(model.tok_emb.num_embeddings)
    raise ValueError("cannot infer vocab_size")


def save_model_bundle(
    name: str,
    model: nn.Module,
    path: str,
    *,
    tokenizer=None,
    metrics: Optional[Dict] = None,
    protocol: Optional[Dict] = None,
) -> str:
    """Dispatch brain vs baseline save."""
    extra = {"metrics": metrics or {}, "protocol": protocol or {}, "name": name}
    if name in ("brain", "brain_wm") or isinstance(model, BrainLanguageModel):
        return save_brain(model, path, tokenizer, extra=extra)
    kind = name if name in ("lstm", "rnn", "cnn", "gpt") else getattr(model, "kind", name)
    arch = arch_from_baseline(kind, model)
    return save_baseline(
        model, path, kind=kind, arch=arch,
        vocab_size=_vocab_size_of(model),
        tokenizer=tokenizer,
        extra=extra,
    )
