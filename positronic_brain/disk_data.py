"""
Disk-backed public LM data plane: stream → shards → memmap tokens.

Goal: train on FineWeb / C4 / TinyStories-scale text **without** holding the
corpus in RAM. Pattern:

  1. ``stream_public_lm_to_shards`` — HF streaming → ``data/shards/*.txt``
  2. ``tokenize_shards_to_memmap`` — tokenizer → ``tokens.uint16.mmap`` + meta
  3. ``MemmapTokenStore`` — random windows via numpy memmap (OS page cache)

Hard-drive-as-RAM for *data* (not for every sparse scatter step).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .subword import SubwordTokenizer


# Friendly public pretrain presets (streamed; caps apply).
PUBLIC_LM_PRESETS: Dict[str, dict] = {
    "tinystories": dict(path="roneneldan/TinyStories", split="train", text_keys=("text", "story")),
    "wikitext": dict(
        path="Salesforce/wikitext", name="wikitext-2-raw-v1", split="train",
        text_keys=("text",), streaming=False,
    ),
    # Open LLM pretrain mixes — large; always use max_docs / max_chars caps.
    "fineweb-edu": dict(
        path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
        text_keys=("text",),
    ),
    "fineweb": dict(
        path="HuggingFaceFW/fineweb", name="sample-10BT", split="train",
        text_keys=("text",),
    ),
    "c4": dict(path="allenai/c4", name="en", split="train", text_keys=("text",)),
    "openwebtext": dict(path="Skylion007/openwebtext", split="train", text_keys=("text",)),
}


def _log(msg: str) -> None:
    print(f"[disk_data {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _iter_hf_texts(
    name: str,
    max_docs: int,
    max_chars: int,
) -> Iterator[str]:
    """Yield plain-text documents from a public HF dataset (streamed when possible)."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets package required for public LM streaming "
            "(pip install 'datasets>=3.0')"
        ) from exc

    key = name.lower().strip()
    if key not in PUBLIC_LM_PRESETS and "/" not in name:
        raise ValueError(
            f"unknown public LM preset {name!r}; choose from "
            f"{sorted(PUBLIC_LM_PRESETS)} or a hub path like 'org/dataset'"
        )

    if key in PUBLIC_LM_PRESETS:
        cfg = dict(PUBLIC_LM_PRESETS[key])
    else:
        cfg = dict(path=name, split="train", text_keys=("text", "content", "story"))

    text_keys = cfg.pop("text_keys", ("text",))
    use_stream = cfg.pop("streaming", True)
    path = cfg.pop("path")
    split = cfg.pop("split", "train")
    ds_name = cfg.pop("name", None)

    try:
        if ds_name is not None:
            ds = load_dataset(path, ds_name, split=split, streaming=use_stream)
        else:
            ds = load_dataset(path, split=split, streaming=use_stream)
    except Exception as exc:
        raise RuntimeError(f"could not load {name!r}: {exc}") from exc

    n_docs = 0
    n_chars = 0
    for row in ds:
        if n_docs >= max_docs or n_chars >= max_chars:
            break
        text = None
        for k in text_keys:
            v = row.get(k) if isinstance(row, dict) else None
            if isinstance(v, str) and v.strip():
                text = v.strip()
                break
        if not text:
            continue
        # skip near-empty wiki headers etc.
        if len(text) < 32:
            continue
        n_docs += 1
        n_chars += len(text)
        yield text
    _log(f"streamed {n_docs} docs / {n_chars:,} chars from {name}")


def stream_public_lm_to_shards(
    name: str,
    out_dir: str,
    *,
    max_docs: int = 50_000,
    max_chars: int = 50_000_000,
    shard_chars: int = 2_000_000,
    seed: int = 42,
) -> Dict:
    """Download/stream public text into newline-joined shards on disk.

    Returns meta dict with shard paths and totals. Idempotent if meta exists
    with matching config (skip re-download).
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "shards_meta.json")
    want = {
        "name": name,
        "max_docs": max_docs,
        "max_chars": max_chars,
        "shard_chars": shard_chars,
        "seed": seed,
    }
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        if all(old.get(k) == v for k, v in want.items()) and old.get("shards"):
            missing = [p for p in old["shards"] if not os.path.isfile(p)]
            if not missing:
                _log(f"reuse existing shards in {out_dir} ({old.get('n_chars', 0):,} chars)")
                return old

    shards: List[str] = []
    buf: List[str] = []
    buf_n = 0
    total_chars = 0
    total_docs = 0
    shard_i = 0

    def flush() -> None:
        nonlocal buf, buf_n, shard_i
        if not buf:
            return
        path = os.path.join(out_dir, f"shard_{shard_i:05d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(buf))
            f.write("\n")
        shards.append(path)
        _log(f"wrote {path} ({buf_n:,} chars)")
        shard_i += 1
        buf = []
        buf_n = 0

    for doc in _iter_hf_texts(name, max_docs=max_docs, max_chars=max_chars):
        buf.append(doc)
        buf_n += len(doc) + 2
        total_chars += len(doc)
        total_docs += 1
        if buf_n >= shard_chars:
            flush()
    flush()

    meta = {
        **want,
        "shards": shards,
        "n_docs": total_docs,
        "n_chars": total_chars,
        "n_shards": len(shards),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    _log(f"done: {total_docs} docs, {total_chars:,} chars, {len(shards)} shards → {out_dir}")
    return meta


def iter_shard_texts(shard_dir: str) -> Iterator[str]:
    meta_path = os.path.join(shard_dir, "shards_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"no shards_meta.json in {shard_dir}")
    with open(meta_path) as f:
        meta = json.load(f)
    for path in meta["shards"]:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield f.read()


def train_tokenizer_from_shards(
    shard_dir: str,
    out_path: str,
    *,
    vocab_size: int = 4096,
    max_chars: int = 2_000_000,
) -> SubwordTokenizer:
    """Train BPE on the first ``max_chars`` of shard text and save JSON."""
    def texts():
        n = 0
        for blob in iter_shard_texts(shard_dir):
            yield blob
            n += len(blob)
            if n >= max_chars:
                break

    _log(f"training BPE vocab_size={vocab_size} on up to {max_chars:,} chars")
    tok = SubwordTokenizer.train(texts(), vocab_size=vocab_size, max_chars=max_chars)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tok.save(out_path)
    _log(f"tokenizer vocab={tok.vocab_size} → {out_path}")
    return tok


def tokenize_shards_to_memmap(
    shard_dir: str,
    tokenizer: SubwordTokenizer,
    out_path: str,
    *,
    dtype: str = "uint16",
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Encode all shards to a single memmap file + split index ranges.

    Layout: one flat 1-D array of token ids. Splits are contiguous ranges
    (train | val | test) so memmap windows never cross disk randomly for val.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # First pass: count tokens (streaming encode)
    counts: List[int] = []
    for blob in iter_shard_texts(shard_dir):
        # encode per document-ish block to keep peak memory low
        blocks = blob.split("\n\n")
        for b in blocks:
            if not b.strip():
                continue
            counts.append(len(tokenizer.encode(b, add_eos=True)))
    total = int(sum(counts))
    if total < 1000:
        raise RuntimeError(f"too few tokens ({total}) — check shards / tokenizer")

    np_dtype = np.uint16 if dtype == "uint16" else np.uint32
    if tokenizer.vocab_size >= np.iinfo(np_dtype).max:
        np_dtype = np.uint32
        dtype = "uint32"

    mmap = np.memmap(out_path, dtype=np_dtype, mode="w+", shape=(total,))
    pos = 0
    for blob in iter_shard_texts(shard_dir):
        for b in blob.split("\n\n"):
            if not b.strip():
                continue
            ids = tokenizer.encode(b, add_eos=True)
            n = len(ids)
            mmap[pos : pos + n] = np.asarray(ids, dtype=np_dtype)
            pos += n
    mmap.flush()
    assert pos == total

    # Contiguous splits: shuffle is approximate via seed offset into train;
    # for true random we'd need a second index — keep simple + reproducible.
    n_test = int(total * test_frac)
    n_val = int(total * val_frac)
    n_train = total - n_val - n_test
    meta = {
        "path": out_path,
        "dtype": dtype,
        "n_tokens": total,
        "vocab_size": tokenizer.vocab_size,
        "splits": {
            "train": [0, n_train],
            "val": [n_train, n_train + n_val],
            "test": [n_train + n_val, total],
        },
        "seed": seed,
    }
    meta_path = out_path + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    _log(
        f"memmap {out_path}: {total:,} tokens  "
        f"train={n_train:,} val={n_val:,} test={n_test:,}"
    )
    return meta


@dataclass
class MemmapTokenStore:
    """Random window sampler over a disk-backed token array."""

    path: str
    dtype: str = "uint16"
    split: str = "train"
    meta: Optional[Dict] = None

    def __post_init__(self) -> None:
        meta_path = self.path + ".meta.json"
        if self.meta is None:
            with open(meta_path) as f:
                self.meta = json.load(f)
        lo, hi = self.meta["splits"][self.split]
        self.lo = int(lo)
        self.hi = int(hi)
        np_dtype = np.uint16 if self.meta.get("dtype", self.dtype) == "uint16" else np.uint32
        self._mm = np.memmap(self.path, dtype=np_dtype, mode="r")
        self.n = self.hi - self.lo

    @property
    def n_tokens(self) -> int:
        return self.n

    def sample_batch(
        self, seq_len: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return ``(B, seq_len+1)`` long tensor of token ids."""
        max_start = max(1, self.n - seq_len - 1)
        starts = np.random.randint(0, max_start, size=batch_size)
        rows = []
        for s in starts:
            a = self.lo + int(s)
            rows.append(self._mm[a : a + seq_len + 1].astype(np.int64))
        arr = np.stack(rows, axis=0)
        return torch.from_numpy(arr).to(device)

    def eval_batches(
        self, seq_len: int, batch_size: int, max_windows: int, device: torch.device
    ) -> Iterator[torch.Tensor]:
        """Deterministic contiguous windows for evaluation."""
        step = seq_len
        starts = list(range(0, max(1, self.n - seq_len - 1), step))[:max_windows]
        for i in range(0, len(starts), batch_size):
            chunk = starts[i : i + batch_size]
            rows = [
                self._mm[self.lo + s : self.lo + s + seq_len + 1].astype(np.int64)
                for s in chunk
            ]
            if not rows:
                break
            yield torch.from_numpy(np.stack(rows)).to(device)


def prepare_public_lm_disk(
    name: str,
    work_dir: str,
    *,
    max_docs: int = 20_000,
    max_chars: int = 20_000_000,
    vocab_size: int = 4096,
    force: bool = False,
) -> Dict:
    """One-shot: stream → shards → train BPE → memmap. Returns paths dict."""
    shard_dir = os.path.join(work_dir, "shards")
    tok_path = os.path.join(work_dir, "tokenizer.json")
    mmap_path = os.path.join(work_dir, "tokens.mmap")
    done_path = os.path.join(work_dir, "prepare_done.json")

    if os.path.isfile(done_path) and not force:
        with open(done_path) as f:
            d = json.load(f)
        if os.path.isfile(d.get("mmap_path", "")) and os.path.isfile(d.get("tokenizer_path", "")):
            _log(f"reuse prepared store in {work_dir}")
            return d

    stream_public_lm_to_shards(
        name, shard_dir, max_docs=max_docs, max_chars=max_chars,
    )
    # Cap BPE sample so pure-Python training stays minutes, not hours.
    bpe_chars = min(1_000_000, max_chars)
    tok = train_tokenizer_from_shards(
        shard_dir, tok_path, vocab_size=vocab_size, max_chars=bpe_chars,
    )
    meta = tokenize_shards_to_memmap(shard_dir, tok, mmap_path)
    out = {
        "dataset": name,
        "work_dir": work_dir,
        "shard_dir": shard_dir,
        "tokenizer_path": tok_path,
        "mmap_path": mmap_path,
        "mmap_meta": meta,
        "vocab_size": tok.vocab_size,
    }
    with open(done_path, "w") as f:
        json.dump(out, f, indent=2)
    return out
