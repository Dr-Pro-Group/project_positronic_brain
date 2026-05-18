"""
Public multimodal dataset streaming for *initial* (pre-interaction) seeding.

These helpers stream small, capped subsets of well-known public datasets from the
Hugging Face Hub and yield **raw** samples (PIL images, caption strings, audio
waveforms) ready to feed into :meth:`positronic_brain.online.LiveBrain.perceive`.

Design goals
------------
* **Streaming** (``streaming=True``) so we never download a whole dataset — we
  pull only the first ``limit`` usable samples.
* **Generic field detection** so the same code works across datasets with
  slightly different column names (``image``/``img``, ``caption``/``text`` ...).
* **Graceful fallbacks**: a prioritized list of candidate datasets is tried in
  order; the first that streams successfully is used.

Defaults
--------
* image+text  -> ``clip-benchmark/wds_mscoco_captions`` (fallback: ``...wds_flickr30k``)
* audio+text  -> ``google/fleurs`` en_us (fallback: LibriSpeech dummy)

Nothing here is required for the brain to run; it is purely an optional
convenience for bootstrapping a model from public data before live interaction.
"""

from __future__ import annotations

import io
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

RawSample = Dict[str, object]  # modality -> raw input (PIL.Image | str | np.ndarray)

# Prioritized candidate datasets per source. Each entry is kwargs for
# ``datasets.load_dataset`` (always streamed). These are parquet/webdataset
# repos (datasets>=4 dropped support for script-based datasets).
IMAGE_TEXT_CANDIDATES: List[dict] = [
    dict(path="clip-benchmark/wds_mscoco_captions", split="test"),  # COCO captions
    dict(path="clip-benchmark/wds_flickr30k", split="test"),        # Flickr30k
]
AUDIO_TEXT_CANDIDATES: List[dict] = [
    dict(path="google/fleurs", name="en_us", split="train"),        # speech + transcript
    dict(path="hf-internal-testing/librispeech_asr_dummy", name="clean", split="validation"),
]


# --------------------------------------------------------------------------- io
def _load_dataset():
    try:
        from datasets import load_dataset  # lazy, optional dependency
        return load_dataset
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The 'datasets' library is required for public-dataset seeding.\n"
            "Install it with:  pip install datasets"
        ) from exc


def _resample(wave: np.ndarray, sr: int, target: int = 16000) -> np.ndarray:
    """Linear-interpolation resample to ``target`` Hz (no extra dependencies)."""
    wave = np.asarray(wave, dtype=np.float32).ravel()
    if sr == target or wave.size == 0:
        return wave
    n = int(round(wave.size * target / sr))
    x_old = np.linspace(0.0, 1.0, num=wave.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=max(n, 1), endpoint=False)
    return np.interp(x_new, x_old, wave).astype(np.float32)


# --------------------------------------------------------------- field detection
_TEXT_KEYS = ("caption", "captions", "text", "sentence", "sentences",
              "transcription", "raw_transcription", "english_transcription",
              "title", "txt")
_IMAGE_KEYS = ("image", "img", "jpg", "png", "webp", "jpeg", "image_0")
_AUDIO_KEYS = ("audio", "speech")


def _extract_text(sample: dict) -> Optional[str]:
    for k in _TEXT_KEYS:
        v = sample.get(k)
        if v:
            if isinstance(v, (list, tuple)):
                return str(v[0]) if len(v) else None
            return str(v)
    # webdataset captions are sometimes nested in a 'json' metadata dict
    meta = sample.get("json")
    if isinstance(meta, dict):
        for k in ("caption", "captions", "text", "title"):
            v = meta.get(k)
            if v:
                return str(v[0]) if isinstance(v, (list, tuple)) and v else str(v)
    return None


def _extract_image(sample: dict):
    from PIL import Image
    for k in _IMAGE_KEYS:
        v = sample.get(k)
        if v is None:
            continue
        if isinstance(v, Image.Image):
            return v
        if isinstance(v, dict) and v.get("bytes"):
            return Image.open(io.BytesIO(v["bytes"]))
        if isinstance(v, str):
            return v  # path / URL handled by the image encoder
    return None


def _extract_audio(sample: dict) -> Optional[np.ndarray]:
    for k in _AUDIO_KEYS:
        v = sample.get(k)
        if v is None:
            continue
        # Already-decoded array (rare with datasets>=4 without torchcodec).
        if isinstance(v, dict) and v.get("array") is not None:
            return _resample(np.asarray(v["array"], dtype=np.float32),
                             int(v.get("sampling_rate", 16000)))
        # decode=False path: raw bytes / file path decoded with soundfile.
        if isinstance(v, dict) and (v.get("bytes") or v.get("path")):
            try:
                import soundfile as sf  # decodes FLAC/WAV without torchcodec
                if v.get("bytes"):
                    data, sr = sf.read(io.BytesIO(v["bytes"]), dtype="float32")
                else:
                    data, sr = sf.read(v["path"], dtype="float32")
                if data.ndim > 1:           # stereo -> mono
                    data = data.mean(axis=1)
                return _resample(np.asarray(data, dtype=np.float32), int(sr))
            except Exception:
                return None
    return None


# --------------------------------------------------------------------- streamers
def _stream_candidates(candidates: List[dict]):
    """Yield (name, iterable_dataset) for the first candidate that loads."""
    load_dataset = _load_dataset()
    last_exc = None
    for kw in candidates:
        try:
            ds = load_dataset(streaming=True, **kw)
            return kw.get("path", "?"), ds
        except Exception as exc:  # pragma: no cover - network/dataset specific
            last_exc = exc
            print(f"[datasets] candidate {kw.get('path')!r} unavailable ({exc}); trying next.")
    raise RuntimeError(f"No image/audio dataset candidate could be loaded: {last_exc}")


def stream_image_text(limit: int, candidates: Optional[List[dict]] = None
                      ) -> Iterator[RawSample]:
    """Yield up to ``limit`` ``{"image": PIL/str, "text": str}`` samples."""
    name, ds = _stream_candidates(candidates or IMAGE_TEXT_CANDIDATES)
    print(f"[datasets] image+text source: {name}")
    n = 0
    for row in ds:
        if n >= limit:
            break
        img = _extract_image(row)
        txt = _extract_text(row)
        if img is None or not txt:
            continue
        yield {"image": img, "text": txt}
        n += 1


def stream_audio_text(limit: int, candidates: Optional[List[dict]] = None
                      ) -> Iterator[RawSample]:
    """Yield up to ``limit`` ``{"audio": np.float32(16k), "text": str}`` samples."""
    name, ds = _stream_candidates(candidates or AUDIO_TEXT_CANDIDATES)
    # Disable dataset-side audio decoding (datasets>=4 needs torchcodec); we
    # decode raw bytes ourselves with soundfile in `_extract_audio`.
    try:
        from datasets import Audio
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    print(f"[datasets] audio+text source: {name}")
    n = 0
    for row in ds:
        if n >= limit:
            break
        wav = _extract_audio(row)
        txt = _extract_text(row)
        if wav is None or wav.size == 0 or not txt:
            continue
        yield {"audio": wav, "text": txt}
        n += 1


# -----------------------------------------------------------------------------
# Large-scale / LLM-scale hooks (roadmap Phase 2).
# -----------------------------------------------------------------------------
# A curated registry of public, citable datasets to scale beyond the toy demos.
# Everything is streamed (no full download) so a 100k+ sample run only pulls what
# it consumes. These are *hooks*: pass the name to the helpers below or to
# train_language.py's --hf / --hf-chat. Honest note: with frozen perceptual
# encoders (CLIP / Wav2Vec2) the brain learns the shared associative dynamics,
# not perception — that is the intended scope.
LARGE_SCALE_DATASETS = {
    # Plain text (language modelling / perplexity).
    "openwebtext":   dict(path="Skylion007/openwebtext", split="train"),
    "wikitext103":   dict(path="wikitext", name="wikitext-103-raw-v1", split="train"),
    "c4":            dict(path="allenai/c4", name="en", split="train"),
    # Image–text (cross-modal recall at scale).
    "laion400m":     dict(path="laion/laion400m", split="train"),
    "coco":          dict(path="clip-benchmark/wds_mscoco_captions", split="test"),
    # Speech–text.
    "librispeech":   dict(path="openslr/librispeech_asr", name="clean", split="train.100"),
    "fleurs":        dict(path="google/fleurs", name="en_us", split="train"),
    # Preference data (DPO / RLHF; see positronic_brain.preference).
    "hh-rlhf":       dict(path="Anthropic/hh-rlhf", split="train"),
    "ultrafeedback": dict(path="HuggingFaceH4/ultrafeedback_binarized", split="train_prefs"),
}


def stream_text(name_or_kwargs, limit: int, text_key: str = "text") -> Iterator[str]:
    """
    Stream raw text documents from a large public corpus.

    ``name_or_kwargs`` is either a key of :data:`LARGE_SCALE_DATASETS` (e.g.
    ``"openwebtext"``, ``"wikitext103"``) or a dict of ``load_dataset`` kwargs.
    Yields up to ``limit`` non-empty document strings — feed them straight into
    ``CharTokenizer`` / a BPE tokenizer for language-model training.
    """
    load_dataset = _load_dataset()
    kw = LARGE_SCALE_DATASETS.get(name_or_kwargs, name_or_kwargs) \
        if isinstance(name_or_kwargs, str) else name_or_kwargs
    ds = load_dataset(streaming=True, **kw)
    n = 0
    for row in ds:
        if n >= limit:
            break
        txt = row.get(text_key) or _extract_text(row)
        if not txt or not str(txt).strip():
            continue
        yield str(txt)
        n += 1


def stream_preference_pairs(
    name: str = "hh-rlhf", limit: int = 2000
) -> Iterator[Tuple[str, str, str]]:
    """
    Stream ``(prompt, chosen, rejected)`` triples for DPO.

    Works with Anthropic/hh-rlhf (``chosen``/``rejected`` transcripts) and
    UltraFeedback-style binarized sets. The prompt is taken as the shared prefix
    of the two responses so it can drive the brain before scoring divergent
    continuations. Pairs feed directly into
    :func:`positronic_brain.preference.dpo_finetune`.
    """
    load_dataset = _load_dataset()
    kw = LARGE_SCALE_DATASETS.get(name, dict(path=name, split="train"))
    ds = load_dataset(streaming=True, **kw)
    n = 0
    for row in ds:
        if n >= limit:
            break
        chosen, rejected = row.get("chosen"), row.get("rejected")
        if isinstance(chosen, list):  # message-list format -> last reply text
            chosen = chosen[-1].get("content") if chosen else None
        if isinstance(rejected, list):
            rejected = rejected[-1].get("content") if rejected else None
        if not chosen or not rejected:
            continue
        # Shared prefix = the prompt both responses answer.
        k = 0
        for a, b in zip(str(chosen), str(rejected)):
            if a != b:
                break
            k += 1
        prompt = str(chosen)[:k]
        yield prompt, str(chosen)[k:], str(rejected)[k:]
        n += 1
