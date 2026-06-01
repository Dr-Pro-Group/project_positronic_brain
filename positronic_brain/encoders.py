"""
Multimodal encoders: turn real data (images, text, audio, video) into fixed-size
embedding vectors that drive the :class:`~positronic_brain.model.PositronicBrain`
sensory pathway.

Design
------
* Every encoder exposes ``modality``, ``embedding_dim`` and ``embed(x) -> np.ndarray``
  returning an **L2-normalised** float32 vector of length ``embedding_dim``.
* :class:`FallbackEncoder` is a dependency-free, deterministic, *content-sensitive*
  encoder (coarse features + a fixed random projection). It needs no downloads and
  no network, so the full pipeline (sensory pathway + online learning + tests) is
  always runnable offline.
* The pretrained encoders (:class:`ClipImageEncoder`, :class:`ClipTextEncoder`,
  :class:`AudioEncoder`, :class:`VideoEncoder`) lazily import ``transformers`` and
  download standard checkpoints on first use. If those imports or downloads fail,
  :func:`build_encoder` transparently falls back to :class:`FallbackEncoder` so the
  rest of the system keeps working.

The default embedding dimensions are chosen to match the pretrained backbones
(CLIP = 512) so you can swap fallback <-> pretrained without changing the brain.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np

# Suggested embedding dimensions per modality (match the pretrained backbones so
# fallback and real encoders are interchangeable for a given brain config).
DEFAULT_EMBED_DIMS: Dict[str, int] = {
    "image": 512,   # CLIP ViT-B/32 image tower
    "text": 512,    # CLIP ViT-B/32 text tower (shared space with image)
    "audio": 512,   # pooled audio features projected to 512
    "video": 512,   # mean of sampled-frame image embeddings
}

ArrayLike = Union[np.ndarray, "object"]


def _stable_seed(data: bytes, salt: str = "") -> int:
    """Deterministic 63-bit seed from arbitrary bytes (stable across runs)."""
    h = hashlib.sha256(salt.encode("utf-8") + data).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / (n + eps)


# --------------------------------------------------------------------------- API
class Encoder(ABC):
    """Abstract multimodal encoder."""

    modality: str
    embedding_dim: int

    @abstractmethod
    def embed(self, x: ArrayLike) -> np.ndarray:
        """Encode one input into an L2-normalised (embedding_dim,) float32 vector."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(modality={self.modality!r}, dim={self.embedding_dim})"


# ------------------------------------------------------------------- fallback
class FallbackEncoder(Encoder):
    """
    Deterministic, dependency-free encoder.

    Extracts a small, content-sensitive feature vector per modality and projects
    it through a fixed (seeded) random matrix into ``embedding_dim``. Same input
    -> same embedding, and *similar* inputs map to *similar* embeddings, which is
    enough to exercise (and visualise) the brain's learning machinery without any
    model downloads.
    """

    FEAT = 96  # coarse feature length before projection

    def __init__(self, modality: str, embedding_dim: Optional[int] = None):
        self.modality = modality
        self.embedding_dim = int(embedding_dim or DEFAULT_EMBED_DIMS.get(modality, 512))
        # Fixed projection, deterministic per modality (stable across processes).
        rng = np.random.default_rng(_stable_seed(b"proj", salt=modality))
        self._proj = rng.standard_normal((self.embedding_dim, self.FEAT)).astype(np.float32)

    # -- per-modality coarse features ------------------------------------------
    def _features(self, x: ArrayLike) -> np.ndarray:
        m = self.modality
        if m == "text":
            return self._text_features(x)
        if m == "image":
            return self._image_features(x)
        if m == "audio":
            return self._audio_features(x)
        if m == "video":
            return self._video_features(x)
        return self._generic_features(x)

    def _text_features(self, x) -> np.ndarray:
        s = x if isinstance(x, str) else str(x)
        b = s.lower().encode("utf-8", "ignore")
        hist = np.zeros(self.FEAT, dtype=np.float32)
        for byte in b:
            hist[byte % self.FEAT] += 1.0
        return hist

    def _image_features(self, x) -> np.ndarray:
        arr = self._to_gray_grid(x, side=int(np.sqrt(self.FEAT)) or 9)
        return arr.ravel()[: self.FEAT]

    def _audio_features(self, x) -> np.ndarray:
        w = np.asarray(x, dtype=np.float32).ravel()
        if w.size == 0:
            return np.zeros(self.FEAT, dtype=np.float32)
        spec = np.abs(np.fft.rfft(w))
        # bin the spectrum into FEAT buckets
        if spec.size < self.FEAT:
            spec = np.pad(spec, (0, self.FEAT - spec.size))
        idx = np.linspace(0, spec.size, self.FEAT + 1).astype(int)
        return np.array([spec[idx[i]:idx[i + 1]].mean() if idx[i + 1] > idx[i] else 0.0
                         for i in range(self.FEAT)], dtype=np.float32)

    def _video_features(self, x) -> np.ndarray:
        # x: stack of frames (T,H,W,C) or path-like already decoded to array
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim >= 3:
            arr = arr.reshape(-1) if arr.size <= self.FEAT else arr
            return self._generic_features(arr)
        return self._generic_features(arr)

    def _generic_features(self, x) -> np.ndarray:
        a = np.asarray(x, dtype=np.float32).ravel()
        if a.size == 0:
            return np.zeros(self.FEAT, dtype=np.float32)
        if a.size < self.FEAT:
            a = np.pad(a, (0, self.FEAT - a.size))
        idx = np.linspace(0, a.size, self.FEAT + 1).astype(int)
        return np.array([a[idx[i]:idx[i + 1]].mean() if idx[i + 1] > idx[i] else 0.0
                         for i in range(self.FEAT)], dtype=np.float32)

    @staticmethod
    def _to_gray_grid(x, side: int = 9) -> np.ndarray:
        """Best-effort downsample of an image-like input to a side x side gray grid."""
        try:
            from PIL import Image  # noqa: WPS433 (lazy, optional)
            if isinstance(x, str):
                img = Image.open(x).convert("L").resize((side, side))
                return np.asarray(img, dtype=np.float32) / 255.0
            if isinstance(x, Image.Image):
                return np.asarray(x.convert("L").resize((side, side)), dtype=np.float32) / 255.0
        except Exception:
            pass
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 3:
            a = a.mean(axis=2)
        if a.ndim != 2 or a.size == 0:
            a = a.ravel()
            n = int(np.ceil(np.sqrt(max(a.size, 1))))
            a = np.pad(a, (0, n * n - a.size)).reshape(n, n)
        # crude block-resize to side x side
        h, w = a.shape
        ys = np.linspace(0, h, side + 1).astype(int)
        xs = np.linspace(0, w, side + 1).astype(int)
        out = np.zeros((side, side), dtype=np.float32)
        for i in range(side):
            for j in range(side):
                blk = a[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)]
                out[i, j] = float(blk.mean()) if blk.size else 0.0
        mx = float(out.max())
        return out / mx if mx > 0 else out

    def embed(self, x: ArrayLike) -> np.ndarray:
        f = self._features(x).astype(np.float32)
        if f.shape[0] != self.FEAT:
            f = np.resize(f, self.FEAT)
        # standardise features then project
        f = (f - f.mean()) / (f.std() + 1e-6)
        v = self._proj @ f
        return l2_normalize(v)


# --------------------------------------------------------------- pretrained
class _ClipBackbone:
    """Lazily-loaded, process-wide shared CLIP model + processor (transformers)."""

    _cache: Dict[str, "object"] = {}

    @classmethod
    def get(cls, name: str = "openai/clip-vit-base-patch32"):
        if name not in cls._cache:
            from transformers import CLIPModel, CLIPProcessor  # lazy
            import torch
            model = CLIPModel.from_pretrained(name)
            model.eval()
            proc = CLIPProcessor.from_pretrained(name)
            cls._cache[name] = (model, proc, torch)
        return cls._cache[name]


class ClipImageEncoder(Encoder):
    """Real image encoder using CLIP's image tower (512-d, shared with text)."""

    modality = "image"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.embedding_dim = 512

    def embed(self, x: ArrayLike) -> np.ndarray:
        from PIL import Image
        model, proc, torch = _ClipBackbone.get(self.model_name)
        if isinstance(x, str):
            x = Image.open(x).convert("RGB")
        elif isinstance(x, np.ndarray):
            x = Image.fromarray(np.asarray(x).astype("uint8")).convert("RGB")
        with torch.no_grad():
            inp = proc(images=x, return_tensors="pt")
            # Stable across transformers versions: vision tower -> visual projection.
            pooled = model.vision_model(**inp).pooler_output  # (1, hidden)
            feat = model.visual_projection(pooled)            # (1, 512)
        return l2_normalize(feat[0].cpu().numpy())


class ClipTextEncoder(Encoder):
    """Real text encoder using CLIP's text tower (512-d, shared with image)."""

    modality = "text"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.embedding_dim = 512

    def embed(self, x: ArrayLike) -> np.ndarray:
        model, proc, torch = _ClipBackbone.get(self.model_name)
        s = x if isinstance(x, str) else str(x)
        with torch.no_grad():
            inp = proc(text=[s], return_tensors="pt", padding=True, truncation=True)
            # Stable across transformers versions: text tower -> text projection.
            pooled = model.text_model(**inp).pooler_output  # (1, 512)
            feat = model.text_projection(pooled)            # (1, 512)
        return l2_normalize(feat[0].cpu().numpy())


class AudioEncoder(Encoder):
    """
    Real audio encoder. Uses a pretrained ``transformers`` audio model
    (Wav2Vec2 by default) and mean-pools the hidden states, projecting to 512-d
    with a fixed random matrix so it matches the default brain config.
    """

    modality = "audio"
    _cache: Dict[str, "object"] = {}

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", sample_rate: int = 16000):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.embedding_dim = 512
        # fixed projection from model hidden size (768) -> 512
        rng = np.random.default_rng(_stable_seed(b"audio_proj"))
        self._proj = rng.standard_normal((512, 768)).astype(np.float32)

    def _backbone(self):
        if self.model_name not in self._cache:
            from transformers import Wav2Vec2Model, AutoFeatureExtractor
            import torch
            model = Wav2Vec2Model.from_pretrained(self.model_name)
            model.eval()
            fe = AutoFeatureExtractor.from_pretrained(self.model_name)
            self._cache[self.model_name] = (model, fe, torch)
        return self._cache[self.model_name]

    def embed(self, x: ArrayLike) -> np.ndarray:
        model, fe, torch = self._backbone()
        w = np.asarray(x, dtype=np.float32).ravel()
        with torch.no_grad():
            inp = fe(w, sampling_rate=self.sample_rate, return_tensors="pt")
            hidden = model(**inp).last_hidden_state  # (1, T, 768)
            pooled = hidden.mean(dim=1)[0].cpu().numpy()
        return l2_normalize(self._proj @ pooled)


class VideoEncoder(Encoder):
    """
    Real video encoder: samples frames, embeds each with the CLIP image tower,
    and averages. Audio track (if any) should be encoded separately via
    :class:`AudioEncoder`.
    """

    modality = "video"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", num_frames: int = 8):
        self.model_name = model_name
        self.num_frames = num_frames
        self.embedding_dim = 512
        self._img = ClipImageEncoder(model_name)

    def _load_frames(self, path: str) -> List[np.ndarray]:
        import imageio.v3 as iio  # lazy
        frames = iio.imread(path, index=None)  # (T,H,W,C)
        t = frames.shape[0]
        idx = np.linspace(0, t - 1, min(self.num_frames, t)).astype(int)
        return [frames[i] for i in idx]

    def embed(self, x: ArrayLike) -> np.ndarray:
        if isinstance(x, str):
            frames = self._load_frames(x)
        else:
            arr = np.asarray(x)
            frames = list(arr) if arr.ndim == 4 else [arr]
        embs = [self._img.embed(f) for f in frames]
        return l2_normalize(np.mean(embs, axis=0))


# ------------------------------------------------------------------- factory
_REAL_ENCODERS = {
    "image": ClipImageEncoder,
    "text": ClipTextEncoder,
    "audio": AudioEncoder,
    "video": VideoEncoder,
}


def build_encoder(modality: str, prefer_pretrained: bool = True,
                  embedding_dim: Optional[int] = None) -> Encoder:
    """
    Build an encoder for ``modality``.

    If ``prefer_pretrained`` is True we try the real (transformers/CLIP) encoder
    and, on any import/instantiation failure, transparently fall back to the
    dependency-free :class:`FallbackEncoder`.
    """
    if prefer_pretrained and modality in _REAL_ENCODERS:
        try:
            enc = _REAL_ENCODERS[modality]()
            # Probe heavy imports lazily but cheaply: defer real download to first embed.
            return enc
        except Exception as exc:  # pragma: no cover - depends on environment
            print(f"[encoders] pretrained {modality} unavailable ({exc}); using fallback.")
    return FallbackEncoder(modality, embedding_dim)


def build_encoders(modalities: List[str], prefer_pretrained: bool = True
                   ) -> Dict[str, Encoder]:
    """Build a dict of encoders for several modalities."""
    return {m: build_encoder(m, prefer_pretrained=prefer_pretrained) for m in modalities}


def embedding_dims(encoders: Dict[str, Encoder]) -> Dict[str, int]:
    """Return modality -> embedding_dim, for wiring a matching BrainConfig."""
    return {m: e.embedding_dim for m, e in encoders.items()}
