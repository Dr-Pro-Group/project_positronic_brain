"""
Project Positronic Brain v3
A biomimetic, learnable, sparse 3D neuronal network simulator.

This package provides a configurable, scalable 3D recurrent brain model with:
  * sparse distance-biased graph connectivity (O(E), not O(N^2)),
  * conductance-based, leaky-integrator membrane dynamics ("neuron = linear"),
  * logistic firing-rate synaptic transfer ("synapse = logistic"),
  * learnable per-edge synapses obeying Dale's law (E/I balance),
  * configurable multi-modal zones (Visual, Auditory, Somatosensory, Memory,
    Emotion, Association).

Optional GPU acceleration is supported on Apple Silicon MacBook Pros via
Metal (MPS). Use device="auto" (default), "mps", or "cpu" when constructing
or loading a PositronicBrain. On non-Apple hardware "auto" falls back to CPU.
"""

from .model import PositronicBrain, BrainConfig
from .zones import (
    Zone,
    ZoneSpec,
    ZoneInfo,
    DEFAULT_ZONES,
    ZONE_NAMES,
    ZONE_COLORS,
    zone_names,
    zone_colors,
    num_zones,
    get_zone_for_position,
    get_zone_info,
    assign_zones,
)
from .connectivity import (
    build_sparse_graph,
    init_edge_weights,
    neuron_positions,
    build_distance_biased_connectivity,
)
from .utils import sigmoid, normalize, grid_to_index, index_to_grid, compute_distance, get_device
from .encoders import (
    Encoder,
    FallbackEncoder,
    ClipImageEncoder,
    ClipTextEncoder,
    AudioEncoder,
    VideoEncoder,
    build_encoder,
    build_encoders,
    embedding_dims,
    l2_normalize,
    DEFAULT_EMBED_DIMS,
)
from .online import (
    ReplayBuffer,
    OnlineLearner,
    OnlineConfig,
    LiveBrain,
)
from . import datasets
from .datasets import stream_image_text, stream_audio_text
from .language import BrainLanguageModel, CharTokenizer, LMConfig
from . import corpus
from .corpus import build_conversational_corpus, load_corpus
from . import repro
from .repro import seed_everything, run_metadata
from . import preference
from .preference import dpo_loss, dpo_finetune, sequence_logprob, DPOConfig, BrainPolicy

__version__ = "3.3.0"
__license__ = "MIT"
__author__ = "Project Positronic Brain Contributors"
__all__ = [
    "PositronicBrain",
    "BrainConfig",
    "Zone",
    "ZoneSpec",
    "ZoneInfo",
    "DEFAULT_ZONES",
    "ZONE_NAMES",
    "ZONE_COLORS",
    "zone_names",
    "zone_colors",
    "num_zones",
    "get_zone_for_position",
    "get_zone_info",
    "assign_zones",
    "build_sparse_graph",
    "init_edge_weights",
    "neuron_positions",
    "build_distance_biased_connectivity",
    "sigmoid",
    "normalize",
    "grid_to_index",
    "index_to_grid",
    "compute_distance",
    "get_device",
    # Multimodal encoders
    "Encoder",
    "FallbackEncoder",
    "ClipImageEncoder",
    "ClipTextEncoder",
    "AudioEncoder",
    "VideoEncoder",
    "build_encoder",
    "build_encoders",
    "embedding_dims",
    "l2_normalize",
    "DEFAULT_EMBED_DIMS",
    # Online / live learning
    "ReplayBuffer",
    "OnlineLearner",
    "OnlineConfig",
    "LiveBrain",
    # Public-dataset seeding
    "datasets",
    "stream_image_text",
    "stream_audio_text",
    # Generative language model (the talking brain)
    "BrainLanguageModel",
    "CharTokenizer",
    "LMConfig",
    "corpus",
    "build_conversational_corpus",
    "load_corpus",
    # Reproducibility
    "repro",
    "seed_everything",
    "run_metadata",
    # Preference optimisation / RL hooks
    "preference",
    "dpo_loss",
    "dpo_finetune",
    "sequence_logprob",
    "DPOConfig",
    "BrainPolicy",
]
