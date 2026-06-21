# Architecture

Project Positronic Brain v3 implements a spatially structured recurrent neural network whose behavior emerges from biologically motivated ingredients:

1. Explicit 3D geometry (neurons live on a cubic lattice with real coordinates).
2. A **sparse, distance-biased synaptic graph** (capped fan-in, Gaussian distance bias).
3. **Conductance-based membrane dynamics** with **Dale's law** (each source neuron is purely excitatory or purely inhibitory).
4. **Learnable synapses** — unlike v2, the connectome weights are trained end-to-end.
5. Configurable **multi-modal zones** (Visual, Auditory, Somatosensory, Memory, Emotion, Association).

> **v3.1 extension.** The brain can also ingest real images, text, audio and video
> through a per-neuron **sensory pathway** and learn live via self-supervised
> gradient replay. That subsystem is documented separately in
> [multimodal_live_learning.md](multimodal_live_learning.md).

> Design intuition (the project's guiding metaphor): a **neuron behaves like a linear integrator** (it sums currents across its membrane), while a **synapse behaves like a logistic unit** (the presynaptic firing rate is a saturating sigmoid of membrane potential). The model makes this explicit.

## High-Level Components

```
┌──────────────────────────────────────────────────────────────┐
│                       PositronicBrain                        │
│  ┌───────────────┐   ┌────────────────────────────────────┐  │
│  │ BrainConfig   │   │  Recurrent conductance dynamics    │  │
│  └───────────────┘   │  (T steps, leaky membrane)         │  │
│         │            └────────────────────────────────────┘  │
│         ▼                          │                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Sparse synaptic graph (edge_index, edge_weight)    │   │
│  │   • Gaussian distance-biased sampling                │   │
│  │   • capped fan-in k_max                               │   │
│  │   • Dale's law sign per SOURCE neuron (~20% inhib.)  │   │
│  │   • weights are LEARNABLE (nn.Parameter magnitudes)  │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                          │                         │
│         ▼                          ▼                         │
│  ┌───────────────┐        ┌──────────────────────────────┐   │
│  │ Zone input    │        │  Small readout head          │   │
│  │ injection     │        │  (Linear→Tanh→Linear→σ)      │   │
│  └───────────────┘        └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Core Modules

### `positronic_brain/zones.py`
- `Zone(IntEnum)` with six members: `VISUAL`, `AUDITORY`, `SOMATOSENSORY`, `MEMORY`, `EMOTION`, `ASSOCIATION`.
- `ZoneSpec` dataclass — name, hex color, normalized 3D seed position, modality flag.
- `DEFAULT_ZONES` — the six default specs with dark-theme-friendly colors.
- `assign_zones(grid_size, zones)` — **Voronoi** assignment: each neuron belongs to the nearest zone seed. This is deterministic and generalizes to any number of zones and any grid size.
- `get_zone_info(grid_size, zones)` — per-zone neuron counts and metadata.

### `positronic_brain/connectivity.py`
- `neuron_positions(grid_size)` → `(N, 3)` integer lattice coordinates (row-major).
- `build_sparse_graph(...)` → `(edge_index (2,E), edge_dist (E,), pos (N,3))`.
  - `edge_index[0]` = presynaptic (source) neuron `j`; `edge_index[1]` = postsynaptic (target) neuron `i`.
  - Connections are sampled with a **Gaussian bias** in Euclidean distance, inside a hard `connection_radius`, with a **capped fan-in** `k_max`. This keeps the graph sparse and locally dense, like cortex.
- `init_edge_weights(...)` → `(edge_weight (E,) signed, is_inhibitory (N,) bool)`.
  - **Dale's law**: the sign of every outgoing synapse is determined by the source neuron's type (~20% of neurons are inhibitory). Inhibitory magnitudes are scaled up (`inh_scale`) to balance the E/I budget.
- `build_distance_biased_connectivity(...)` — deprecated dense wrapper kept only for backward compatibility.

### `positronic_brain/model.py`
- `BrainConfig` dataclass — every hyperparameter in one place (grid size, radius, fan-in cap, E/I fraction, conductance reversal potentials, membrane time constant, gain, readout size, seed, zone names). `to_dict` / `from_dict` make checkpoints self-describing and tolerant of legacy keys.
- `PositronicBrain(nn.Module)` — the dynamics live in `step()` / `forward()`:
  - **Firing rate (synapse = logistic):** `r = σ(γ · (V − v_thr))`.
  - **Conductances:** `gE = Σ_{excitatory j} w·r_j`, `gI = Σ_{inhibitory j} |w|·r_j` (accumulated efficiently with `index_add_`).
  - **Synaptic current (conductance-based):** `I_syn = gE·(E_E − V) + gI·(E_I − V)`.
  - **Membrane (neuron = linear leaky integrator):** `τ_m dV/dt = −(V − E_L) + I_syn + I_ext + bias`, integrated with step size `α = dt/τ_m`.
  - **Dale's law enforcement:** the learnable parameter stores **unsigned magnitudes**; `signed_weights()` re-applies the fixed per-source sign, so training can never flip an excitatory neuron into an inhibitory one.
  - Readout pools the final firing rates through a tiny MLP (`Linear→Tanh→Linear`) and a sigmoid to produce a scalar output probability.
- `run_with_inputs(external)` returns a numpy dict: `output (B,)`, `rates (B,N)`, `trace (T+1,B,N)`, `positions (N,3)`, `zones (N,)`.
- `save` / `load` persist `state_dict` + config; `load(..., strict=False)` falls back to a fresh brain if a checkpoint is structurally incompatible.

### `positronic_brain/utils.py`
Coordinate transforms (`grid_to_index`, `index_to_grid`), stable `sigmoid`, distance helpers, and `get_device("auto"/"cpu"/"mps"/"cuda")`.

## Data Flow in a Forward Pass

1. Caller supplies a length-`Z` zone drive vector ∈ [0, 1] (one entry per zone).
2. Each zone drive is scaled by a learned `zone_gain` and the global `input_gain`, then **scattered** to every neuron belonging to that zone (spatial masking via the `zones` buffer).
3. Membrane potentials are initialized to the leak reversal `E_L`.
4. For `recurrent_steps` iterations the conductance-based update is applied; firing rates are recomputed each step.
5. The final firing rates are pooled by the readout MLP → sigmoid → scalar output probability.

Unlike v2, **the synaptic weights themselves are trained**. The fixed scientific priors are now: the 3D geometry, the sparse distance-biased *topology*, the Dale's-law *sign* of each neuron, and the conductance dynamics. The trainable parameters are the synaptic magnitudes, per-neuron biases, per-zone gains, and the readout head.

## Device Handling

- `get_device("auto")` returns `torch.device("mps")` on Apple Silicon when available, else CPU (or CUDA where present).
- `PositronicBrain(cfg, device="auto")` and `PositronicBrain.load(path, device="auto")` move all parameters and registered buffers (`edge_index`, `edge_sign`, `is_inhibitory`, `positions`, `zones`) via `.to()`.
- Saved checkpoints are device-agnostic (`state_dict` + config). `run_with_inputs()` always returns CPU numpy arrays.

For the tiny default `grid_size=4` (64 neurons) CPU latency is often lower than MPS dispatch overhead; MPS becomes worthwhile at larger grids.

## Why This Design?

- **Interpretability**: every neuron has an (x, y, z) coordinate, a zone label, and a fixed E/I type. You can point at "the visual region" or "the inhibitory cells" in the 3D view.
- **Biological analogy**: local circuits are dense, long-range fibers are sparse and distance-dependent; excitation and inhibition are segregated by cell type (Dale's law); membranes integrate conductances.
- **Learnability + scalability**: synapses are real parameters, so the network can be grown (larger `grid_size`, more zones) and trained like a small neural network.
- **Configurability**: grid size, radius, fan-in, E/I fraction, reversal potentials, zones — all first-class parameters.

## Limitations (by design)

- Rate-based, not spiking; no explicit spike timing or STDP.
- Discrete-time Euler integration of a simplified single-compartment membrane.
- The default grid is tiny. This is a research/education instrument, not a large-scale cortical simulator.

See [scientific_background.md](scientific_background.md) for the broader motivation and [zones_and_brain_regions.md](zones_and_brain_regions.md) for the spatial layout rationale.
