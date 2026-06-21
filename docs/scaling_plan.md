# Scaling Plan: Brain Zones Talking Together via Learned Neuro-Pathways

**Status:** technical plan / design document. Ambitious but grounded in the
current implementation. Nothing here is a reported result; every cost is an
explicit, order-of-magnitude estimate clearly labelled as such.

This document specifies how to take Positronic Brain from a single
32,768-neuron cube trained on a laptop to a **modular system of 100k–500k+
neurons** organised into specialised, communicating brain zones, connected by
**learned long-range neuro-pathways** whose formation is biologically motivated
(distance prior + activity-dependent plasticity). It is the natural successor to
the connectivity work already in `positronic_brain/connectivity.py` and the
zone scaffolding in `positronic_brain/zones.py`.

---

## 1. Where we are, precisely

The current architecture is a **single cubic lattice** of `N = G³` neurons.
Connectivity is one directed sparse graph `edge_index ∈ ℤ^{2×E}` built by
`build_sparse_graph` with a distance-biased, fan-in-capped rule (`k_max=16`,
radius `r≈2.6`), giving `E ≈ N·k`. Membrane dynamics are an O(E) `index_add`
scatter per integration step (`PositronicBrain.step`). "Zones" (`zones.py`) are
a **Voronoi labelling of one shared graph** — functional tags, not separate
populations. There is exactly one recurrent state vector `V ∈ ℝ^{B×N}`.

Two facts set the agenda:

1. **Construction is already O(N·k)** and builds 110k-neuron graphs in ≈5 s, so
   memory of the *graph* is not the bottleneck.
2. **Training is BPTT-compute-bound.** Each character costs `inner_steps`
   sequential scatters over `E` edges, and the backward graph stores activations
   for every step. This caps the *trainable* size near 32k neurons on 16 GB.

So scaling is not a connectivity problem; it is a **training-compute and
organisation** problem. The plan attacks both: intra-area training efficiency
first, then a modular multi-area design that keeps each area small enough to
train while letting many areas compose into a large brain.

---

## 2. Target architecture: areas, columns, pathways

We replace the single monolithic cube with a **graph of brain areas**.

```
            ┌─────────── long-range pathways (sparse, learned, gated) ───────────┐
            │                                                                     │
   ┌────────▼────────┐     ┌─────────────────┐     ┌─────────────────┐   ┌────────▼────────┐
   │  Sensory area   │     │  Association A   │     │  Association B   │   │   Motor / LM    │
   │  (Visual/Aud.)  │────▶│  (integration)   │────▶│  (working mem.)  │──▶│   read-out      │
   │  cube G_s³      │     │  cube G_a³       │     │  cube G_b³       │   │  cube G_m³      │
   └─────────────────┘     └─────────────────┘     └─────────────────┘   └─────────────────┘
   dense *local* graph      dense *local* graph      dense *local* graph    dense *local* graph
   (intra-area, O(N_s·k))                                                  (intra-area, O(N_m·k))
```

Three structural levels, each with a biological referent:

- **Columns (microcircuits).** Within an area, the existing distance-biased local
  graph already produces compact neighbourhoods. We make this explicit by tiling
  each area cube into **cortical columns** (e.g. 6×6×6 sub-blocks) with dense
  intra-column wiring and sparse inter-column wiring — the canonical cortical
  motif. Columns are the unit of locality and of the gradient-checkpointing
  boundary (§4).
- **Areas (a cube each).** Each area is an independent `PositronicBrain`-style
  cube with its own `edge_index`, its own E/I population (Dale's law per area),
  and its own membrane state `V_area`. Areas are sized to *individually* fit the
  per-area training budget (≈30k–60k neurons), so the system scales by **adding
  areas**, not by growing one cube.
- **Pathways (between areas).** A separate, much sparser inter-area edge set
  carries long-range communication. These are the **neuro-pathways** and are the
  heart of this plan (§3).

Total neuron count is `Σ_a N_a`. Ten 50k-neuron areas is a 500k-neuron brain in
which **no single backprop graph ever exceeds 50k neurons**.

### 2.1 Data structure

Implementation is a strict generalisation of today's code, so most of
`model.py` is reused per area:

```python
@dataclass
class AreaSpec:
    name: str
    grid_size: int                 # area cube side (N_a = grid_size**3)
    frac_inhibitory: float = 0.2   # Dale's law is per-area
    role: str = "association"      # sensory | association | memory | motor

@dataclass
class PathwaySpec:
    src_area: str
    dst_area: str
    k_path: int = 4                # fan-in per target neuron from the source area
    plasticity: str = "hebbian"    # static | hebbian | structural
    gated: bool = True             # MoE-style routing gate (§5)
```

A `ModularBrain` owns `{area_name: PositronicBrain}` plus a list of
`Pathway` objects, each holding its own `edge_index_path ∈ ℤ^{2×E_path}`
mapping source-area neuron indices to target-area neuron indices, with its own
learnable weights. The forward step becomes: (i) advance every area's local
dynamics one step; (ii) for each pathway, scatter source firing rates into the
target area's external current `I_ext`. Both are O(E) `index_add`s — the same
primitive already in `step()`.

---

## 3. Biologically-motivated pathway formation

Pathways must not be hand-wired. They should **grow and prune** from a spatial
prior under activity, mirroring how cortical long-range tracts develop
(distance-penalised but refined by correlated firing). Three escalating
mechanisms, each a drop-in for `PathwaySpec.plasticity`:

### 3.1 Distance prior (static) — the baseline

Place each area at a centroid in a shared global space. A candidate pathway
`j(src) → i(dst)` is sampled with probability falling off with **inter-area
centroid distance** plus intra-area position, exactly reusing the Gaussian rule
of `build_sparse_graph` but across areas:

$$ p(j\to i) \propto \exp\!\Big(-\frac{\lVert \mathrm{pos}_i^{\,\text{global}} - \mathrm{pos}_j^{\,\text{global}}\rVert^2}{2\sigma_{\text{path}}^2}\Big). $$

This gives a fixed, sparse, distance-respecting wiring (small-world by
construction). It is the cheapest option and the control condition for ablations.

### 3.2 Activity-dependent / Hebbian refinement

Pathway **weights** adapt with a local correlation rule layered on top of BPTT,
so a synapse strengthens when pre- and post-synaptic areas fire together:

$$ \Delta w_{ij} \;=\; \eta\,\big(r_j^{\text{src}}\, r_i^{\text{dst}} - \lambda\, w_{ij}\big), $$

a Hebbian term with weight decay (Oja-style normalisation to prevent runaway).
This runs as an **auxiliary, label-free update** during or between BPTT steps,
giving pathways a self-organising component that gradient descent on the task
loss does not directly supervise. It is the mechanism most aligned with the
project's "keeps learning while you use it" online-learning story.

### 3.3 Structural plasticity (grow/prune)

The strongest form changes **topology**, not just weights. Periodically:

- **Prune** pathway edges whose `|w_ij|` falls below a threshold (synaptic
  elimination).
- **Grow** new candidate edges between source/target neurons with high recent
  activity correlation but no current connection (activity-dependent
  synaptogenesis), sampled under the §3.1 distance prior so growth stays local-ish.

This keeps `E_path` roughly constant (a fixed wiring budget) while letting the
*pattern* of long-range connectivity be discovered. It is implemented as an
edge-list rewrite between epochs — cheap, because pathways are sparse — and is
the direct analogue of `connectivity.py`'s rebuild, now driven by activity.

> **Honesty note.** Hebbian + structural plasticity interacting with BPTT can be
> unstable and is an open research question, not a solved feature. The plan is to
> ship §3.1 first (a clean, trainable baseline), then add §3.2/§3.3 behind flags
> and ablate them. We expect to *report whether* activity-dependent pathways beat
> a static distance prior — a publishable result either way.

---

## 4. Efficient training strategies

The system is designed so that **no backward graph is ever larger than one
area**, and several orthogonal techniques cut the remaining cost. The first two
are already implemented (`positronic_brain/language.py`,
`train_language.py`); the rest are the scaling roadmap.

| Technique | Status | What it buys | Mechanism |
|---|---|---|---|
| **Truncated BPTT** | ✅ implemented (`tbptt_step`, `--tbptt-chunk`) | sequence length ⟂ memory | detach `V` every `chunk` chars |
| **Gradient checkpointing** | ✅ implemented (`grad_checkpoint`, `--grad-checkpoint`) | ~½–¼ activation memory | recompute the `inner_steps` reverberation in backward |
| **Mixed precision (bf16/AMP)** | planned | ~2× throughput, ½ memory | `autocast` around `step()`; keep scatters in fp32 |
| **CUDA + segment-reduction kernel** | planned | big-graph throughput | replace `index_add` with `torch_scatter`/Triton segment-sum |
| **Per-area data/model parallel** | planned | areas on separate GPUs | each area is an independent module; pathways are the only cross-GPU traffic |
| **Sparse gradients** | planned | update only active edges | gate Hebbian/structural updates to recently-active edges |
| **MoE-style routing** | planned | compute ∝ active areas | §5 |

### 4.1 Per-area parallelism is the key unlock

Because areas communicate only through sparse pathways, a 500k-neuron brain maps
naturally onto multiple devices: put each area (or a few) on its own GPU and
exchange only the pathway messages (source firing rates at the pathway
endpoints — a few thousand floats per step, not the full state). This is
**pipeline + tensor parallelism with a tiny communication surface**, which is
exactly the property that makes the modular design scale where a monolithic cube
would not.

### 4.2 Why the sequential cost remains the honest limiter

BPTT over `inner_steps × T` is inherently sequential per area; TBPTT and
checkpointing reduce memory and enable longer sequences but do **not** remove the
sequential dependency. Utilisation will be lower than a transformer's. We state
this as a standing limitation and target *throughput per dollar at fixed
quality*, not peak FLOPs.

---

## 5. Routing between zones (mixture-of-experts analogy)

Not every area needs to fire on every token. A lightweight **router** (a small
learned gate conditioned on the association area's state) can decide which
pathways are *active* this step, so compute scales with the number of *engaged*
areas rather than all of them — the same economics as a sparse MoE, but the
"experts" are biological areas and the routing is over **physical pathways**.
Concretely, each gated pathway multiplies its scattered current by a
`sigmoid(gate)` that the router can drive toward 0 to silence a tract. This also
gives a clean substrate for studying **task-dependent functional connectivity**
(which pathways light up for which inputs), connecting back to the neuroscience
framing.

---

## 6. Phased delivery

| Phase | Neurons | New capability | Hardware | Acceptance |
|---|---|---|---|---|
| **S0 Single-area efficiency** | ≤110k (1 cube) | AMP + CUDA + scatter kernel on today's model | 1× A100 40 GB | G=48 trains 50k steps in hours; MPS↔CUDA parity on a tiny config |
| **S1 Two areas + static pathway** | ~60k (2×30k) | `ModularBrain`, one §3.1 pathway, per-area `V` | 1× A100 80 GB | two areas train jointly; pathway carries gradient; beats a single same-size cube on a memory task |
| **S2 Multi-area + columns** | ~150k (4–6 areas) | columnar tiling, per-area Dale, pathway graph | 2–4× A100/H100 | scaling plot (areas vs throughput); modularity/small-worldness measured |
| **S3 Activity-dependent pathways** | ~300k (8 areas) | §3.2 Hebbian + §3.3 structural plasticity behind flags | 4–8× H100 | ablation: learned vs static pathways on held-out perplexity + retrieval@k |
| **S4 Routing + scale-out** | 500k+ (10+ areas) | §5 gated routing, multi-GPU pipeline | 8–16× H100 (or rented) | compute ∝ engaged areas; end-to-end multimodal demo |
| **S5 Neuromorphic mapping** | research | event-driven readout → Loihi/SpiNNaker mapping study | neuromorphic access | SynOps measured on hardware, not proxy |

Each phase is independently publishable and degrades gracefully: if S3's
plasticity proves unstable, S2 (static modular brain + columns + topology
analysis) is already a complete result.

---

## 7. Hardware requirements and cost estimates

**These are order-of-magnitude planning numbers, not measurements.** They use:
N neurons, `E ≈ 16N` edges, `inner_steps=3`, sequence chunk `T=256` with TBPTT,
batch `B`, and the empirical observation that one O(E) scatter step on an A100 is
roughly `~E/10⁹` seconds at fp16 for moderate `B`. Treat them as ±1 order of
magnitude.

**Per-step compute** scales as `O(E · inner_steps · B)` forward, ~3× that for
forward+backward. For a 50k-neuron area, `E ≈ 8×10⁵`; a full 256-char TBPTT
window with `B=16` is `≈ 256 × 3 × 8×10⁵ × 16 ≈ 10¹⁰` scatter-MACs per step —
seconds-scale on an A100, which is why per-area sizes are capped near 30k–60k.

| Configuration | Neurons | Edges | Est. GPU memory (train) | Est. hardware | Est. cloud cost to a first result |
|---|---:|---:|---|---|---|
| Today (laptop) | 32,768 | 5×10⁵ | ~14 GB unified | M1 Pro 16 GB | $0 (own laptop) |
| S0 single area | 110,592 | 1.8×10⁶ | ~30–40 GB | 1× A100 40 GB | ~$1–3/hr × ~50–150 hr ≈ **$100–400** |
| S2 modular | ~150,000 | 2.4×10⁶ | ~2× 40 GB | 2–4× A100 | ~$5–12/hr × ~100–300 hr ≈ **$1k–3k** |
| S3 plastic | ~300,000 | 4.8×10⁶ | ~4× 80 GB | 4–8× H100 | ~$15–35/hr × ~200–500 hr ≈ **$4k–15k** |
| S4 routed 500k | 500,000+ | 8×10⁶+ | ~8–16× 80 GB | 8–16× H100 | ~$30–70/hr × ~300–800 hr ≈ **$10k–50k** |

Notes that keep this honest: (i) recurrent BPTT under-utilises GPUs, so wall-clock
(and therefore cost) can be **2–5× worse** than a transformer at equal FLOPs;
(ii) rented spot/interruptible instances roughly halve the figures; (iii) the
modular design means S2–S4 can be reached incrementally — each added area is a
bounded, predictable increment, not a re-architecture.

---

## 8. Risks and how the design absorbs them

- **Sequential BPTT cost (high, structural).** Mitigated, not eliminated, by
  TBPTT + checkpointing + per-area parallelism. Stated as a limitation; we target
  throughput-per-dollar at fixed quality.
- **Plasticity × BPTT instability (high, research).** Shipped behind flags after
  a static-pathway baseline; ablated, not assumed. A negative result is still a
  result.
- **Pathway gradient bottlenecks (medium).** Sparse long-range edges may
  vanishing-gradient; mitigated by the Hebbian auxiliary signal that trains
  pathways without relying solely on backprop through them.
- **Communication overhead across GPUs (medium).** Bounded by design: only
  pathway-endpoint firing rates cross devices, a tiny payload versus full state.
- **Conductance may not pay off at scale (open).** The matched experiment shows
  conductance currently *costs* perplexity at small scale (see
  `literature_gap.md` and the paper). Scaling does not assume it helps; every
  phase keeps the `use_conductance` switch so the question stays empirical.

---

## 9. Concrete next implementation step

The smallest credible increment is **S1**: a `ModularBrain` wrapping two existing
`PositronicBrain` cubes plus one static distance-prior pathway, trained on a
memory/copy task where long-range communication is provably required. It reuses
`build_sparse_graph` (for the pathway edge list), `step()` (per area), and
`tbptt_step` (training), so it is largely composition of code that already
exists and is tested. That single experiment validates the entire modular thesis
before any large spend.
