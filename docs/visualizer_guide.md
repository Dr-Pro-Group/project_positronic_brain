# Visualizer Guide

Project Positronic Brain v3 ships with two complementary visualizers:

- **`app.py`** — an interactive Streamlit + Plotly 3D explorer (drag, zoom, scrub through time).
- **`visualize.py`** — a matplotlib CLI for publication-quality snapshots (PNG) and animations (GIF).

Both are **zone-count-agnostic**: they read the zones, positions, and E/I types directly off the loaded brain, so they work unchanged if you add or remove zones.

## The Streamlit App

### Launch

```bash
source .venv/bin/activate
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Sidebar

- **Model checkpoint**: path to a `.pt` file. If it exists it is loaded (`strict=False`, so a structurally different checkpoint gracefully falls back to a fresh model); otherwise a fresh brain is created.
- **Zone input drive**: one slider **per zone** (generated dynamically from the brain's zone list), each injecting external current into that zone's neurons.
- **Preset scenarios**: one-click input combinations — Threat, Calm memory, Visual, Painful touch, Cross-modal, Internal, Baseline — each described in plain language.
- **Compute device**: `auto` / `cpu` / `mps`. On Apple Silicon, `auto` enables Metal acceleration; the active device is shown under the metrics.
- **Visualization toggles**: show/hide synaptic lines and how many to draw.

### Main Tabs

1. **Live Brain** — interactive Plotly 3D scene. Neurons are spheres whose **size and opacity** scale with firing rate; **color = zone**. Synaptic lines are drawn for the strongest active edges, **split between excitatory (warm amber) and inhibitory (cool blue)** with brightness normalized *per sign group* — this prevents the ~4× inhibitory magnitude from visually swamping excitatory fibers. A time slider scrubs the recurrent trajectory.
2. **Activity Traces** — mean firing rate per zone across timesteps, revealing each zone's "personality" (sensory zones rise fast, internal zones linger), plus a final-state firing-rate histogram.
3. **Connectivity** — summary statistics of the sparse graph: number of synapses, E/I counts, distance-vs-magnitude structure.
4. **Insights** — a short scientific interpretation and the exact `BrainConfig` that produced the current brain.

### Recommended First Tour (5 minutes)

1. Launch the app and load the **Threat** preset, then **Run**.
2. In Live Brain, drag the time slider from t=0 to the end. The sensory regions ignite first, then activity spreads toward the Association hub.
3. Enable synaptic lines: warm lines are excitatory, cool lines inhibitory.
4. Switch to Activity Traces — note how internal zones (Memory/Emotion) decay more slowly than sensory ones.
5. Load **Calm memory recall** and re-run; the internal regions dominate.
6. Try **Painful touch** (Somatosensory + Emotion) and compare the readout to Somatosensory alone.

## The matplotlib CLI (`visualize.py`)

Generate a static 3-panel snapshot or an animated GIF of the recurrent dynamics:

```bash
# Snapshot of a preset, saved to assets/
python visualize.py --preset cross --save assets/cross_test.png

# Animated GIF over the full trajectory
python visualize.py --preset threat --animate --save assets/threat_test.gif

# Custom per-zone input (name=value pairs)
python visualize.py --input "Visual=0.4,Memory=0.7,Association=0.55" --timesteps 12
```

Flags: `--preset`, `--input`, `--timesteps`, `--animate`, `--save`, `--device`.

Per-point alpha is baked into RGBA via `matplotlib.colors.to_rgba`, so opacity tracks each neuron's firing rate correctly. At t=0 the brain is quiet; as the trajectory unfolds, the driven zones ignite in their characteristic colors.

## Common "Is this a bug?" Observations

- Some neurons never activate: they belong to undriven zones and receive only weak incoming fibers.
- A sensory zone's activity can *decrease* when you raise another drive: shared inhibitory neurons exert a normalizing, competitive effect (E/I balance at work).
- Reloading the app keeps a cached brain (`@st.cache_resource`); change the checkpoint path or restart to force a fresh model.

## Extending the Visualizers

In `app.py` the heavy lifting is in `run_simulation(...)` (a thin wrapper over `brain.run_with_inputs`), `top_edges(...)` (selects and balances the strongest E/I synapses), and `make_brain_figure(...)` (builds the Plotly figure). In `visualize.py` see `plot_3d_snapshot`, `save_snapshots`, and `save_animation`. Because positions, zones, and E/I types are always available on the brain, new overlays stay spatially consistent.

Enjoy exploring — and if you find a particularly striking regime, consider contributing a new preset.
