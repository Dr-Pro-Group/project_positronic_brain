#!/usr/bin/env python3
"""
app.py — Project Positronic Brain v3
The interactive 3D biomimetic brain visualizer.

Launch with:
    streamlit run app.py

Features:
- Plotly 3D rendering of the cubic neuronal lattice (any grid size).
- Neurons coloured by functional zone (configurable, multi-modal).
- Neuron size / opacity driven by instantaneous firing rate.
- Sparse synapses drawn as 3D lines, coloured by sign:
  warm = excitatory, cool = inhibitory; brightness ~ live signal flow |w|·r.
- Time slider + "Play" animation across the recurrent trajectory.
- Dynamic per-zone input sliders + preset scenarios.
- Live readout of the scalar output + per-zone activation bars.
- Tabs: Live Brain, Activity Traces, Connectivity, Insights.
- Apple Metal (MPS) support via the device selector.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from positronic_brain.model import PositronicBrain, BrainConfig
from positronic_brain.zones import DEFAULT_ZONES, get_zone_info

# -----------------------------------------------------------------------------
# Page config & theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Positronic Brain v3 — Interactive 3D Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
.stApp { background: linear-gradient(180deg, #0f1117 0%, #0a0c12 100%); }
h1, h2, h3 { color: #e6e9f0 !important; font-family: "Inter", system-ui, sans-serif; }
.stMarkdown, .stText, p, li { color: #b8bcc8; }
.stButton>button {
    background: #1f2533; color: #e6e9f0; border: 1px solid #30384a;
    border-radius: 8px; padding: 0.45rem 1rem; transition: all 0.1s ease;
}
.stButton>button:hover { border-color: #3b82f6; background: #252c3c; color: white; }
.zone-label { font-weight: 600; font-size: 0.95rem; }
hr { border-color: #2a2f3d; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model loading (cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Positronic Brain...")
def load_brain(
    model_path: str | None = None,
    grid_size: int = 4,
    recurrent_steps: int = 12,
    device: str = "auto",
) -> PositronicBrain:
    """Load a trained model or instantiate a fresh one with good defaults."""
    if model_path and Path(model_path).exists():
        try:
            brain = PositronicBrain.load(model_path, device=device, strict=False)
            brain.eval()
            return brain
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not load {model_path} ({e}). Using fresh model.")

    cfg = BrainConfig(grid_size=grid_size, recurrent_steps=recurrent_steps, seed=42)
    brain = PositronicBrain(cfg, device=device)
    brain.eval()
    return brain


def zone_color_list(brain: PositronicBrain) -> List[str]:
    """Resolve hex colours for the brain's active zones."""
    by_name = {z.name: z.color for z in DEFAULT_ZONES}
    return [by_name.get(nm, "#9CA3AF") for nm in brain.config.zone_names]


def top_edges(
    brain: PositronicBrain,
    rates: np.ndarray,
    k: int = 48,
) -> List[Tuple[int, int, float, bool]]:
    """
    Return the k synapses carrying the most live signal flow.

    Flow = |w_ij| * r_j (presynaptic firing). To keep the view informative we
    split the budget between excitatory and inhibitory synapses (inhibitory
    weights are stronger and would otherwise dominate the ranking). Each entry
    is (src, dst, flow, is_inhibitory_source).
    """
    ei = brain.edge_index.cpu().numpy()
    w = brain.signed_weights().detach().cpu().numpy()
    src, dst = ei[0], ei[1]
    flow = np.abs(w) * rates[src]
    is_inh_src = brain.is_inhibitory.cpu().numpy()[src]

    def pick(mask, n):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return []
        order = idx[np.argsort(flow[idx])[::-1][:n]]
        return [(int(src[i]), int(dst[i]), float(flow[i]), bool(is_inh_src[i])) for i in order]

    n_exc = k // 2
    edges = pick(~is_inh_src, n_exc) + pick(is_inh_src, k - n_exc)
    edges.sort(key=lambda e: e[2], reverse=True)
    return edges


# -----------------------------------------------------------------------------
# 3D Visualization
# -----------------------------------------------------------------------------
def make_brain_figure(
    positions: np.ndarray,
    activations: np.ndarray,
    zones: np.ndarray,
    zone_colors: List[str],
    edges: List[Tuple[int, int, float, bool]] | None,
    grid_size: int,
    size_scale: float = 7.5,
    timestep: int = 0,
) -> go.Figure:
    """Build a Plotly 3D figure of neurons + signed synapse lines."""
    N = len(positions)
    G = grid_size

    colors = [zone_colors[int(z)] for z in zones]
    sizes = np.clip((0.6 + activations * 1.8) * size_scale, 2.8, 17.0)
    opacities = np.clip(0.28 + activations * 0.66, 0.22, 0.98)

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    scatter = go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=sizes, color=colors, opacity=float(opacities.mean()),
                    line=dict(width=0.6, color="rgba(230,233,240,0.35)")),
        customdata=np.stack([np.arange(N), activations], axis=1),
        hovertemplate="Neuron %{customdata[0]:d}<br>Act %{customdata[1]:.3f}<extra></extra>",
        name="Neurons",
    )
    traces = [scatter]

    if edges:
        max_exc = max((e[2] for e in edges if not e[3]), default=1.0) or 1.0
        max_inh = max((e[2] for e in edges if e[3]), default=1.0) or 1.0
        for src, dst, flow, is_inh in edges:
            norm = flow / (max_inh if is_inh else max_exc)
            alpha = min(0.9, 0.25 + 0.65 * norm)
            # warm (amber) = excitatory, cool (cyan) = inhibitory
            rgb = "239,158,11" if not is_inh else "56,189,248"
            traces.append(
                go.Scatter3d(
                    x=[positions[src, 0], positions[dst, 0]],
                    y=[positions[src, 1], positions[dst, 1]],
                    z=[positions[src, 2], positions[dst, 2]],
                    mode="lines",
                    line=dict(color=f"rgba({rgb},{alpha:.2f})", width=1.6),
                    hoverinfo="skip", showlegend=False,
                )
            )

    fig = go.Figure(data=traces)
    axis = dict(title="", showbackground=False, showgrid=False, zeroline=False,
                showticklabels=False, range=[-0.6, G - 0.4])
    fig.update_layout(
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube",
                   camera=dict(eye=dict(x=1.65, y=1.35, z=1.15)), bgcolor="#0f1117"),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        margin=dict(l=0, r=0, t=10, b=0), height=620, showlegend=False,
        uirevision="brain",
    )
    fig.add_annotation(text=f"t = {timestep}", x=0.02, y=0.98, xref="paper", yref="paper",
                       showarrow=False, font=dict(color="#6b7280", size=11), align="left")
    return fig


# -----------------------------------------------------------------------------
# Presets (named by zone, so they work for any configuration)
# -----------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    "Threat detected (visual+auditory)": {
        "values": {"Visual": 0.92, "Auditory": 0.88, "Emotion": 0.5},
        "description": "Strong external sensory drive. Fast ignition in sensory zones and elevated output.",
    },
    "Calm memory recall": {
        "values": {"Memory": 0.95},
        "description": "Dominant memory activation with slow reverberation; lower overall output.",
    },
    "Painful touch + emotion": {
        "values": {"Somatosensory": 0.9, "Emotion": 0.85},
        "description": "Embodied/affective salience: somatosensory and limbic zones co-activate.",
    },
    "Cross-modal integration": {
        "values": {"Visual": 0.6, "Auditory": 0.55, "Association": 0.8},
        "description": "Balanced multi-zone input; the Association hub bridges sensory streams.",
    },
    "Internal thought / mind-wandering": {
        "values": {"Memory": 0.55, "Association": 0.7},
        "description": "Low sensory, high association + memory; good for seeing long-range fibers.",
    },
    "Baseline (low activity)": {
        "values": {},
        "description": "Near resting state. Observe residual connectivity and E/I balance.",
    },
}


def input_vector(brain: PositronicBrain, values: Dict[str, float]) -> np.ndarray:
    names = brain.config.zone_names
    vec = np.zeros(len(names), dtype=np.float32)
    for nm, val in values.items():
        if nm in names:
            vec[names.index(nm)] = val
    return vec


def run_simulation(brain: PositronicBrain, vec: np.ndarray) -> Dict[str, Any]:
    with torch.no_grad():
        res = brain.run_with_inputs(vec)
    res["input"] = vec
    return res


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
def main() -> None:
    st.title("🧠 Project Positronic Brain v3")
    st.caption(
        "Learnable, sparse, conductance-based 3D recurrent brain • multi-modal zones • "
        "Dale's law E/I balance • biologically motivated dynamics"
    )

    with st.sidebar:
        st.header("Simulation Controls")
        model_path = st.text_input(
            "Model checkpoint (optional)",
            value="trained_models/positronic_brain_v2.pt",
            help="Path to a .pt file produced by train.py. Falls back to a fresh model if missing.",
        )

        with st.expander("Advanced model parameters", expanded=False):
            gsize = st.slider("Grid size (G)", 3, 7, 4, 1)
            rsteps = st.slider("Recurrent steps", 6, 24, 12, 1)
            device_choice = st.selectbox(
                "Compute device", options=["auto", "cpu", "mps"], index=0,
                help="auto = MPS (Metal) on Apple Silicon if available, else CPU.",
            )

        use_ckpt = Path(model_path).exists()
        brain = load_brain(
            model_path if use_ckpt else None,
            grid_size=gsize, recurrent_steps=rsteps, device=device_choice,
        )
        if use_ckpt:
            st.success(f"Loaded checkpoint ({brain.num_neurons} neurons, {brain.num_edges} synapses)")

        names = brain.config.zone_names
        colors = zone_color_list(brain)
        st.divider()

        st.subheader("Zone Input Drive")
        # Dynamic per-zone sliders, two columns.
        if "inputs" not in st.session_state or len(st.session_state["inputs"]) != len(names):
            st.session_state["inputs"] = {nm: 0.0 for nm in names}
        cols = st.columns(2)
        for i, nm in enumerate(names):
            with cols[i % 2]:
                st.session_state["inputs"][nm] = st.slider(
                    nm, 0.0, 1.0, float(st.session_state["inputs"].get(nm, 0.0)), 0.01, key=f"z_{nm}"
                )

        st.divider()
        st.subheader("Preset Scenarios")
        preset_name = st.selectbox("Choose a scenario", list(PRESETS.keys()),
                                   index=0, label_visibility="collapsed")
        preset = PRESETS[preset_name]
        if st.button("Load Preset", use_container_width=True):
            for nm in names:
                st.session_state[f"z_{nm}"] = float(preset["values"].get(nm, 0.0))
            st.rerun()
        st.caption(preset["description"])

        st.divider()
        show_connections = st.checkbox("Show strongest synapses", value=True)
        top_k = st.slider("Number of synapses to show", 12, 120, 48, 4)
        play_speed = st.slider("Animation speed (sec/step)", 0.03, 0.35, 0.09, 0.01)

        st.divider()
        if st.button("🚀 Run Forward Pass", type="primary", use_container_width=True):
            st.session_state["run_trigger"] = True
        if st.button("Reset inputs", use_container_width=True):
            for nm in names:
                st.session_state[f"z_{nm}"] = 0.0
            st.rerun()

    # --- Run simulation ---
    vec = input_vector(brain, st.session_state["inputs"])
    run_now = st.session_state.pop("run_trigger", False) or ("last_result" not in st.session_state)
    if run_now:
        result = run_simulation(brain, vec)
        st.session_state["last_result"] = result
    else:
        result = st.session_state["last_result"]

    rates = result["rates"][0]            # (N,)
    trace = result["trace"][:, 0, :]      # (T, N)
    positions = result["positions"]
    zones = result["zones"]
    colors = zone_color_list(brain)
    T = trace.shape[0] - 1
    out_prob = float(result["output"].ravel()[0])

    # --- Top metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Output Probability", f"{out_prob:.3f}")
    m2.metric("Recurrent Steps", str(brain.config.recurrent_steps))
    m3.metric("Neurons / Synapses", f"{brain.num_neurons} / {brain.num_edges}")
    zone_means = np.array([rates[zones == i].mean() if np.any(zones == i) else 0.0
                           for i in range(brain.config.num_zones)])
    m4.metric("Dominant Zone", names[int(np.argmax(zone_means))])
    st.caption(f"Compute device: **{brain.device}** • Inhibitory neurons: "
               f"{int(brain.is_inhibitory.sum())}/{brain.num_neurons}")

    tab_live, tab_traces, tab_conn, tab_insights = st.tabs(
        ["🧠 Live Brain View", "📈 Activity Traces", "🔗 Connectivity", "💡 Insights"]
    )

    edges = top_edges(brain, rates, k=top_k) if show_connections else None

    # ---------------- Live 3D View ----------------
    with tab_live:
        st.markdown("**Interactive 3D Volume** — drag to rotate, scroll to zoom, hover neurons.")
        t = st.slider("Recurrent timestep", 0, T, T, 1, key="timestep_slider")
        current_act = trace[t]
        edges_t = top_edges(brain, current_act, k=top_k) if show_connections else None
        fig = make_brain_figure(positions, current_act, zones, colors, edges_t,
                                brain.config.grid_size, timestep=t)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        st.caption("Synapse colour: warm = excitatory, cool = inhibitory; brightness ~ live signal flow |w|·r.")

        anim_col1, anim_col2, _ = st.columns([1, 1, 3])
        with anim_col1:
            if st.button("▶ Play animation", use_container_width=True):
                placeholder = st.empty()
                for tt in range(0, T + 1):
                    et = top_edges(brain, trace[tt], k=top_k) if show_connections else None
                    f = make_brain_figure(positions, trace[tt], zones, colors, et,
                                          brain.config.grid_size, timestep=tt)
                    placeholder.plotly_chart(f, use_container_width=True, key=f"anim_{tt}")
                    time.sleep(play_speed)
                st.toast("Animation complete", icon="✅")
        with anim_col2:
            if st.button("⟲ Reset to t=0", use_container_width=True):
                st.session_state["timestep_slider"] = 0
                st.rerun()

        st.subheader("Zone Activation (current timestep)")
        zone_info = get_zone_info(brain.config.grid_size,
                                  [z for z in DEFAULT_ZONES if z.name in names])
        zcols = st.columns(min(len(names), 6))
        for i, nm in enumerate(names):
            act_mean = float(current_act[zones == i].mean()) if np.any(zones == i) else 0.0
            with zcols[i % len(zcols)]:
                st.markdown(f"<span class='zone-label' style='color:{colors[i]}'>{nm}</span>",
                            unsafe_allow_html=True)
                st.progress(min(act_mean, 1.0), text=f"{act_mean:.3f}")

    # ---------------- Activity Traces ----------------
    with tab_traces:
        st.markdown("**Mean firing rate per zone across the recurrent trajectory.**")
        trace_fig = go.Figure()
        for zid, nm in enumerate(names):
            mask = zones == zid
            if not np.any(mask):
                continue
            trace_fig.add_trace(go.Scatter(
                x=list(range(trace.shape[0])), y=trace[:, mask].mean(axis=1),
                mode="lines+markers", name=nm,
                line=dict(color=colors[zid], width=2.5), marker=dict(size=5),
            ))
        trace_fig.update_layout(
            height=420, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#b8bcc8"),
            xaxis=dict(title="Recurrent timestep", gridcolor="#252a38"),
            yaxis=dict(title="Mean firing rate", gridcolor="#252a38", range=[0, 1.02]),
            legend=dict(orientation="h", y=1.12), margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(trace_fig, use_container_width=True)

        st.subheader("Final firing-rate distribution")
        hist_fig = go.Figure(go.Histogram(x=trace[-1], nbinsx=28, marker_color="#3b82f6", opacity=0.85))
        hist_fig.update_layout(height=260, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                               xaxis_title="Final firing rate", yaxis_title="Neuron count")
        st.plotly_chart(hist_fig, use_container_width=True)

    # ---------------- Connectivity ----------------
    with tab_conn:
        st.markdown("**Sparse, distance-biased synaptic graph** (Dale's law: each neuron is purely "
                    "excitatory or inhibitory).")
        w = brain.signed_weights().detach().cpu().numpy()
        exc = w[w > 0]
        inh = w[w < 0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Synapses (edges)", f"{brain.num_edges}")
        c2.metric("Excitatory / Inhibitory", f"{len(exc)} / {len(inh)}")
        c3.metric("Mean |w| (E / I)",
                  f"{exc.mean():.3f} / {abs(inh.mean()) if len(inh) else 0:.3f}")

        dense_equiv = brain.num_neurons ** 2
        st.write(f"Graph density: **{100*brain.num_edges/dense_equiv:.1f}%** of a dense "
                 f"{brain.num_neurons}×{brain.num_neurons} connectome "
                 f"({brain.num_edges} vs {dense_equiv} possible edges).")
        st.caption("Long-range connections are deliberately sparse but functionally critical for "
                   "cross-zone integration — visible when you choose the 'Cross-modal' preset.")

    # ---------------- Insights ----------------
    with tab_insights:
        st.markdown("### What you are seeing")
        st.markdown(
            """
            - **Neuron = linear leaky integrator** of membrane potential `V`
              (`τ_m dV/dt = -(V - E_L) + I_syn + I_ext`).
            - **Synapse = logistic + conductance**: presynaptic firing rate is
              `r = σ(γ(V - V_thr))`; current depends on voltage through reversal
              potentials, so excitation and inhibition have opposite driving forces.
            - **Learnable synapses**: every edge weight is a trainable parameter,
              so the brain adapts its wiring to the task (see `train.py`).
            - **Dale's law**: a neuron is purely excitatory or inhibitory; ~20% are
              inhibitory and keep the network stable (E/I balance).
            - **Local clusters ignite first** because connection strength decays with
              distance; long-range fibers carry cross-modal integration.
            """
        )
        st.divider()
        st.write("**Per-zone mean firing rate:**",
                 {nm: round(float(zone_means[i]), 4) for i, nm in enumerate(names)})
        st.write("**Current input vector:**",
                 {nm: round(float(vec[i]), 3) for i, nm in enumerate(names)})
        with st.expander("Model configuration (current)"):
            st.json(brain.config.to_dict())
        st.info("Tip: load 'Threat detected', run, then lower the sensory sliders while watching "
                "memory and association keep a residual echo alive across timesteps.", icon="💡")

    st.divider()
    st.caption(
        "Project Positronic Brain v3 • CPU + optional Metal (MPS) on Apple Silicon • Open source • "
        "A scientific toy model exploring learnable 3D structured recurrence."
    )


if __name__ == "__main__":
    main()
