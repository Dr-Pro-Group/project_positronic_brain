#!/usr/bin/env python
"""Build every manuscript figure directly from the committed result files.

Each figure declares the JSON it is drawn from and stamps that provenance into
its own footer, so a figure can never quietly outlive the numbers behind it.
That failure has already happened once here: the matched-budget figure was built
from a run that was later invalidated, and because the figure was generated into
one directory and hand-copied into another, the correction never reached it.
There is no copy step any more — this writes straight into the directory LaTeX
reads.

    python experiments/make_figures.py                  # every target, every figure
    python experiments/make_figures.py --only matched,scaling
    python experiments/make_figures.py --outdir /tmp/preview

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The authoritative result files. MATCHED points at the SODA content-disjoint run:
# runs/matched_SUPERSEDED_val_leak.json is the invalidated run and must never be plotted.
MATCHED_JSON = "runs/matched_soda.json"
SCALING_JSON = "runs/scaling.json"
FIXED_READOUT_JSON = "runs/scaling_fixed_readout.json"
DALE_JSON = "runs/dale_stability.json"
LOSS_LOG = "runs/train_lm.txt"

PUBLIC_FIGDIR = "docs/figures"                  # published with the repository
PAPER_FIGDIR = "research_paper/paper/figures"   # local manuscript build only

# Categorical slots, validated colourblind-safe against a light surface
# (worst all-pairs CVD dE 9.2, normal-vision dE 24.0).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#6b7280"        # conventional baselines: context, not subject
INK = "#1f2328"
MUTED = "#6e7781"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": "#3d444d", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _load(rel: str) -> dict:
    with open(os.path.join(ROOT, rel)) as fh:
        return json.load(fh)


def _stamp(fig, source: str) -> None:
    """Print the figure's data source and commit into its footer."""
    fig.text(0.005, -0.055, f"source: {source} @ {_git_sha()}",
             fontsize=6.5, color="#8b949e", family="monospace")


def _save(fig, outdir: str, name: str, source: str) -> None:
    _stamp(fig, source)
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}   [{source}]")


# ------------------------------------------------------- matched-budget ablations
def fig_matched(outdir: str) -> None:
    """Ranked comparison at a matched parameter budget.

    The measure is magnitude on one scale, so colour carries grouping, not identity:
    the conventional baselines recede to gray, the reference brain takes the accent,
    and its single-constraint ablations share a second hue.
    """
    data = _load(MATCHED_JSON)
    res = data["results"]
    seeds = data.get("seeds", [])

    # Keys as written by matched_experiment.py (note the U+2212 minus sign).
    spec = [
        ("dense RNN (matched)",        "Dense RNN\n(matched)",   GRAY),
        ("brain − no conductance", "Brain\n− conductance", ORANGE),
        ("lstm (matched)",             "LSTM\n(matched)",        GRAY),
        ("brain (full biology)",       "Brain\n(full biology)",  BLUE),
        ("brain − no spatial wiring", "Brain\n− spatial",  ORANGE),
        ("brain − no Dale's law",  "Brain\n− Dale",     ORANGE),
    ]
    spec = [s for s in spec if s[0] in res]
    spec.sort(key=lambda s: res[s[0]]["ppl_mean"])

    means = [res[k]["ppl_mean"] for k, _, _ in spec]
    stds = [res[k]["ppl_std"] for k, _, _ in spec]
    labels = [lab for _, lab, _ in spec]
    colors = [c for _, _, c in spec]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(spec))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.95,
           edgecolor="white", linewidth=1.6,
           error_kw={"ecolor": "#3d444d", "elinewidth": 1.2})

    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.12, f"{m:.2f}±{s:.2f}", ha="center",
                fontsize=9, fontweight="bold", color=INK)

    # Divergence is reported from the data, never hardcoded: an earlier figure kept
    # asserting a NaN that the corrected runs do not contain.
    for xi, (key, _, _) in zip(x, spec):
        runs = res[key].get("ppl_runs", [])
        bad = sum(1 for v in runs if v is None or v != v)
        if bad:
            ax.text(xi, 0.25, f"({bad} of {len(runs)} seeds\ndiverged → NaN)",
                    ha="center", fontsize=7.5, color="#c1442e")

    lstm = res.get("lstm (matched)", {}).get("ppl_mean")
    if lstm:
        ax.axhline(lstm, color=GRAY, ls="--", lw=1.0, alpha=0.65)
        ax.text(-0.42, lstm, "matched LSTM ", va="bottom", ha="left",
                fontsize=8, color=MUTED)

    params = res[spec[0][0]]["params"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel("held-out perplexity ↓ (lower is better)")
    ax.set_title(f"Matched-budget held-out perplexity (~{params/1000:.0f}k params, "
                 f"{len(seeds)} seeds: {'/'.join(map(str, seeds))})\n"
                 "conductance is the costly constraint; removing it beats the LSTM")
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) + 0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    _save(fig, outdir, "fig2_matched.png", MATCHED_JSON)


# --------------------------------------------------------------- neuron scaling
def fig_scaling(outdir: str) -> None:
    """Held-out bits-per-char against neuron count, at fixed data and compute.

    Three series, each direct-labelled: colour alone never has to carry identity.
    """
    data = _load(SCALING_JSON)
    res = data["results"]
    series = [
        ("full", "full biology", BLUE, "o"),
        ("no_conductance", "− conductance", ORANGE, "s"),
        ("frozen_reservoir", "frozen reservoir", AQUA, "^"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for key, label, color, marker in series:
        rows = res.get(key, [])
        if not rows:
            continue
        n = [r["neurons"] for r in rows]
        bpc = [r["val_bpc"] for r in rows]
        ax.plot(n, bpc, color=color, lw=2.0, marker=marker, ms=7,
                markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
        ax.annotate(label, (n[-1], bpc[-1]), textcoords="offset points",
                    xytext=(9, -2), fontsize=9, fontweight="bold", color=color)

    # Log-linear fit on the full-biology series, reported with its own R^2 so the
    # extrapolation in the text can be checked against the curve it came from.
    rows = res["full"]
    logn = np.log10([r["neurons"] for r in rows])
    bpc = np.array([r["val_bpc"] for r in rows])
    slope, intercept = np.polyfit(logn, bpc, 1)
    pred = slope * logn + intercept
    ss_res = float(((bpc - pred) ** 2).sum())
    ss_tot = float(((bpc - bpc.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    xs = np.linspace(logn.min(), logn.max(), 50)
    ax.plot(10 ** xs, slope * xs + intercept, color=BLUE, lw=1.0, ls=":",
            alpha=0.65, zorder=2)
    ax.text(0.03, 0.06,
            f"bpc ≈ {slope:.3f}·log₁₀(N) + {intercept:.2f}   "
            f"(R² = {r2:.2f})",
            transform=ax.transAxes, fontsize=9, color=BLUE, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("neurons (log scale)")
    ax.set_ylabel("held-out bits per character ↓")
    ax.set_title(f"More neurons help, at fixed data and compute\n"
                 f"{data['args']['steps']} steps, seq {data['args']['seq_len']}, "
                 f"SODA content-disjoint ({data['val_tokens']:,} val tokens)")
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)   # room for the direct labels
    _save(fig, outdir, "fig5_scaling.png", SCALING_JSON)


def fig_scaling_deltas(outdir: str) -> None:
    """What the conductance constraint costs, and what training the core buys.

    Both are differences between curves in the same figure above, so plotting them
    directly is what makes the trend legible: the cost grows with scale, the gain
    does not.
    """
    data = _load(SCALING_JSON)
    res = data["results"]
    full = {r["grid"]: r for r in res["full"]}
    nocond = {r["grid"]: r for r in res.get("no_conductance", [])}
    frozen = {r["grid"]: r for r in res.get("frozen_reservoir", [])}
    grids = sorted(g for g in full if g in nocond and g in frozen)

    neurons = [full[g]["neurons"] for g in grids]
    cost = [full[g]["val_bpc"] - nocond[g]["val_bpc"] for g in grids]
    gain = [frozen[g]["val_bpc"] - full[g]["val_bpc"] for g in grids]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(neurons, cost, color=ORANGE, lw=2.0, marker="s", ms=7,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.plot(neurons, gain, color=AQUA, lw=2.0, marker="^", ms=7,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.annotate("conductance cost\n(full − no-conductance)", (neurons[-1], cost[-1]),
                textcoords="offset points", xytext=(10, -4), fontsize=9,
                fontweight="bold", color=ORANGE)
    ax.annotate("dynamics gain\n(frozen − full)", (neurons[-1], gain[-1]),
                textcoords="offset points", xytext=(10, -4), fontsize=9,
                fontweight="bold", color=AQUA)

    ax.axhline(0, color="#3d444d", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("neurons (log scale)")
    ax.set_ylabel("Δ bits per character")
    ax.set_title("The conductance penalty grows with scale; the dynamics bonus does not\n"
                 "positive cost = conductance hurts · positive gain = training the core helps")
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(right=ax.get_xlim()[1] * 3.0)
    _save(fig, outdir, "fig6_scaling_deltas.png", SCALING_JSON)


# ------------------------------------------------------- the read-out confound
def _loglin(rows):
    """Least-squares fit of bpc against log10(neurons); returns slope, intercept, R^2."""
    x = np.log10([r["neurons"] for r in rows])
    y = np.array([r["val_bpc"] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    return slope, intercept, r2


def fig_fixed_readout(outdir: str) -> None:
    """How much of the neuron-scaling gain is the neurons, and how much the read-out.

    The default head is ``Linear(N, vocab)`` and therefore grows with the brain, so
    the naive curve cannot separate "more neurons" from "more read-out". Re-running
    it with a frozen projection into a constant-width head separates them.
    """
    path = os.path.join(ROOT, FIXED_READOUT_JSON)
    if not os.path.exists(path):
        print(f"skip fig8_fixed_readout: {FIXED_READOUT_JSON} not found")
        return
    data = _load(FIXED_READOUT_JSON)
    full, fixed = data["results"]["full"], data["results"]["fixed_readout"]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for rows, label, color, marker in (
        (full, "growing read-out\n(Linear(N, vocab))", BLUE, "o"),
        (fixed, f"fixed read-out\n({fixed[0]['head_params']:,} params at every size)",
         ORANGE, "s"),
    ):
        n = [r["neurons"] for r in rows]
        bpc = [r["val_bpc"] for r in rows]
        slope, intercept, r2 = _loglin(rows)
        ax.plot(n, bpc, color=color, lw=2.0, marker=marker, ms=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.annotate(f"{label}\n{slope:.3f} bpc / 10× neurons",
                    (n[-1], bpc[-1]), textcoords="offset points", xytext=(10, -6),
                    fontsize=8.5, fontweight="bold", color=color)

    s_full = _loglin(full)[0]
    s_fixed = _loglin(fixed)[0]
    ax.text(0.03, 0.06,
            f"holding the read-out fixed retains {s_fixed / s_full:.0%} of the slope",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=INK)

    ax.set_xscale("log")
    ax.set_xlabel("neurons (log scale)")
    ax.set_ylabel("held-out bits per character ↓")
    ax.set_title("Roughly half of “more neurons help” was the read-out growing\n"
                 f"{data['args']['steps']} steps, seq {data['args']['seq_len']}, "
                 "SODA content-disjoint, identical data at every point")
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(right=ax.get_xlim()[1] * 3.4)
    _save(fig, outdir, "fig8_fixed_readout.png", FIXED_READOUT_JSON)


# --------------------------------------------------- zone specialisation vs scale
def _spec_means(path: str):
    """Mean entry / non-entry selectivity, before and after training."""
    data = _load(path)
    rows = data["seeds"]
    entry = rows[0]["trained"]["entry_zones"]
    non_entry = rows[0]["trained"]["non_entry_zones"]

    def avg(phase, zones):
        per_seed = [np.mean([r[phase]["selectivity"][z]["index"] for z in zones])
                    for r in rows]
        return float(np.mean(per_seed)), float(np.std(per_seed))

    return {"entry_untrained": avg("untrained", entry), "entry_trained": avg("trained", entry),
            "non_untrained": avg("untrained", non_entry), "non_trained": avg("trained", non_entry),
            "n_seeds": len(rows)}


def fig_specialization(outdir: str) -> None:
    """Whether zone selectivity is emergent, and whether brain size changes that.

    The untrained series is the whole point of the figure: it is the level the
    architecture supplies for free, and it is what a trained number has to beat
    before ``specialisation`` is the right word for it.
    """
    grids = [(8, "runs/zone_spec_grid8.json"), (12, "runs/zone_spec_grid12.json"),
             (16, "runs/zone_spec_grid16.json")]
    have = [(g, p) for g, p in grids if os.path.exists(os.path.join(ROOT, p))]
    if len(have) < 2:
        print("skip fig9_specialization: need the grid sweep (experiments/zone_specialization.py)")
        return
    stats = [(g, _spec_means(p)) for g, p in have]
    neurons = [g ** 3 for g, _ in stats]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7))

    # Left: the architectural floor versus what training leaves behind.
    for key, label, color, marker, style in (
        ("entry_untrained", "entry zones, untrained", GRAY, "o", "--"),
        ("entry_trained", "entry zones, trained", BLUE, "o", "-"),
        ("non_untrained", "non-entry, untrained", "#b9c0c8", "^", "--"),
        ("non_trained", "non-entry, trained", ORANGE, "^", "-"),
    ):
        mu = [s[key][0] for _, s in stats]
        sd = [s[key][1] for _, s in stats]
        ax.errorbar(neurons, mu, yerr=sd, color=color, lw=2.0, ls=style, marker=marker,
                    ms=7, markeredgecolor="white", markeredgewidth=1.2, capsize=3,
                    label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("neurons (log scale)")
    ax.set_ylabel("selectivity index")
    ax.set_title("Training never beats the untrained floor")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    ax.grid(alpha=0.18); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: the fraction of the architectural prior that survives training.
    keep = [100.0 * s["entry_trained"][0] / s["entry_untrained"][0] for _, s in stats]
    ax2.plot(neurons, keep, color=BLUE, lw=2.2, marker="o", ms=8,
             markeredgecolor="white", markeredgewidth=1.4, zorder=3)
    for n, k in zip(neurons, keep):
        ax2.annotate(f"{k:.0f}%", (n, k), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9.5, fontweight="bold", color=BLUE)
    ax2.axhline(100, color=GRAY, ls="--", lw=1.0, alpha=0.7)
    ax2.text(neurons[0], 101.5, "untrained level", fontsize=8, color=MUTED)
    ax2.set_xscale("log")
    ax2.set_xlabel("neurons (log scale)")
    ax2.set_ylabel("% of entry-zone selectivity retained")
    ax2.set_title("Bigger brains lose less of it")
    ax2.set_ylim(0, 118)
    ax2.grid(alpha=0.18); ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Zone selectivity is architectural, not emergent — but scale protects it",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig9_specialization.png", "runs/zone_spec_grid{8,12,16}.json")


# ------------------------------------------------------------- Dale's-law stability
def fig_dale(outdir: str) -> None:
    """Divergence rate of the no-Dale ablation against sequence length.

    Only drawn when the sweep has been run; it is the evidence for the stability
    claim, which a single anecdotal NaN could not support.
    """
    path = os.path.join(ROOT, DALE_JSON)
    if not os.path.exists(path):
        print(f"skip fig7_dale: {DALE_JSON} not found (run experiments/dale_stability.py)")
        return
    data = _load(DALE_JSON)
    runs = data["runs"]
    seq_lens = sorted({r["seq_len"] for r in runs})
    conditions = [("no-Dale", ORANGE), ("Dale intact", BLUE)]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    width = 0.36
    x = np.arange(len(seq_lens))
    ceiling = 0.0
    for i, (cond, color) in enumerate(conditions):
        rates, counts = [], []
        for sl in seq_lens:
            rows = [r for r in runs if r["condition"] == cond and r["seq_len"] == sl]
            bad = sum(1 for r in rows if r["diverged_at"] is not None)
            rates.append(bad / len(rows) if rows else 0.0)
            counts.append((bad, len(rows)))
        ceiling = max(ceiling, max(rates))
        pos = x + (i - 0.5) * width
        ax.bar(pos, rates, width * 0.92, color=color, alpha=0.95,
               edgecolor="white", linewidth=1.6, label=cond)
        for xi, rate, (bad, n) in zip(pos, rates, counts):
            # A zero rate is a result, not an absence: draw a visible floor stub so
            # the "never diverged" condition reads as a mark and not a blank gap.
            if rate == 0:
                ax.plot([xi - width * 0.46, xi + width * 0.46], [0, 0],
                        color=color, lw=3.4, solid_capstyle="butt", zorder=4)
            ax.text(xi, rate + 0.022, f"{bad}/{n}", ha="center", fontsize=9,
                    fontweight="bold", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([f"unroll {sl} characters" for sl in seq_lens], color=INK)
    ax.set_ylabel("fraction of seeds that diverged")
    ax.set_ylim(0, ceiling * 1.42 + 0.05)
    ax.set_title("Dale's law buys stability, and more of it over a longer unroll\n"
                 f"{data['args']['steps']} steps, SODA content-disjoint, grid "
                 f"{data['args']['grid_size']} · labels are diverged/total seeds")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, outdir, "fig7_dale_stability.png", DALE_JSON)


# ------------------------------------------------------------------ training loss
def fig_loss(outdir: str) -> None:
    path = os.path.join(ROOT, LOSS_LOG)
    if not os.path.exists(path):
        print(f"skip fig3_loss: {LOSS_LOG} not found")
        return
    txt = open(path).read()
    pairs = re.findall(r"step\s+(\d+)/3000\s+loss=([0-9.]+)", txt)
    steps = np.array([int(s) for s, _ in pairs])
    loss = np.array([float(l) for _, l in pairs])

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(steps, loss, color=BLUE, lw=1.8)
    ax.scatter(steps[-1], loss[-1], color=ORANGE, zorder=5, s=42,
               edgecolor="white", linewidth=1.4)
    ax.annotate(f"final {loss[-1]:.3f}", (steps[-1], loss[-1]),
                textcoords="offset points", xytext=(-52, 10),
                fontsize=10, fontweight="bold", color=ORANGE)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training cross-entropy")
    ax.set_title("Generative brain training (32,768 neurons, SODA, Apple MPS)\n"
                 "3000 steps, batch 4, seq 48 · ~64.5 min")
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, outdir, "fig3_loss.png", LOSS_LOG)


# -------------------------------------------------------------- biological scale
def fig_bioscale(outdir: str) -> None:
    rows = sorted([
        ("C. elegans", 302),
        ("Larval Drosophila", 3016),
        ("Jellyfish (Aurelia)", 5600),
        ("Sea slug (Aplysia)", 20000),
        ("Positronic Brain G=32", 32768),
        ("Zebrafish larva", 100000),
        ("Adult Drosophila", 139255),
        ("Honeybee", 960000),
        ("Mouse", 71000000),
        ("Human", 86000000000),
    ], key=lambda r: r[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    is_model = [l.startswith("Positronic") for l in labels]

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=[ORANGE if m else GRAY for m in is_model],
            alpha=0.95, edgecolor="white", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    for yi, v, m in zip(y, vals, is_model):
        ax.text(v * 1.5, yi, f"{v:,}", va="center", fontsize=8,
                fontweight="bold" if m else "normal", color=ORANGE if m else MUTED)
    ax.set_xlabel("number of neurons (log scale)")
    ax.set_title("Where 32,768 model units sit among nervous systems\n"
                 "(~a sea slug's nervous system; ~¼ of a fruit-fly brain)")
    ax.set_xlim(100, 5e11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, -0.02,
             "Caveat: a model unit is a single scalar leaky integrator; a real "
             "neuron ≈ a 5–8 layer temporal CNN (Beniaguev et al. 2021).",
             fontsize=7.5, style="italic", color=MUTED)
    _save(fig, outdir, "fig4_bioscale.png", "literature values (see caption)")


# ------------------------------------------------------------------- 3D lattice
def fig_brain3d(outdir: str) -> None:
    from positronic_brain.model import PositronicBrain, BrainConfig
    brain = PositronicBrain(BrainConfig(grid_size=12), device="cpu")
    pos = brain.positions.cpu().numpy()
    zones = brain.zones.cpu().numpy()
    znames = brain.config.zone_names
    is_inh = brain.is_inhibitory.cpu().numpy()

    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    for zid, zn in enumerate(znames):
        m = zones == zid
        if not m.any():
            continue
        ax.scatter(pos[m, 0], pos[m, 1], pos[m, 2], s=14, alpha=0.55,
                   color=cmap(zid % 10), label=zn, depthshade=True)
    ax.scatter(pos[is_inh, 0], pos[is_inh, 1], pos[is_inh, 2], s=26,
               facecolors="none", edgecolors="#222", linewidths=0.4, alpha=0.5)

    ei = brain.edge_index.cpu().numpy()
    rng = np.random.default_rng(0)
    sel = rng.choice(ei.shape[1], size=min(500, ei.shape[1]), replace=False)
    for e in sel:
        s, d = ei[0, e], ei[1, e]
        ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                [pos[s, 2], pos[d, 2]],
                color=ORANGE if is_inh[s] else BLUE, lw=0.25, alpha=0.18)

    ax.set_title("Positronic Brain — 3D neuron lattice (G=12, 1,728 neurons)\n"
                 "coloured by zone · warm/cool edges = inhibitory/excitatory (Dale)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
    ax.view_init(elev=18, azim=35)
    _save(fig, outdir, "fig1_brain3d.png", "generated (BrainConfig grid_size=12)")


FIGURES = {
    "matched": fig_matched,
    "scaling": fig_scaling,
    "deltas": fig_scaling_deltas,
    "readout": fig_fixed_readout,
    "specialization": fig_specialization,
    "dale": fig_dale,
    "loss": fig_loss,
    "bioscale": fig_bioscale,
    "brain3d": fig_brain3d,      # last: imports torch and builds a brain
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="",
                   help="comma-separated output directories; defaults to the public "
                        "docs/figures plus the manuscript's figure directory when "
                        "that is present locally, so one command keeps both in step")
    p.add_argument("--only", default="", help=f"comma-separated subset of {list(FIGURES)}")
    args = p.parse_args()

    if args.outdir:
        targets = [d.strip() for d in args.outdir.split(",") if d.strip()]
    else:
        # The manuscript is not in the repository (it goes to arXiv), so its figure
        # directory is written only when it happens to exist on this machine. Both
        # targets are filled in the same run: a figure that lives in two places and
        # is refreshed in only one is how the last set went stale.
        targets = [PUBLIC_FIGDIR]
        if os.path.isdir(os.path.join(ROOT, PAPER_FIGDIR)):
            targets.append(PAPER_FIGDIR)

    wanted = [n.strip() for n in args.only.split(",") if n.strip()] or list(FIGURES)
    unknown = [n for n in wanted if n not in FIGURES]
    if unknown:
        p.error(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")

    for target in targets:
        outdir = target if os.path.isabs(target) else os.path.join(ROOT, target)
        os.makedirs(outdir, exist_ok=True)
        for name in wanted:
            FIGURES[name](outdir)
    print("done.")


if __name__ == "__main__":
    main()
