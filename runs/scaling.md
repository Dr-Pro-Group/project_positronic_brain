# Neuron-count scaling study (held-out, SODA)

**Date:** 2026-06-26 · grid 6→16 (216 → 4,096 neurons) · SODA content-disjoint
(136k val tokens) · seq_len 48 · batch 16 · lr 8e-4 · **400 steps (fixed)** ·
seed 42 · MPS · `runs/scaling.json` · harness `research_paper/scaling_study.py`.

Capacity scaling at **fixed data and compute**: corpus, tokenizer, seq_len,
optimizer, and step count are held constant; only `grid_size` varies. So
tokens-seen is identical across points and the curve isolates *more neurons*.

| grid | neurons | params | full bpc↓ | no-cond | frozen-reservoir | conductance cost | dynamics gain | it/s | mem |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 216 | 35,791 | 3.326 | 3.196 | 3.491 | +0.130 | +0.165 | 8.6 | 0.2 GB |
| 8 | 512 | 77,283 | 2.999 | 2.800 | 3.181 | +0.199 | +0.182 | 7.6 | 0.6 GB |
| 10 | 1,000 | 145,954 | 2.833 | 2.586 | 3.044 | +0.247 | +0.210 | 5.2 | 1.3 GB |
| 12 | 1,728 | 246,925 | 2.742 | 2.456 | 2.907 | +0.286 | +0.165 | 3.3 | 2.7 GB |
| 16 | 4,096 | 580,941 | 2.590 | 2.272 | 2.754 | +0.318 | +0.164 | 1.6 | 5.7 GB |

- **conductance cost** = full − no-conductance (how much the conductance driving
  force *hurts*); **dynamics gain** = frozen-reservoir − full (how much *training*
  the recurrent core helps over freezing it).

## Three findings

1. **More neurons clearly help — and have not plateaued.** Held-out bits-per-char
   falls monotonically 3.33 → 2.59 over a 19× range. A log-linear fit is tight:
   **bpc ≈ −0.564·log₁₀(N) + 4.58 (R² = 0.96)** — about **−0.56 bpc per 10×
   neurons**. The architecture genuinely uses the extra capacity at fixed
   compute. (Consistency check: grid-12 here is 2.74 bpc = 6.66 perplexity,
   matching the matched-experiment table exactly.)

2. **The conductance cost GROWS with scale (+0.13 → +0.32).** Removing the
   conductance driving force helps *more* as the brain gets bigger, not less. So
   over this range the paper's open question — "does conductance help at 10⁵+
   neurons?" — trends toward **no**: at fixed compute it becomes a larger
   liability. (Caveat: fixed 400 steps; conductance may need more training to pay
   off, and the trend could bend at larger N — this is a within-range result.)

3. **The recurrent dynamics are real but their contribution is ~flat (~0.16–0.21
   bpc).** Training the core beats freezing it at every size, so the brain is
   **not** a pure fixed reservoir — but the gap does not widen with N, so most of
   the per-neuron improvement comes from the growing read-out, with a roughly
   constant dynamics bonus on top.

## Projection (optimistic log-linear extrapolation)

Treat as *directional only* — log-linear ignores the eventual plateau, and the
fixed-400-step budget *underfits* larger N (so more compute would shift these
down). Bands are wide.

| grid | neurons | projected bpc | projected ppl | biological scale |
|---:|---:|---:|---:|---|
| 32 | 32,768 | ~2.03 | ~4.1 | laptop ceiling |
| 46 | 97,336 | ~1.76 | ~3.4 | ≈ fruit fly (~140k) |
| 64 | 262,144 | ~1.52 | ~2.9 | — |
| 100 | 1,000,000 | ~1.19 | ~2.3 | small-mammal cortex patch |

If the trend even half-holds, a ~10⁵-neuron brain (a big-GPU run) would reach the
**laptop dense-RNN baseline (4.85 ppl)** and beyond — but whether the biology
*advantage* ever appears is a separate question, and finding (2) suggests the
*full conductance* brain keeps paying a tax that the current-based variant avoids.

## What to run on a big GPU

Extend with `research_paper/scaling_study.py --grids 24,32,46 --device cuda
--grad-checkpoint`-style memory knobs and **more steps** (so larger N is not
underfit), plus the matched LSTM/dense-RNN at each scale to locate any crossover.
Memory ≈ O(batch·seq·inner·N) is the ceiling (TBPTT + checkpointing decouple
seq_len); compute is sequential in time and does not parallelize across steps.
