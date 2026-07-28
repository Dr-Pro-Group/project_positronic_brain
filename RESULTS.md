# Results

Every number here was measured on a single 16 GB Apple M1 Pro (MPS), and every
figure is regenerated from the JSON record beside it by
[`experiments/make_figures.py`](experiments/make_figures.py) — there is no step
where a number is copied by hand.

**Read the evaluation note first.** Held-out text is separated at the level of
whole conversations, *before* the seed dialogues are duplicated, and the tokenizer
is fitted on the training partition alone. This matters more than it sounds: the
conversational corpus repeats a small set of dialogues many times, so a positional
tail-split puts the same sentences on both sides and measures memorisation. An
earlier version of this work did exactly that. Correcting it left the *ordering* of
every system unchanged but moved every absolute perplexity by roughly a factor of
two. The invalidated record is kept as `runs/matched_SUPERSEDED_val_leak.json` so
the correction is auditable; nothing should cite it.

---

## 1. What each biological constraint costs, at a matched budget

Data, tokenizer, sequence length, optimiser, step count and trainable parameter
count are all held fixed; only the constraint under test changes.

`G=12` (1,728 neurons) · SODA content-disjoint · seq_len 48 · batch 16 · lr 8e-4 ·
400 steps · 3 seeds (42/43/44) · [`runs/matched_soda.json`](runs/matched_soda.json)

| Model | Params | Held-out perplexity ↓ |
|---|---:|---:|
| **Dense RNN (matched)** | 245,726 | **4.85 ± 0.13** |
| Brain − conductance | 246,925 | **5.53 ± 0.13** |
| LSTM (matched) | 245,606 | 6.30 ± 0.08 |
| Brain (full biology) | 246,925 | 6.66 ± 0.14 |
| Brain − spatial wiring | 246,925 | 7.42 ± 0.22 |
| Brain − Dale's law | 246,925 | 8.41 ± 0.40 |

![matched-budget ablations](docs/figures/fig2_matched.png)

**At this budget the full biology costs accuracy, and we do not dress that up.**
A plain dense RNN is the strongest model in the table, and a matched LSTM beats the
full brain. The ablations localise the cost:

- **Conductance is the expensive constraint.** Removing the driving force improves
  perplexity to 5.53 and beats the LSTM on every seed (5.44–5.69 vs 6.21–6.36,
  non-overlapping). *This is the claim §2 shows does not survive longer training.*
- **Spatial wiring earns its place.** Randomising the graph at identical edge count
  costs 0.76 perplexity (6.66 → 7.42). *§2 shows this benefit does not persist either.*
- **Dale's law earns its place twice** — on accuracy here, and on stability below.
  This is the one ablation conclusion that holds at every budget we have tested.

**Read §2 before quoting any of these three.** At 400 steps the models are well
short of one epoch, and two of the three conclusions change when they are trained
properly.

## 2. Two of those three conclusions do not survive longer training

The table above is 400 optimiser steps, which turns out to be far from converged —
that run consumed well under one epoch. Repeating the identical comparison at
**3000 steps** (batch 16, 4,000 SODA conversations, ~0.8 epochs, one seed) changes
the answer. [`runs/matched_long.json`](runs/matched_long.json)

| model | 400 steps (3 seeds) | 3000 steps (1 seed) |
|---|---:|---:|
| Dense RNN | 4.85 | **3.36** |
| LSTM | 6.30 | **3.70** |
| Brain − conductance | 5.53 | 4.08 |
| Brain − spatial wiring | 7.42 | 4.44 |
| Brain (full biology) | 6.66 | 4.53 |
| Brain − Dale's law | 8.41 | 4.82 |

Three claims, three different fates:

- **"Removing conductance beats the matched LSTM" — does not survive.** At 400
  steps no-conductance (5.53) beat the LSTM (6.30). At 3000 steps the LSTM is
  ahead, 3.70 against 4.08. The 0.38 gap is roughly 3× the seed spread measured at
  400 steps, so this reversal is probably real. It was the most striking result in
  §1 and it appears to have been an artifact of an undertrained baseline: the LSTM
  gains far more from additional training (−41%) than the brain does (−32%).
- **"Spatial wiring buys accuracy" — collapses into noise.** The 0.76 advantage at
  400 steps becomes −0.09 at 3000. We do *not* claim the effect reverses: 0.09 sits
  well inside the ±0.14–0.22 seed spread and this is a single seed. What we can say
  is that the measured benefit does not persist at a realistic training budget.
- **"Dale's law buys accuracy" — survives.** Removing it remains the worst variant
  (4.82 vs 4.53), and its stability cost (§3) is independent of budget.

The general lesson is the uncomfortable one: **ablation conclusions drawn at a
small step budget are not safe to extrapolate.** Constraints that look costly or
beneficial early can swap places once the baselines are trained properly, and the
convenient direction is the one that gets reported. The 3000-step column is a
single seed and needs replication before it is treated as settled — but it is
already enough to retire the LSTM headline.

## 3. Dale's law buys stability, and more of it over a longer unroll

The accuracy cost above is only half of what the sign constraint is worth. The
other half is a rare-event claim, so it was measured rather than inferred from a
single observation: 8 seeds per condition at two unroll lengths, scoring a run as
diverged when its training loss first becomes non-finite.

`G=12` · 150 steps · SODA content-disjoint ·
[`runs/dale_stability.json`](runs/dale_stability.json) ·
[`experiments/dale_stability.py`](experiments/dale_stability.py)

| Condition | Unroll | Seeds | Diverged | Perplexity of survivors |
|---|---:|---:|---:|---:|
| Brain − Dale's law | 48 | 8 | 1 (12%) | 10.94 ± 0.15 |
| Brain − Dale's law | 64 | 8 | **4 (50%)** | 10.56 ± 0.19 |
| Brain (Dale intact) | 48 | 3 | 0 (0%) | 8.74 ± 0.04 |
| Brain (Dale intact) | 64 | 3 | 0 (0%) | 8.46 ± 0.01 |

![Dale's-law stability](docs/figures/fig7_dale_stability.png)

Removing the constraint makes divergence **probable but not certain**, and the
probability rises sharply with the length of the backpropagated window. Every
divergence occurred on the *first* optimiser step, before any weight update — so
what fails is the forward recursion at initialisation, and a longer unroll simply
compounds an already super-unit gain more times. A diverged run contributes no
perplexity, so the right-hand column averages survivors only and *understates* the
cost of dropping Dale's law.

Note that the matched-budget table in §1 runs at unroll 48 and shows no divergence
at all. On its own it would have hidden this effect entirely.

## 4. More neurons help — at fixed data and compute

Corpus, tokenizer, sequence length, optimiser and step count are held constant; only
`grid_size` varies, so tokens-seen is identical at every point and the curve
isolates *more neurons*.

`G=6→16` (216 → 4,096 neurons) · 400 steps · seq_len 48 · SODA (136k val tokens) ·
[`runs/scaling.json`](runs/scaling.json)

| grid | neurons | params | full bpc ↓ | − conductance | frozen reservoir |
|---:|---:|---:|---:|---:|---:|
| 6 | 216 | 35,791 | 3.326 | 3.196 | 3.491 |
| 8 | 512 | 77,283 | 2.999 | 2.800 | 3.181 |
| 10 | 1,000 | 145,954 | 2.833 | 2.586 | 3.043 |
| 12 | 1,728 | 246,925 | 2.742 | 2.456 | 2.907 |
| 16 | 4,096 | 580,941 | 2.590 | 2.272 | 2.754 |

![neuron scaling](docs/figures/fig5_scaling.png)

Held-out bits-per-char falls monotonically over a 19× range, with a tight
log-linear fit — **bpc ≈ −0.564·log₁₀(N) + 4.58 (R² = 0.96)**, about −0.56 bpc per
10× neurons, with no plateau in range.

![conductance cost and dynamics gain](docs/figures/fig6_scaling_deltas.png)

Two things move in opposite directions as the brain grows:

- **The conductance penalty grows with scale** (+0.13 → +0.32 bpc). Over this range
  the open question "does conductance start paying off at 10⁵ neurons?" trends
  toward *no*: at fixed compute it becomes a larger liability, not a smaller one.
- **The dynamics bonus is real but flat** (~0.16–0.21 bpc). Training the recurrent
  core beats freezing it at every size, so this is not a pure reservoir — but the
  gap does not widen with N.

## 5. About half of that scaling gain was the read-out, not the neurons

The curve above has a confound worth taking seriously: the read-out is
`Linear(N, vocab)`, so it grows with the brain. Over this sweep the head alone
expands **18.9×** (20,398 → 385,118 parameters). "More neurons help" and "more
read-out helps" are not separated.

So we separated them. The sweep was repeated with a *frozen* Gaussian projection
ℝ^N → ℝ¹²⁸ (scaled 1/√N, never trained) in front of a head that is then the same
size at every brain size. The head still sees every neuron; it just cannot grow.

[`runs/scaling_fixed_readout.json`](runs/scaling_fixed_readout.json)

| grid | neurons | growing head | bpc ↓ | fixed head | bpc ↓ |
|---:|---:|---:|---:|---:|---:|
| 6 | 216 | 20,398 | 3.326 | 12,126 | 3.380 |
| 8 | 512 | 48,222 | 2.999 | 12,126 | 3.231 |
| 10 | 1,000 | 94,094 | 2.833 | 12,126 | 3.178 |
| 12 | 1,728 | 162,526 | 2.742 | 12,126 | 3.138 |
| 16 | 4,096 | 385,118 | 2.590 | 12,126 | 3.035 |

![fixed-readout control](docs/figures/fig8_fixed_readout.png)

| | slope | R² | gain 216 → 4,096 |
|---|---:|---:|---:|
| growing read-out | −0.564 bpc / 10× neurons | 0.96 | −0.736 |
| **fixed read-out** | **−0.257 bpc / 10× neurons** | 0.97 | **−0.345** |

**Holding the read-out fixed retains 46% of the slope.** Enlarging the recurrent
population genuinely does improve held-out prediction at fixed data and compute —
but by about −0.26 bpc per tenfold, not −0.56. Any extrapolation built on the
uncontrolled slope overstates the case by roughly a factor of two, and the
uncontrolled slope is the one that would normally get reported.

Two caveats on the control itself: the fixed-read-out models are strictly smaller
in total parameters, so this is a *lower* bound on the neurons' contribution rather
than a perfectly matched comparison; and one projection width at one seed is a thin
basis for the exact ratio, though the gap is large and consistent at all five sizes.

## 6. The zones do not spontaneously specialise

This is the project's most-repeated hypothesis — different data streams enter
through spatially distinct zones, so do those zones *become* a visual area, an
auditory area, and so on? Until now it had no measured answer anywhere in this
repository. It does now, and the answer is no.

Three streams (vision → Visual, audio → Auditory, text → Association) are trained
by self-supervised reconstruction on counterbalanced synthetic scenes, where every
stream in a scene shares a latent content class. Grid 12, 1,500 scenes, 1,500
steps, 5 seeds, probes on held-out scenes and repeated over 5 initialisations.
[`experiments/zone_specialization.py`](experiments/zone_specialization.py) ·
[`runs/zone_specialization.json`](runs/zone_specialization.json)

| metric | untrained | trained |
|---|---:|---:|
| selectivity, **entry** zones | 0.641 ± 0.008 | **0.263 ± 0.011** |
| selectivity, **non-entry** zones | 0.074 ± 0.010 | 0.030 ± 0.007 |
| stream-decoding accuracy (chance 0.333) | **1.000** | 1.000 |

Read the untrained column first, because it is the whole story. A zone at an entry
door already responds preferentially to its own stream **before any training**, at
a selectivity of 0.64, and a linear probe can already identify which stream drove
the network **perfectly** at initialisation. Both numbers measure the wiring
diagram. Reporting either from a trained network as evidence of emergent
specialisation would be reporting the architecture.

And training does not improve on that floor — it *erodes* it. Entry-zone
selectivity more than halves, non-entry zones never acquire any, and the decoding
probe is saturated at 1.000 throughout and so cannot discriminate anything.

### Why it erodes: modality dropout

The training objective masks a random subset of streams and asks the network to
reconstruct them from the others. That is precisely a demand for *shared,
cross-modal* representation, which is the opposite of zone-local specialisation.
Re-running with the masking switched off isolates it
([`runs/zone_specialization_nodropout.json`](runs/zone_specialization_nodropout.json)):

| modality dropout | entry selectivity: init → trained | final loss |
|---:|---|---:|
| 0.5 (default) | 0.641 → **0.263** | 0.121 |
| 0.0 | 0.641 → **0.639** | 0.031 |

Without the cross-modal pressure, selectivity survives training intact — but it
still does not *grow*. So across both objectives the conclusion is the same: the
architecture supplies the zone preference, and training either leaves it alone or
trades it away for cross-modal integration. Nothing here is emergent.

**What this does and does not establish.** It does not show that functional
specialisation can never emerge in this architecture; it shows that under the two
objectives actually implemented here, on synthetic scenes at grid 12, it does not,
and that the metrics one would naturally report are saturated by the input routing
before training begins. Any future claim of emergent specialisation in this system
needs the untrained control alongside it, and needs a measurement that is not
already at ceiling at initialisation.

## 7. Axonal conduction delays (`--delays`)

The 3D embedding has so far only decided *which* neurons connect and *how
strongly*. `--delays` lets it decide *when* a signal lands: each edge carries an
integer latency `round(edge_dist / delay_velocity)` and transmits the presynaptic
rate from that many steps ago. It is the mechanism that makes the geometry
load-bearing rather than decorative.

`G=12` · 400 steps · seq_len 48 · batch 16 · SODA content-disjoint ·
3 seeds · [`runs/delays_measurement.log`](runs/delays_measurement.log)

| seed | baseline | `--delays` | Δ bpc |
|---:|---:|---:|---:|
| 42 | 2.7304 | 2.6958 | −0.0346 |
| 43 | 2.7138 | 2.6917 | −0.0221 |
| 44 | 2.7634 | 2.7135 | −0.0499 |
| **mean** | 2.7359 ± 0.0206 | **2.7003 ± 0.0095** | **−0.0355** |

Held-out bits-per-char improves on all three seeds, by ~1.3% relative.

### But the improvement is lag, not distance

The obvious reading of that number — that the 3D geometry has finally become
load-bearing — is wrong, and two controls show it. `--delay-mode uniform` gives
every edge the *same* latency (lag, no spatial structure at all). `--delay-mode
shuffled` keeps the exact distance-derived latency histogram and reassigns it at
random across edges (same distribution, same mean, geometry destroyed).
[`runs/delay_controls.log`](runs/delay_controls.log)

| arm | seed 42 | seed 43 | seed 44 | mean | vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 2.7304 | 2.7138 | 2.7634 | 2.7359 | — |
| distance (the mechanism) | 2.6958 | 2.6917 | 2.7135 | 2.7003 | −0.0355 |
| uniform (no geometry) | 2.6898 | 2.6905 | 2.7173 | 2.6992 | −0.0367 |
| shuffled (geometry destroyed) | 2.6918 | 2.6874 | 2.7214 | 2.7002 | −0.0357 |

Distance beats uniform by 0.0011 and shuffled by 0.0001, against a baseline seed
spread of 0.0206 — differences one to two orders of magnitude below the noise
floor, and distance wins on only 1 of 3 seeds against each control.

**So `--delays` buys a small, real improvement, and none of it is attributable to
distance.** What helps is having *any* transmission lag, which gives the network a
second temporal timescale; which edge gets which lag is irrelevant. An earlier
version of this document claimed the mechanism made the spatial embedding
load-bearing. It does not, and the claim is retracted.

Why it comes out this way is visible in the delay histogram: at the default
velocity and `connection_radius=2.6`, every edge is either 1 or 2 steps
(9,637 and 18,011 edges at grid 12, the same proportions at grid 32). A one-step
spread is not enough spatial variation for geometry to express itself. Testing
whether distance *per se* ever matters needs a genuinely wider spectrum — slower
conduction (`--delay-velocity`) or a physically larger brain — and that experiment
has not been run.

The delay line costs about **7% per step** (1.22 → 1.30 ms at grid 12, batch 16),
after the implementation was changed from a per-delay-class scatter to a single
fused gather; the first version was 13.8× slower and would have made the flag
unusable.

## 8. Reproducing all of it

```bash
python experiments/matched_experiment.py --mode all --grid-size 12 --steps 400 \
    --seq-len 48 --seeds 42,43,44 --hf-chat soda --json runs/matched_soda.json

python experiments/dale_stability.py --seeds 8 --steps 150 --seq-lens 48,64 \
    --hf-chat soda --json runs/dale_stability.json

python experiments/scaling_study.py --grids 6,8,10,12,16 --steps 400 \
    --no-conductance-too --frozen-too --hf-chat soda --json runs/scaling.json

python experiments/scaling_study.py --grids 6,8,10,12,16 --steps 400 \
    --fixed-readout 128 --hf-chat soda --json runs/scaling_fixed_readout.json

python experiments/zone_specialization.py --grid-size 12 --scenes 1500 --steps 1500 \
    --seeds 5 --device cpu --json runs/zone_specialization.json
python experiments/zone_specialization.py --grid-size 12 --scenes 1500 --steps 1500 \
    --seeds 5 --modality-dropout 0.0 --device cpu \
    --json runs/zone_specialization_nodropout.json

python experiments/make_figures.py
```

## What these results are not

Small budgets, one corpus, character level, three seeds for the matched table.
They are preliminary and directional. The contribution is a *measurement* of what
each biological constraint costs in a generator, not a performance claim — and on
the headline metric the biology currently loses to a plain RNN.
