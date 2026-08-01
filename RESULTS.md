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

**Superseded — see below.** This 46% figure comes from a 400-step budget, and a
later sweep at 3,000 steps over a wider range finds **76.8% ± 7.5%** across three seeds. The 400-step
measurement was depressed by the same undertraining that reverses two ablation
conclusions in §2. It is kept here because the comparison between the two is
informative; **the number to cite is 76.8% ± 7.5%** (per-seed: 69.6 / 76.3 / 84.5).

| | slope | range | budget | survives |
|---|---:|---|---|---:|
| this section | −0.257 bpc/10× | 19× | 400 steps | 46% |
| **wide sweep** | **−0.207 bpc/10×** | **27×** | **3,000 steps** | **76.8% ± 7.5%** |

Neuron count over 512 → 13,824 at 3,000 steps: growing read-out 2.4363 → 2.0473,
fixed 128-wide read-out 2.5224 → 2.2290. Neither curve has plateaued.
[`runs/wide_sweep_neurons.json`](runs/wide_sweep_neurons.json)

### Training budget dominates everything

100× more training (400 → 40,000 steps, corpus grown to 8.6M characters so the
longest run stays at 3.6 epochs rather than memorising) buys **−0.885 bpc**:
2.6288 → 2.0129 → 1.7437. [`runs/wide_sweep_training.json`](runs/wide_sweep_training.json)

### Density beats volume at a matched synaptic budget

| arm | neurons | edges | total params | bpc ↓ |
|---|---:|---:|---:|---:|
| volume (G=16, k=16) | 4,096 | 65,536 | 618,390 | 2.1523 |
| **density (G=12, k=38)** | 1,728 | 64,576 | **299,990** | **2.0788** |

Density wins by **0.0671 bpc** — replicated across three seeds (volume 2.1534 ± 0.0029, density 2.0864 ± 0.0095, t = 11.7 on 4 df, winning on 3/3 seeds with the worst density run still ahead of the best volume run) — on **51% fewer
trainable parameters**, 2.4× fewer neurons and a 2.4× smaller read-out — every
asymmetry handicaps the winner. This is the trade by which avian brains reach
primate-like forebrain neuron counts in a much smaller volume, and it says this
project has been scaling `grid_size` when it should have been scaling `k_max` and
`connection_radius`. [`runs/controls.json`](runs/controls.json)

**Multi-seed (complete).** Seeds 42–44 all prefer density. Source: seed 42 in
[`runs/controls.json`](runs/controls.json); seeds 43–44 in
[`runs/replication.json`](runs/replication.json) (`REPLICATION COMPLETE`).

| seed | volume G16/k16 | density G12/k38 | Δ (density − volume) |
|---:|---:|---:|---:|
| 42 | 2.1523 | **2.0788** | **−0.0735** |
| 43 | 2.1513 | **2.0833** | **−0.0680** |
| 44 | 2.1567 | **2.0970** | **−0.0597** |
| **mean ± s.d.** | **2.1534 ± 0.0023** | **2.0864 ± 0.0077** | **−0.0670** |

Density wins on every seed, by ~0.067 bpc on average (~3× the volume seed
spread). The density arm still uses ~51% fewer total parameters.

**Neuron ladder multi-seed (grids 8→16, fixed 128-wide readout).** Same
replication job. Survival of the growing-readout gain under fixed readout:

| seed | growing gain | fixed gain | survival |
|---:|---:|---:|---:|
| 42 (wide_sweep) | 0.2841 | 0.1978 | 69.6% |
| 43 | 0.2845 | 0.2172 | 76.4% |
| 44 | 0.2532 | 0.2138 | 84.4% |
| **mean** | | | **~77%** |

So the “most of the neuron-count gain is real” claim holds across seeds in the
**8→16** range (the paper’s 75.4% figure also used the G=24 single-seed endpoint;
that wider extension remains one-seed).

### The ordering across everything measured

| lever | range | gain |
|---|---|---:|
| training budget | 100× steps | **−0.885** |
| neuron count (read-out controlled) | 27× | −0.294 |
| initial weight scale `g_max` 0.4 → 0.691 | one number | −0.113 |
| best biological mechanism (`--stp`) | — | −0.0845 |
| density over volume, matched budget | — | −0.0671 |
| the other seven mechanisms | — | ~0 or worse |

Two caveats on the fixed-read-out control: those models are strictly smaller in
total parameters, so the neuron contribution is a *lower* bound rather than a
perfectly matched comparison; and one projection width is a thin basis for the
exact ratio, though the gap is large and consistent. The 76.8% survival figure
itself is pending multi-seed confirmation on the ladder endpoints (grids 8 and 16)
in the same `replicate.py` job.

## 6. Every mechanism at a real budget

The mechanism flags were meant to be measurable; most had only been trained at 400
steps, and some had never been run at a serious budget. At `G=16`, 3,000 steps,
seed 42, only differences beyond the ±0.0206 seed floor are claims; the rest are
indistinguishable from baseline.
[`runs/long_program_stage12.json`](runs/long_program_stage12.json) ·
[`experiments/long_program.py`](experiments/long_program.py)

| configuration | bpc ↓ | Δ vs baseline |
|---|---:|---:|
| **`--stp`** | **2.0679** | **−0.0845** |
| `--delays` | 2.1278 | −0.0246 |
| `--dendrites` | 2.1329 | −0.0195 |
| `--laminar` | 2.1505 | −0.0019 |
| baseline | 2.1524 | — |
| `--adaptation` | 2.1536 | +0.0012 |
| `--oscillation` | 2.1598 | +0.0074 |
| `--divnorm` | 2.2266 | +0.0742 |
| `--homeostasis` | 2.2801 | +0.1277 |

**One helps, two hurt, five do nothing** (at this budget and seed). Caveats: (1)
divnorm and homeostasis are *stability* mechanisms scored here only on accuracy;
(2) `--dendrites` is effectively inert as implemented (branch gate ~gain, not a
clear dendritic nonlinearity); (3) single seed — the ranking of the nulls should
not be over-read. Against all of them, raising the shipped `g_max` from 0.4 to
0.691 reaches **2.0393** (−0.113) — larger than every named mechanism
([`runs/controls.json`](runs/controls.json)).

**Multi-seed (seeds 42–44)** at G=16 / 3000 steps for the three load-bearing
levers ([`runs/lever_seeds.json`](runs/lever_seeds.json) + seed-42 priors):

| config | mean bpc ↓ (n=3) | vs baseline |
|---|---:|---:|
| baseline | 2.1535 ± 0.0023 | — |
| **`g_max` 0.691** | **2.0611 ± 0.0166** | **−0.092** |
| `--stp` | 2.0826 ± 0.0105 | −0.071 |

So init scale still beats the best mechanism after multi-seed, and both beat
baseline. (Mini M4 wall-time ~17 min/arm for baseline/g_max; STP ~53 min.)

## 7. A trivial baseline the project never had

Matched LSTM and dense RNN do not answer whether any of the machinery is needed.
An order-4 character *n*-gram, scored under the **same cold-start windowed
protocol** as the model (not free-running), reaches **2.0505 bpc**. The shipped
`G=16` baseline is **2.1522**; the best trained config in the long program is
**2.0369**; the `G=32` checkpoint is **2.0044**.
([`runs/CLAIMS_LEDGER.md`](runs/CLAIMS_LEDGER.md) G2–G3; free-running n-gram curves
in [`runs/context_value.json`](runs/context_value.json) are a different protocol
and must not be mixed with model scores.)

A four-character lookup table therefore **beats the shipped baseline** and sits
within ~0.1 bpc of everything else. That is the comparison that says how much the
biology is earning on this corpus: barely.

## 8. What the recurrent core actually computes

Before asking what a constraint costs, measure whether the substrate computes.

- **Dimensional expansion** of a rank-64 drive is only ~1.06–1.12× across
  grids 10→16 — almost a passthrough
  ([`runs/dimensional_expansion.json`](runs/dimensional_expansion.json)).
- **Linear memory capacity** with held-out scoring is **0.20–0.32% of N** (an
  earlier 3% figure was an overfit read-out artifact; corrected in the claims
  ledger).
- **Perturbation half-life** is ~1 character.
- **Effective population:** clamping the least-modulated 75% of a 32,768-unit
  trained model costs only **+0.0033 bpc**; clamping the same fraction at random
  costs **+2.08 bpc**
  ([`runs/effective_n_grid32.json`](runs/effective_n_grid32.json)).
- Of the most-modulated 5% of units, **all** sit in the text-injection zone.

So mechanism ablations are measurements on a near-passthrough core, not on a rich
reservoir. That reframes every null result above.

## 9. Claims of ours that did not survive their controls

Full audit trail: [`runs/CLAIMS_LEDGER.md`](runs/CLAIMS_LEDGER.md). Seven claims made
during this work were withdrawn after a later control contradicted them:

1. **`--delays` works because distance sets timing** — uniform and shuffled lag
   match distance within noise (delays section below).
2. **Memory capacity is 3% of N** — read-out overfit; true value ~0.2%.
3. **Driving toward criticality restores memory** — raising τ_m raises growth
   toward 1 while MC *falls*.
4. **Signal reaches only 1.9% of the largest grid** — point-kick-at-rest artifact;
   under zone injection reach is tens of percent.
5. **Density lifts participation 15%→87%** — density arm was parameter-confounded;
   the clean result is the matched-budget bpc contrast in §5.
6. **An n-gram beats every config by 0.43 bpc** — protocol mismatch; under matched
   windows the gap is ~0.10 and the best configs win.
7. **Flattening `decay_sigma` is an organisational win** — weight-matched `g_max`
   control reproduces the entire gain; topology was never the mechanism.

Six of seven share one structure: a variable was changed, an effect appeared, and
it was attributed to whichever property of that variable we were thinking about.
The retraction rate is part of the contribution.

## 10. The zones do not spontaneously specialise

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

## 11. Axonal conduction delays (`--delays`)

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

## 12. Reproducing all of it

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

# density / ladder multi-seed (long; currently the in-flight job)
python experiments/replicate.py --seeds 43,44 --json runs/replication.json

python experiments/make_figures.py
```

**How to cite a number.** Prefer the JSON record under `runs/` over any prose table
here or in the LaTeX. Claim status lives in
[`runs/CLAIMS_LEDGER.md`](runs/CLAIMS_LEDGER.md). The journal manuscript is
[`research_paper/paper/`](research_paper/paper/) (`main.tex` / `main_nature.tex`);
older markdown drafts under `research_paper/` are superseded.

## Intelligence protocol (first principles — supersedes Track C as headline)

**Governing doc:** [`docs/INTELLIGENCE_PROGRAM.md`](docs/INTELLIGENCE_PROGRAM.md)

### What we measure as “intelligence” here

Today’s default operationalization of model intelligence is **LLM-class**:

1. Train on **public natural-language** corpora (FineWeb / TinyStories / WikiText / …)
2. Score **held-out bpc / bits-per-byte / perplexity** + generation samples  
3. Compare **architectures at matched params, tokens, and steps**

Tiny GPT = from-scratch **LLM inductive bias** (not GPT-4 weights).  
Brain / brain_wm = **biomimetic** prior.  
LSTM / RNN = classical sequence models.  
CNN = **local floor / negative control** — must *lose* on open language; if CNN
wins open LM bpc, the protocol is broken.

### Hard tasks are not the IQ board

Track C (delayed_copy / addition / associative) can be dominated by a CNN on
addition (~0.99 at matched ~220k params). That shows the **task is local
pattern matching**, not intelligence. Synthetic probes may remain as secondary
diagnostics **only if CNN fails them**. They must not headline “how smart is
the brain?”

### Primary harness (LLM track)

| Piece | Path |
|---|---|
| Program | `docs/INTELLIGENCE_PROGRAM.md` |
| Disk public data + BPE | `positronic_brain/disk_data.py`, `scale_train.py prepare` |
| Full suite on public LM metrics | `experiments/llm_public_benchmark.py` |
| Mini queue | `experiments/queue_llm_intelligence.sh` |
| Outputs | `runs/llm_bench_*.json` |

Suite always: **lstm, rnn, cnn, gpt, brain, brain_wm**.

### Standard model suite

| Tag | Role |
|---|---|
| `lstm` | Gated recurrent baseline |
| `rnn` | Dense Elman RNN |
| `cnn` | Local n-gram **floor** (negative control on language) |
| `gpt` | Tiny decoder Transformer = LLM prior (**not** pretrained GPT-4) |
| `brain` | Positronic Brain G=12 |
| `brain_wm` | Brain + WM + zone routing |

## 13. Public LM qualification (Track A) — char-level (historical + continuity)

Text quality on **public plain-text corpora** with standard char-LM metrics
(**held-out bpc / perplexity**) and fixed-prompt samples. Models are
parameter-matched (~0.27M), trained **from scratch** under the same recipe
(Adam, random windows, BPTT for the brain). Tiny GPT is a small decoder-only
Transformer — **not** pretrained GPT-4.

**This remains valid.** The expanded LLM track (BPE + FineWeb + full suite
including CNN/brain_wm) is the **primary** continuation in §13.4 / queue logs.

Harness (char): [`experiments/public_lm_eval.py`](experiments/public_lm_eval.py)  
Harness (BPE/public disk): [`experiments/llm_public_benchmark.py`](experiments/llm_public_benchmark.py)

### 13.1 TinyStories (complete)

~70.8M train / 8.8M val / 8.8M test chars · vocab 111 · G=12 · 20k steps ·
seq_len 128 · seed 42 · pure public text (no built-in dialogue).

Sources: [`runs/public_tinystories_g12.json`](runs/public_tinystories_g12.json),
[`runs/public_tinystories_g12_gpt.json`](runs/public_tinystories_g12_gpt.json)

| Rank | Model | best val bpc ↓ | test bpc ↓ | best step | wall |
|---:|---|---:|---:|---:|---:|
| 1 | **LSTM** | **1.626** | **1.604** | 20k | 3 min |
| 2 | Dense RNN | 1.644 | 1.634 | 18k | 6 min |
| 3 | Tiny GPT (d=64, L=5, H=2) | 1.837 | 1.813 | 20k | 5 min |
| 4 | **Brain** | 2.090 | 2.058 | 17k | 132 min |

**Ordering: LSTM > RNN > tiny GPT > brain.** The brain **does learn** story-like
character English (finite bpc, improving curves, partial samples) but stays
~0.45–0.46 val bpc behind the matched LSTM under this industrial recipe. Tiny GPT
does **not** beat LSTM at this size — Transformer advantage needs scale, not a
0.27M fair fight.

#### 13.1b Full suite continuity — char TinyStories G12 (complete 2026-08-01)

**Continuity board only** (not the primary intelligence track — that is BPE §13.4–13.6).
Same harness family, shorter budget, **full suite** including CNN / brain_wm.

Source: [`runs/public_tinystories_fullsuite_g12.json`](runs/public_tinystories_fullsuite_g12.json)  
Recipe: HF TinyStories limit 30k · G=12 · **8k** steps · seq 128 · B=8 · seed 42 · char vocab.

| Rank | Model | best val bpc ↓ | test bpc | test ppl | params | notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | **rnn** | **1.750** | **1.765** | **3.40** | 257k | done |
| 2 | cnn | 1.752 | 1.761 | 3.39 | 263k | done · ≈tied #1 |
| 3 | lstm | 1.791 | 1.810 | 3.51 | 255k | done |
| 4 | **brain_wm** | **2.152** | **2.173** | **4.51** | 319k | done · beats brain+gpt |
| 5 | **brain** | **2.252** | **2.266** | **4.81** | 258k | done · stable |
| 6 | gpt | 2.328 | 2.343 | 5.07 | 258k | done |

- **⚠ Protocol flag (char only):** CNN ≈ RNN beat LSTM — expected short-budget char-LM quirk. **Do not** use as IQ ranking. Primary BPE boards still have LSTM on top and CNN not owning.
- **brain_wm** is the best brain arm and beats tiny GPT; plain brain also beats GPT here.
- brain_wm **samples** failed post-test (`generate` left WM buffer at train batch size 8). Metrics intact from log. Fix: `BrainLanguageModel.generate` now calls `_wm_reset(1)`; sample failures are non-fatal in the harness.

**Honest headline (char continuity):** dense RNNs still dominate short char TinyStories; bio stack is mid-pack with **brain_wm > brain > gpt**. Not a regime flip vs LSTM/RNN.

### 13.2 WikiText-2 (complete, valid)

~8.69M train / 1.07M val / 1.11M test chars · vocab 920 · G=12 · 15k steps ·
seq_len 128 · seed 42 · `Salesforce/wikitext` · **no** built-in fallback.
`corpus.train_chars = 8_685_162` ≫ 100k (earlier invalid runs had ~3.5k).
Params ~1.73M each (larger vocab than TinyStories; GPT scaled to d=160, L=5, H=2).

Source: [`runs/public_wikitext2_g12.json`](runs/public_wikitext2_g12.json)
(+ [`runs/public_wikitext2_g12_samples.json`](runs/public_wikitext2_g12_samples.json))

| Rank | Model | best val bpc ↓ | test bpc ↓ | best step | wall |
|---:|---|---:|---:|---:|---:|
| 1 | **LSTM** | **2.055** | **2.058** | 15k | 6 min |
| 2 | Tiny GPT (d=160, L=5, H=2) | 2.190 | 2.207 | 14.5k | 7 min |
| 3 | Dense RNN | 2.200 | 2.204 | 15k | 9 min |
| 4 | **Brain** | 2.540 | 2.563 | 15k | 111 min |

**Ordering: LSTM > tiny GPT ≈ RNN > brain.** Same qualitative story as TinyStories:
the brain trains (monotonic curves, finite held-out bpc) but stays ~0.48 val bpc
behind the matched LSTM. Tiny GPT is competitive with the dense RNN at this size
but does **not** beat LSTM. Earlier “wikitext” artifacts with ~3.5k train chars
(silent built-in fallback) are **superseded** — do not cite them.

### 13.3 Track B — brain-like training (complete)

Same **TinyStories** data (~70.8M train chars); vary training regime only.
15k steps · seq 128 · G=12 · seed 42 · cold-start val/test bpc.

Source: [`runs/brain_training_tinystories.json`](runs/brain_training_tinystories.json)

| Arm | best val bpc ↓ | test bpc ↓ | wall |
|---|---:|---:|---:|
| **LSTM (BPTT)** | **1.673** | **1.637** | 3 min |
| Brain **persistent** | 2.135 | 2.104 | 97 min |
| Brain **BPTT** | 2.136 | 2.108 | 98 min |
| Brain **eprop** | 2.308 | 2.282 | 191 min |

**Honest outcome:** under this protocol, **brain-like training does not beat BPTT**
on held-out bpc, and does **not** close the gap to LSTM.

- **Persistent state:** essentially tied with BPTT (−0.001 val bpc) — no clear win
  when scored with cold-start windows (fair cross-regime metric).
- **e-prop:** *worse* than BPTT by ~0.17 val bpc — local credit works (trains,
  no divergence) but is a quality cost here, matching the original “fidelity /
  online path, not a quality booster” framing in `eprop.py`.
- **LSTM** still leads by ~0.46 bpc over the best brain arm.

So Track A’s ranking is not an artifact of forgetting persistent state; switching
the trainer did not reverse the story on TinyStories next-char bpc.

> **⚠️ Units — read before comparing any number in §13.4–13.6.**
> In these BPE runs the `bpc` field is cross-entropy in bits per **token**, not per
> character (`ce / ln 2`, no character normalisation — see doubt **D9**). It is valid
> for model-vs-model within one corpus, and **invalid** across corpora or against the
> character-level bpc in §13.1–13.2. Use **`bpb`** (bits per byte) for any comparison
> that crosses a tokenizer. In bpb the margins are ~2.4× smaller than the token
> figures suggest — e.g. brain over CNN on FineWeb-Edu is **0.009 bpb**, a tie.

### 13.4 Primary LLM track — BPE TinyStories full suite (complete 2026-07-31)

**Primary intelligence board** (public data + BPE + full suite). Not char-LM §13.1.

Source: [`runs/llm_bench_tinystories_g12.json`](runs/llm_bench_tinystories_g12.json)  
Harness: `experiments/llm_public_benchmark.py` · queue: `experiments/queue_llm_intelligence.sh`

| Setting | Value |
|---|---|
| Corpus | TinyStories public · disk BPE vocab **4096** |
| Tokens | train **6.74M** / val·test **374k** each · ~2.39 bytes/tok |
| Budget | **10k** steps · seq 128 · batch 8 · G=12 · seed 42 · MPS |
| Param target | plain brain **7.42M** (match rule) |

| Rank | Model | best val bpc ↓ | test bpc | test ppl | test bpb | best step | params | wall | status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **lstm** | **2.224** | **2.297** | **4.92** | **0.963** | 9500 | 7.43M | 14 min | done |
| 2 | cnn | 2.731 | 2.764 | 6.79 | 1.158 | 10000 | **1.53M** | 2 min | done ⚠ under-param |
| 3 | gpt | 2.750 | 2.768 | 6.81 | 1.160 | 10000 | 5.85M | 7 min | done |
| 4 | **brain_wm** | **2.786** | **2.761** | **6.78** | **1.157** | 10000 | 7.48M | 130 min | done |
| 5 | rnn | 2.911 | 2.955 | 7.75 | 1.238 | 8500 | 7.42M | 17 min | done |
| 6 | brain | ~2.886@9.5k (curve) | — | — | — | 9500 | 7.42M | ~86 min | **diverged@9878** |

**Ordering (val bpc): LSTM ≫ CNN ≳ gpt ≳ brain_wm > rnn > brain (diverged).**  
**Ordering (test bpc): LSTM ≫ brain_wm ≳ CNN ≳ gpt > rnn** — brain_wm **beats CNN/gpt on test** despite #4 val.

**Protocol flags:**
- **CNN control OK** vs LSTM (local floor loses open LM to gated baseline). Soft warning: CNN ranks #2 on **val** with only ~1/5 params (under-matched causal stack); do not over-read CNN vs gpt/brain_wm without a param-matched CNN retune.
- **brain** trained cleanly to val **2.886** then **NaN@9878** → no test/samples this run.
- **brain_wm** finished full 10k (survived past brain’s failure zone); **+0.10 val bpc vs plain brain curve**; still **~0.56 behind LSTM**. Generation samples hit a **tensor shape bug** at sample time (`[1,1728]` vs `[8,328]`) — **metrics valid, samples not**.
- Tiny GPT (from-scratch, not GPT-4) does **not** beat LSTM at ~6–7M under this recipe.

**Honest headline:** under matched ~7.4M BPE TinyStories, **LSTM still owns open LM bpc**; brain_wm is a real gain over plain brain and competitive with tiny GPT on test, not a regime flip.

### 13.5 Primary LLM track — BPE WikiText-2 full suite (complete 2026-07-31)

**Primary intelligence board** (public data + BPE + full suite). Lower bpc better.

Source: [`runs/llm_bench_wikitext_g12.json`](runs/llm_bench_wikitext_g12.json)  
Harness: `experiments/llm_public_benchmark.py` · queue: `experiments/queue_llm_intelligence.sh`

| Setting | Value |
|---|---|
| Corpus | WikiText-2 public · disk BPE vocab **4096** |
| Tokens | train **4.48M** / val·test **249k** each · ~2.34 bytes/tok |
| Budget | **8k** steps · seq 128 · batch 8 · G=12 · seed 42 · MPS |
| Param target | plain brain **7.42M** (match rule) |

| Rank | Model | best val bpc ↓ | test bpc | test ppl | test bpb | best step | params | wall | status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **lstm** | **3.976** | **4.083** | **16.95** | **1.742** | 8000 | 7.43M | 11 min | done |
| 2 | **brain_wm** | **4.346** | **4.483** | **22.36** | **1.912** | 8000 | 7.48M | 105 min | done |
| 3 | **brain** | **4.397** | **4.538** | **23.23** | **1.936** | 8000 | 7.42M | 73 min | done |
| 4 | gpt | 4.459 | 4.611 | 24.44 | 1.967 | 8000 | 5.85M | 5 min | done |
| 5 | cnn | 4.498 | 4.604 | 24.31 | 1.964 | 8000 | **1.53M** | 2 min | done ⚠ under-param |
| 6 | rnn | 4.744 | 4.866 | 29.16 | 2.076 | 8000 | 7.42M | 14 min | done |

**Ordering (val & test bpc): LSTM ≫ brain_wm > brain > gpt ≳ cnn > rnn.**

**Protocol flags:**
- **CNN control OK** vs LSTM (local floor loses open LM). Soft warning: CNN ~1/5 params; ranks #5 not #1 — protocol healthy (unlike synthetic addition where CNN can “win”).
- **brain** finished full 8k with **no late NaN** (contrast TinyStories §13.4 NaN@9878).
- **brain_wm** overtook plain brain late (~7k) and holds **#2 val & test**; gap to LSTM still ~0.37 val / ~0.40 test bpc.
- Tiny GPT (from-scratch) does not beat brain/brain_wm under this WikiText recipe.
- All six models produced samples; metrics primary.

**Honest headline:** on BPE WikiText-2 at ~7.4M, **LSTM still owns open LM bpc**; **brain_wm is the best brain arm and beats tiny GPT/CNN/RNN**; plain brain is stable and competitive (#3). Not a regime flip vs LSTM.


### 13.8 Checkpoints, fine-tune, and alignment hooks (2026-08-01)

Training is a **lifecycle**, not a one-shot table:

1. Pretrain on public data → **save** `checkpoints/<run>/{model}.pt` + tokenizer  
2. Fine-tune / resume → `experiments/finetune_from_checkpoint.py`  
3. Preference alignment (DPO) → `experiments/dpo_from_checkpoint.py`  
4. Scale (G↑, modular areas, more tokens) from saved weights  

API: [`positronic_brain/checkpoints.py`](positronic_brain/checkpoints.py)  
Curriculum: [`docs/TRAIN_AND_SCALE_CURRICULUM.md`](docs/TRAIN_AND_SCALE_CURRICULUM.md)  
DPO core: [`positronic_brain/preference.py`](positronic_brain/preference.py) (`BrainPolicy` = future PPO hook).

`llm_public_benchmark.py` and `overfit_public_lm.py` now default to writing checkpoints
under `checkpoints/<run_name>/` so every board leaves reloadable brain weights.

### 13.7 Overfit stress (public TinyStories BPE)

Push **fixed** public TinyStories BPE slices to many epochs and track
**train_bpc vs val_bpc** (gap / val rise after best = overfit signature).

| Phase | Train freeze | Steps | ≈epochs | Models | Output | Status |
|---|---:|---:|---:|---|---|---|
| A | 1M tokens | 60k | ~60 | lstm,cnn,gpt,brain,brain_wm | `runs/overfit_public_tinystories_1M_60k.json` | **DONE** 2026-08-01 16:49-03 |
| B | 256k tokens | 40k | ~160 | same | `runs/overfit_public_tinystories_256k_40k.json` | **RUNNING** · lstm/cnn/gpt/brain done · **brain_wm live** (20:28-03) |

Harness: [`experiments/overfit_public_lm.py`](experiments/overfit_public_lm.py)  
Queue: `experiments/queue_overfit_public.sh` (starts after FineWeb bench finishes).

Hypothesis: dense models (LSTM/GPT) drive train_bpc down and open a large gap;
brain may train slower and/or show a smaller gap. **Not** an intelligence
leaderboard — a generalization stress test on the same public data.

#### Phase A FINAL (1M tok · 60k steps · ~61 epochs planned)

Source: [`runs/overfit_public_tinystories_1M_60k.json`](runs/overfit_public_tinystories_1M_60k.json)  
Recipe: seq 128 · B=8 · G=12 · lr 8e-4 · grad-clip 0.5 · device mps · vocab from `data/llm_tinystories/`

| Rank (best val ↓) | Model | best val bpc | @step | final train | final val | final gap | max gap | overfit? | diverged | wall | params |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | **gpt** | **2.538** | 44000 | 1.366 | 2.578 | **1.212** | 1.212 | yes | — | 61.0m | 5.85M |
| 2 | **lstm** | **2.829** | 5000 | **0.265** | 4.072 | **3.807** | 3.816 | yes | — | 85.6m | 7.43M |
| 3 | **brain_wm** | **3.158** | 11000 | 2.590 | 3.162 | **0.573** | 0.573 | yes | **12782** | 166.3m | 7.48M |
| 4 | **brain** | **3.220** | 12000 | 2.667 | 3.220 | **0.552** | 0.552 | yes | **12714** | 115.3m | 7.42M |
| 5 | cnn | 3.252 | 8000 | 1.389 | 5.167 | **3.778** | 3.778 | yes | — | 14.4m | **1.53M** ⚠ under-param |

- **Gap story (higher gap ⇒ more overfit):** LSTM ≈ CNN ≫ GPT ≫ brain_wm ≈ brain.
- Dense arms (LSTM/CNN) crush train_bpc and open **~3.8** gap; val rises hard after early best (LSTM best@5k, CNN@8k).
- GPT keeps best **val** of the board and a **moderate** gap (~1.2); val barely rises after best (+0.04).
- brain / brain_wm: **smallest gaps (~0.55)** but both **diverged ~12.7k steps** (~12 epochs) — did not finish 60k. brain_wm slightly better best_val than plain brain; same failure horizon.
- CNN ⚠ under-param (~0.2×); large gap still, not a matched control.
- Shell noise: queue logged `phase A exit=127` from a path line-split artifact (`c_tinystories_1M_60k.json: command not found`); **JSON + metrics are complete**. Phase B started correctly.

**Honest headline (overfit stress, not IQ):** on 1M-token TinyStories freeze, **dense models overfit hard**; **brain arms resist gap but blow up ~12.7k**. GPT is the most stable full-run generalizer here. Do **not** rank intelligence from this table.

### 13.6 Primary LLM track — BPE FineWeb-Edu full suite (complete 2026-08-01)

**Primary intelligence board** (public data + BPE + full suite). Lower bpc better.

Source: [`runs/llm_bench_fineweb_g12.json`](runs/llm_bench_fineweb_g12.json)  
Harness: `experiments/llm_public_benchmark.py` · queue: `experiments/queue_llm_intelligence.sh`

| | |
|---|---|
| Corpus | FineWeb-Edu capped · disk BPE vocab **8192** |
| Tokens | train **14.63M** · val/test **0.81M** each · ~2.56 bytes/tok est |
| Budget | **12k** steps · seq 128 · B=8 · G=12 · seed 42 · param target **~14.8M** |
| Marker | `runs/exp_markers_llm/fineweb_bench.done` @ 07:51-03 |

| Rank | Model | best val bpc ↓ | test bpc | test ppl | test bpb | params | notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **lstm** | **4.588** | **4.406** | **21.20** | **1.722** | 14.74M | matched |
| 2 | **brain_wm** | **5.044** | **4.928** | **30.45** | **1.926** | 14.83M | matched · beats brain |
| 3 | **brain** | **5.063** | **4.969** | **31.31** | **1.942** | 14.77M | matched · stable |
| 4 | cnn | 5.103 | 4.992 | 31.82 | 1.951 | **2.59M** | ⚠ under-param (~0.18×) |
| 5 | gpt | 5.181 | 5.080 | 33.83 | 1.986 | 6.90M | ⚠ under-target (~0.47×) |
| 6 | rnn | 5.948 | 5.823 | 56.60 | 2.276 | 14.77M | matched |

- **Ordering:** LSTM ≫ **brain_wm > brain > cnn > gpt > rnn** (val bpc).
- **brain_wm** finishes **#2** and beats plain brain on both val and test (same story as WikiText §13.5).
- CNN does **not** own open LM vs LSTM (protocol OK); still ⚠ under-param so not a matched control.
- Tiny GPT under-target; not a fair matched comparison to ~14.8M arms.
- Gap LSTM→#2 is large (~0.46 val bpc) — no regime flip on capped FineWeb-Edu either.

**Honest headline:** on capped BPE FineWeb-Edu at ~14.8M / 12k steps, **LSTM still owns open LM bpc**; **brain_wm is the best brain arm (#2)**; plain brain #3 and stable. Not a regime flip vs LSTM.

## 14. Hard-task intelligence probes (Track C — complete, full suite)

Open-ended next-char LM on easy English under-tests **memory, composition, and
binding**. Track C trains matched models on synthetic tasks where those are the
objective — full suite: **GPT · LSTM · CNN · brain · brain_wm**.

| Task | What it demands |
|---|---|
| `delayed_copy` | Hold a string across a blank delay, then echo it |
| `addition` | Multi-digit `a+b=c` (compositional arithmetic as chars) |
| `associative` | Study key=value pairs, answer a query |

Metric: **teacher-forced answer-span character accuracy** (higher better; not open fluency).  
Protocol: 8k steps · batch 32 · eval every 500 · n-eval 256 · seed 42 · MPS · `hard=False`.

| Model | params |
|---|---:|
| tiny GPT | 169,344 |
| LSTM | 161,185 |
| CNN (causal) | 159,405 |
| Brain | 159,068 |
| brain_wm (WM + zone attn) | 220,420 |

Harness: [`experiments/hard_tasks_eval.py`](experiments/hard_tasks_eval.py)  
Merged: [`runs/hard_tasks_g12_full.json`](runs/hard_tasks_g12_full.json)  
(from `hard_tasks_g12.json` + `_brain_wm.json` + `_cnn.json`)

### 14.1 Test answer accuracy (primary)

| Rank | Model | delayed_copy | addition | associative |
|---:|---|---:|---:|---:|
| 1 | **GPT** | **1.000** | **0.999** | 0.336 |
| 2 | **LSTM** | 0.730 | 0.764 | **0.375** |
| 3 | **brain_wm** | 0.508 | **0.866** | 0.170 |
| 4 | Brain | 0.054 | 0.673 | 0.072 |
| 5 | CNN | 0.019 | 0.352 | 0.055 |

Ranks by mean test acc across tasks: GPT 0.778 · LSTM 0.623 · brain_wm 0.515 · brain 0.266 · CNN 0.142. Per-task winners differ (§14.5).

### 14.2 Best val answer accuracy @ step

| Model | delayed_copy | addition | associative |
|---|---|---|---|
| GPT | **1.000** @ 1500 | **1.000** @ 7000 | **0.398** @ 6000 |
| LSTM | 0.750 @ 7500 | 0.805 @ 5500 | 0.322 @ 7500 |
| brain_wm | 0.526 @ 8000 | 0.863 @ 8000 | 0.193 @ 7000 |
| Brain | 0.073 @ 4500 | 0.663 @ 7500 | 0.072 @ 8000 |
| CNN | 0.025 @ 5500 | 0.387 @ 3500 | 0.051 @ 2500 |

### 14.3 Test BPC (secondary; lower better)

| Model | delayed_copy | addition | associative |
|---|---:|---:|---:|
| GPT | **1.070** | **1.257** | **2.658** |
| LSTM | 1.358 | 1.546 | 2.828 |
| brain_wm | 1.482 | **1.455** | 2.927 |
| Brain | 2.242 | 1.770 | 3.196 |
| CNN | 2.569 | 2.092 | 3.331 |

### 14.4 Wall time (min)

| Model | delayed_copy | addition | associative | total |
|---|---:|---:|---:|---:|
| CNN | 1.3 | 1.0 | 1.1 | ~3.4 |
| LSTM | 1.4 | 1.0 | 1.1 | ~3.5 |
| GPT | 1.8 | 1.6 | 1.6 | ~5.0 |
| Brain | 25.2 | 7.2 | 18.1 | ~50.5 |
| brain_wm | 31.1 | 8.5 | 21.7 | ~61.3 |

### 14.5 Rankings and reading

| Task | Order (test answer_acc) | Takeaway |
|---|---|---|
| **delayed_copy** | GPT ≫ LSTM > brain_wm ≫ brain > CNN | GPT solves memory echo; WM/zone lifts brain ~10× (0.51 vs 0.05) but still below LSTM; CNN near chance (no long memory) |
| **addition** | GPT ≫ **brain_wm > LSTM** > brain > CNN | **First brain arm above LSTM** on a hard probe; CNN partial (local digits only) |
| **associative** | LSTM ≳ GPT > brain_wm > brain ≳ CNN | All weak; WM helps a little (0.17 vs 0.07) but far from LSTM/GPT (~0.34–0.38) |

**Did WM + zone attention help?**
- **delayed_copy: yes, strongly vs plain brain** — 0.508 vs 0.054; bpc 1.48 vs 2.24. Matches hold-then-recall. Does **not** close to LSTM (0.73) or GPT (1.0).
- **associative: weak yes** — 0.170 vs 0.072; plateaued ~0.13–0.19 over 8k steps.
- **addition: yes, and competitive** — 0.866 > LSTM 0.764.

**CNN as floor:** causal conv fails delayed_copy / associative and lags on addition — probes need recurrence or attention, not n-grams. Plain brain > CNN on every task, so the sparse 3D stack is not “worse than local convolution,” but still loses to LSTM/GPT except where `brain_wm` wins addition.

**Honest full-suite outcome:** GPT owns structured positional tasks; LSTM best on sparse binding; `brain_wm` is a real gain over plain brain and beats LSTM on addition only; CNN is the lower bound. Track A (TinyStories bpc) remains LSTM-led — Track C shows bio-attention is *task-relevant*, not a free lunch on open LM.

Policy: every intelligence leaderboard keeps **GPT, Brain, LSTM, CNN** (and `brain_wm` when testing attention). Never publish brain-only.

## 15. Scale path implementation (disk data + modular areas)

**Can we train bigger by loading pieces one-by-one and using the hard drive as RAM?**

| Mechanism | Status | Where |
|---|---|---|
| Public LLM corpora streamed to disk shards | **yes** | `positronic_brain/disk_data.py` |
| BPE subword (no extra deps) | **yes** | `positronic_brain/subword.py` |
| Token memmap (SSD as data RAM) | **yes** | `MemmapTokenStore` |
| Multi-area brain, train one area at a time | **yes** | `positronic_brain/modular.py` |
| Area checkpoint save/load on disk | **yes** | `offload_area_to_disk` / `reload_area_from_disk` |
| Per-step edge paging through SSD | **no** | too slow — not attempted |
| CLI | `prepare` / `train-single` / `train-modular` | `experiments/scale_train.py` |

Public presets: `tinystories`, `wikitext`, `fineweb-edu`, `fineweb`, `c4`, `openwebtext`
(always cap with `--max-docs` / `--max-chars` on Mini).

Doubt register: [`runs/SCALE_DOUBTS.md`](runs/SCALE_DOUBTS.md) · design notes:
[`docs/scale_implementation.md`](docs/scale_implementation.md).

### Mini smoke (2026-07-31) — green

Prepare (TinyStories, max-docs=3000, max-chars=3M, vocab=2048): 2.61M chars →
**1,153,392** tokens (train 1,038,054 / val·test 57,669 each).

| Run | Config | best val bpc | test bpc | test ppl | wall | params | artifact |
|---|---|---|---|---|---|---|---|
| modular | 3×G8, 400 steps/area, seq=48, bs=4, MPS | Motor **5.848** (Sensory 6.405 → Assoc 5.926 → Motor 5.848) | **5.864** | 58.2 | 2.25 min | 1.27M | [`runs/scale_modular_g8_smoke.json`](runs/scale_modular_g8_smoke.json) |
| single | G=12, 600 steps, grad_ckpt, seq=48, bs=4, MPS | **4.741** @600 | **4.718** | 26.3 | 1.73 min | 3.75M | [`runs/scale_single_g12_smoke.json`](runs/scale_single_g12_smoke.json) |

Both smokes show finite CE and monotonic val_bpc drop. Modular is a *curriculum /
memory* demo (sequential area freeze), not an equal-N bake-off vs single (D7 open).
First modular attempt hit MPS Embedding placeholder error; fixed by explicit
`.to(device)` on LM IO + pathways — retry green.

**Path claim:** infrastructure + Mini smoke **finalized**. FineWeb overnight and
equal-N modular-vs-single remain optional (D6–D8). Intelligence leaderboards still
require GPT/LSTM/CNN/brain options where relevant.

## Scale & training perspective (larva → monkey)

The Track A–C numbers are only interpretable if the **substrate size** and
**training length** sit next to biology. This section is orientation, not a claim
that rate-model “neurons” equal biological cells.

### Size — where G=12 actually sits

Default brain: **G=12 → N = G³ = 1,728** rate units · ~0.16M trainable params
(~0.22M with WM/zone attention). Matched LSTM/CNN/GPT use the same param budget.

| System | Neurons (order of mag.) | vs this project (G=12) |
|---|---:|---:|
| *C. elegans* (worm) | 302 | ~0.2× |
| **This model (G=12)** | **1,728** | **1×** |
| *Drosophila* **larva** | ~3,000 | ~1.7× |
| This model G=16 / G=24 | 4,096 / 13,824 | 2.4× / 8× |
| Adult fruit fly brain | ~1.4×10⁵ | ~80× |
| Larval zebrafish | ~1×10⁵ | ~60× |
| House mouse (whole CNS) | ~7×10⁷ | ~4×10⁴× |
| Rhesus macaque **cortex alone** | ~1.7×10⁹ | ~10⁶× |
| Human | ~8.6×10¹⁰ | ~5×10⁷× |

**Reading:** G=12 is a **micro-invertebrate / larva-class count**, not a primate
chip. Counting, delayed copy, and associative binding at this N are already
asking a *larva-scale* substrate to do *abstract lab tasks*. A macaque (or even
mouse) “should find these easier” is a **scale intuition**, not a prediction of
this codebase — we have not run 10⁹ units.

Also not 1:1 biology: units are continuous rate units with sparse edges, not
spiking cells with dendritic compartments and ~10³–10⁴ synapses each. Params
(~10⁵) are closer to a tiny ANN than to a larva connectome’s synaptic inventory.

### Training length — how little experience this is

| Track | Budget | Char-presentations (order) | Epochs over train corpus | Wall (brain, Mini) |
|---|---|---:|---:|---:|
| Matched ablations (§1) | 400–3k steps | ~3×10⁵–2×10⁶ | ≪1 → ~1 | minutes |
| **Track C hard tasks** | **8k steps · B=32** | **~3–8×10⁶** | synthetic (online) | ~50–60 min |
| Track A TinyStories | 20k · B=16 · T=128 | ~4×10⁷ | **~0.6** of 71M chars | ~2 h |
| Track A WikiText-2 | 15k · B=16 · T=128 | ~3×10⁷ | **~3.5** of 8.7M chars | ~2 h |
| Wide “long” training | 40k steps | — | ~few epochs (grown corpus) | overnight |

Contrast with animals (order-of-magnitude only):

- A fly larva lives days with continuous sensorimotor streams (≫10⁶ “events”).
- A monkey infant gets **months–years** of multimodal experience before reliable
  symbolic counting / delayed match-to-sample in the lab.
- Frontier LMs see **10¹¹–10¹³** tokens — 10⁴–10⁶× more than Track A.

So Track C is “**does anything learn in a short supervised curriculum?**”, not
“did the animal grow up.” GPT saturating delayed_copy in ~1.5k steps shows the
*task* is easy for attention at matched params; brain_wm needing the full 8k and
still missing perfect copy is a **capacity + inductive-bias** statement at larva
N and short train — not a claim about primate cognition.

### What this implies for the leaderboard

| Expectation | At larva-scale N + short train | At monkey-scale N + long life (not run here) |
|---|---|---|
| Perfect delayed_copy | GPT yes; plain brain no; brain_wm partial | Plausible for many architectures if capacity allows |
| Multi-digit addition | GPT yes; brain_wm competitive with LSTM | Should not be the hard problem |
| Sparse associative binding | All weak (~0.05–0.38) | Where scale + structured memory matter most |
| Open English bpc (Track A) | LSTM still wins at 0.16M | Different game at 10⁹ params / 10¹² tokens |

**Honest framing for the paper:** results measure **mechanism cost and task fit
on a ~10³-unit, ~10⁷-token budget** — invertebrate-scale substrate, lab-curriculum
length. They do **not** forecast monkey or human intelligence. The larva→monkey
ladder is the right mental model: count neurons and steps before claiming
“intelligence level.”

## What these results are not

Small budgets, one corpus, character level, three seeds for the short matched
table and often one seed elsewhere. They are preliminary and directional. The
contribution is a *measurement* of what each biological constraint costs in a
generator, not a performance claim — and on the headline metric the biology
currently loses to a plain RNN (and is competitive with a four-gram). At G=12
the substrate is larva-class by neuron count (§Scale); failures on hard tasks
are failures *at that scale and training length*, not proofs that biology cannot
compute.


---

## A note on the noise floor, and eight retractions

Every "established / not established" judgement in this document originally used a
seed spread of **±0.0206**. That figure was measured at `G=12` with 400 steps and
then applied to `G=16` with 3,000 steps without checking that it transferred.
Measured where the experiments actually ran — five conditions, three seeds each —
the pooled spread is **0.0089**, and per-condition it ranges from **0.0028 to
0.0152**, a 5.4× spread. No single floor was ever going to be correct for all of
them. [`runs/replication_analysis.json`](runs/replication_analysis.json)

Three mechanisms cross the line as a result, so the mechanism split is **3 help /
2 hurt / 3 no effect**, not 1 / 2 / 5. Both `--delays` (2.8σ) and `--dendrites`
(2.2σ) are marginal and should be read as suggestive.

This was the eighth claim of ours withdrawn after a control contradicted it, and
the only one that ran in the *under*-claiming direction — it caused real effects to
be dismissed as noise. The other seven are listed in
[`runs/CLAIMS_LEDGER.md`](runs/CLAIMS_LEDGER.md). Seven of the eight share one
structure: a quantity was carried from the context where it was measured into one
where it had not been checked.
