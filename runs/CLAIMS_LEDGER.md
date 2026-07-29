# Claims ledger

Every claim made during this investigation, with its evidence and current status.
The point is to make contradictions findable rather than let them sit unnoticed in
prose: several claims here were asserted confidently and later refuted by their own
controls, and a few are still in tension with each other.

Status vocabulary:

| status | meaning |
|---|---|
| **confirmed** | measured, and the measurement survived a control or an independent re-run |
| **retracted** | asserted, then refuted by a control I ran afterwards |
| **corrected** | the direction held but the number was wrong |
| **contested** | new evidence disagrees with an earlier measurement; unresolved |
| **untested** | stated as reasoning, never measured |

---

## A. Claims about the published paper and repository

| # | claim | status | evidence |
|---|---|---|---|
| A1 | The LaTeX manuscript was built entirely on validation-leaked numbers | **confirmed** | `sections/00_abstract.tex:19-25`, `06_results.tex:58-63` carried 4.35/3.11/1.57; correction existed only in `paper_final.md` |
| A2 | The leak changed magnitudes but not the ordering of the six systems | **confirmed** | leaked and SODA orderings identical: RNN < no-cond < LSTM < brain < no-spatial < no-Dale |
| A3 | `fig2_matched.png` plotted superseded numbers | **confirmed** | `make_figures.py:76` read `runs/matched_results.json`, the invalidated run |
| A4 | The repo published zero results despite promising held-out evaluation | **confirmed** | 62 tracked files, all code/tests/docs; `runs/`, `research_paper/` gitignored |
| A5 | The paper's param counts were mismatched with its perplexities | **confirmed** | `paper_final.md:294` paired SODA ppl with built-in-corpus param counts |

## B. Claims about biological mechanisms

| # | claim | status | evidence |
|---|---|---|---|
| B1 | Removing Dale's law destabilises training | **confirmed** | 8 seeds: 12% divergence at unroll 48, 50% at 64, 0/6 with Dale intact |
| B2 | The original Dale NaN was from the invalidated run only | **confirmed** | NaN present in `matched_results.json`, absent in both corrected runs |
| B3 | Every divergence occurs on step 1 (forward recursion, not drift) | **confirmed** | `dale_stability.json`, all diverged runs have `diverged_at: 1` |
| B4 | `--delays` improves held-out bpc | **confirmed** | 3/3 seeds at grid 12/400 steps; −0.0246 at grid 16/3000 steps |
| B5 | `--delays` works because distance determines timing | **RETRACTED** | uniform lag 2.6992 and shuffled 2.7002 vs distance 2.7003, against 0.0206 seed spread |
| B6 | `--adaptation` may be quality-positive | **retracted** | 2.1536 vs baseline 2.1524 at grid 16/3000 steps: neutral |
| B7 | `--divnorm` and `--homeostasis` are stability mechanisms | **untested** | both measured quality-negative (+0.0742, +0.1277); stability never measured in this program |
| B8 | `--dendrites` is a per-branch nonlinearity | **contested** | audit found gate pinned 0.450–0.453, ~2% nonlinear residual, acts as ~0.6× gain; the gain-matched control was never run |
| B9 | Of 8 named mechanisms at a real budget, 1 helps / 2 hurt / 5 do nothing | **confirmed** | `long_program_stage12.json`, grid 16, 3000 steps, 1 seed |

## C. Claims about the dynamical regime

| # | claim | status | evidence |
|---|---|---|---|
| C1 | Memory capacity is 3% of N | **CORRECTED** | overfitting bug: readout had more weights than timesteps. True value with held-out scoring: **0.20–0.32%** |
| C2 | Memory half-life is ~1 character | **confirmed** | `diagnostics.memory_horizon`, all perturbations, trained grid-32 model |
| C3 | One-step growth factor is 0.507 (subcritical) | **confirmed** | direct perturbation measurement, reproduced across scripts |
| C4 | Driving toward criticality (ρ≈0.9) will improve memory | **REFUTED** | τ_m 4→60 raises growth 0.507→0.977 while MC falls 29.6→5.8 monotonically |
| C5 | Large input gain suppresses linear memory | **REFUTED** | MC rises with gain (8.2→29.6) then plateaus |
| C6 | Dimensional expansion is ~1.06× and flat in N | **confirmed** | 67.9 / 69.9 / 71.6 dims from rank-64 input at grids 10/12/16 |
| C7 | `g_max` is a usable criticality knob | **refuted** | 0.507 → 0.594 → 1.2e6 between ×2 and ×4; no smooth path |
| C8 | Firing sparsity is already biological | **confirmed** | trained model: 3.5% active per timestep, Treves–Rolls 0.043 |

## D. Claims about signal reach and participation

| # | claim | status | evidence |
|---|---|---|---|
| D1 | Effective population is ≤25% of nominal | **confirmed** | clamping 24,576 of 32,768 least-modulated units costs +0.0033 bpc; random 75% costs +2.0759 |
| D2 | 100% of the most-modulated 5% sit in the input zone | **confirmed** | measured on trained grid-32 model; 0 units in the other five zones |
| D3 | Signal reaches only 1.9% of grid 32 per character | **RETRACTED** | point-kick-at-rest artifact; under zone injection reach is 87% at grid 16, 65% at grid 24 |
| D4 | Volume scaling leaves participation flat at ~600–780 units | **RETRACTED** | same artifact |
| D5 | Density scaling lifts participation 15% → 87% | **RETRACTED** | density arm carried up to 4× the recurrent parameters of the volume arm |
| D6 | Max propagation distance is ~7 lattice units, independent of grid | **confirmed** | stable across injection mode, operating point, and grid size |
| D7 | Reach-percentage is a robust metric | **refuted** | 39.8% → 92.7% at grid 16 depending only on the detection threshold |

## E. Claims about scale and training

| # | claim | status | evidence |
|---|---|---|---|
| E1 | Only 46% of the neuron-scaling gain survives a fixed readout | **CONTESTED** | measured at 400 steps. At 3000 steps: 61% (grid 8→12), **70%** (grid 8→16). See §Contradictions |
| E2 | Two ablation conclusions do not survive a longer budget | **confirmed** | at 3000 steps the LSTM overtakes no-conductance (3.70 vs 4.08); spatial benefit collapses to −0.09 |
| E3 | The headline run saw under 0.2 epochs | **confirmed** | 576,000 chars against a ~3M-char corpus |
| E4 | The plateau at 1.48 is a local minimum | **refuted** | it is a noise floor: batch 4, constant LR, 0.19 epochs |
| E5 | More training closes the gap to the LSTM | **refuted** | over a 7.5× budget the LSTM gap *grew* 0.055 → 0.202 nats |

## F. Claims about method

| # | claim | status | evidence |
|---|---|---|---|
| F1 | Cheap screening predicts which organizations are worth training | **refuted** | `E/I 50-50` ranked 2nd by expansion, trained *worse* than baseline |
| F2 | The 14-config table is a ranking | **corrected** | single seed; only the six rows beyond ±2σ (0.0206) are claims |
| F3 | Reach interventions dominate the trained results | **CONTESTED** | the two `decay sigma` winners are confounded with initialisation scale — see H1 |
| F4 | `decay sigma 12` is the largest single-change effect measured | **confirmed as a number, contested as an explanation** | −0.1155 bpc is real; the attribution to spatial organisation is not established |

---

## G. Claims about what the task requires

| # | claim | status | evidence |
|---|---|---|---|
| G1 | An order-4 n-gram beats every configuration tested by 0.43 bpc | **RETRACTED** | compared free-running n-gram scoring against the model's cold-start windowed scoring. Under matched protocol the gap is 0.10 and the best configs win |
| G2 | The model performs at roughly four-gram level | **confirmed** | matched protocol: 4-gram 2.0505 · grid-16 baseline 2.1522 · best config 2.0369 · grid-32 2.0044 |
| G3 | The shipped baseline is worse than a four-character lookup table | **confirmed** | 2.1522 vs 2.0505, same eval protocol |
| G4 | Context beyond ~4 characters stops helping on this corpus | **untestable by this method** | unseen held-out contexts reach 74% by order 20; the count estimator collapses long before the question is answered |
| G5 | Short context carries most of the available signal | **confirmed** | orders 0→4 buy 2.84 bpc on free-running held-out text, with unseen contexts still under 1% |

## H. Claims about graph organisation

| # | claim | status | evidence |
|---|---|---|---|
| H1 | `decay sigma 12` helps by flattening the spatial prior / extending reach | **CONFOUNDED** | `decay_sigma` also sets initial weight magnitude (`connectivity.py:231`, `exp(-d²/2σ²)`). σ12 gives 1.73× larger mean \|w\| than σ1.75 while changing path length 4.24→4.13 and clustering 0.286→0.249. The topology barely moves; the initialisation scale moves 73%. Control not yet run |
| H2 | "spatial wiring helps" and "flattening the distance bias helps" contradict | **RESOLVED — no contradiction** | `connection_radius` caps max edge length at 2.45 for every σ, so σ12 keeps locality intact (clustering 0.249) while random wiring destroys it (0.018, mean edge 7.90). Different interventions by a wide margin |
| H3 | `connection_radius` is a genuine topology change | **confirmed** | radius 2.6→6.0 moves mean edge 1.86→2.64, path 4.24→3.07, clustering 0.286→0.129 — and does not touch initial weight scale |

## Contradictions requiring resolution

**X1 — the readout-survival figure (E1).** The paper states 46%; the current sweep says
70% at a longer budget and a wider range. These are not compatible as reported. The
likely reconciliation is that 46% was depressed by the 400-step budget, the same way
two ablation conclusions were — but until grids 24 and 32 land, the paper contains a
number this session's own data disputes. **The paper cannot ship with 46% unqualified.**

**X2 — does the reach retraction undermine the zone finding?** D2 (all active units in
the input zone) was offered as *explained by* D3 (limited reach). D3 is retracted. D2
was measured on the trained model and stands on its own, but its explanation no longer
does. Either signal does reach the other zones and they still carry nothing — which
would point at the training objective rather than the geometry — or something else
confines it. **Currently unexplained, and I should stop implying it is explained.**

**X3 — RESOLVED (see H2).** No contradiction: `connection_radius` caps edge length
independently of σ, so flattening the distance bias is a small perturbation of a local
graph, not a step toward random wiring. Superseded by X7.

**X7 — the best result may be an initialisation artifact.** `decay sigma 12` (−0.1155)
and `decay sigma 4.0` (−0.0684) both raise initial weight magnitude (1.73× and 1.55×)
while barely altering topology. The dose-response presented as evidence for an
organisational effect is equally consistent with a dose-response in initialisation
scale. **Control: baseline at σ=1.75 with `g_max` scaled 1.73×.** Until it runs, no
organisational claim should rest on the sigma results — and `connection_radius`
(H3), which changes topology without touching weight scale, becomes the cleaner
instrument for any reach claim.

**X3-original — spatial wiring: does it help or hurt?** The paper claims randomising the graph
costs 0.76 perplexity, so spatial wiring buys accuracy. But `decay sigma 12`, which
flattens the distance bias almost to uniform, is the *best* trained config (−0.1155).
These may be compatible — full randomisation destroys the metric prior entirely, while
sigma 12 keeps distance-biased wiring and merely broadens it — but the paper currently
presents spatial locality as beneficial without qualification, and the strongest single
result in this session comes from weakening it.

**X4 — density vs volume is unresolved, not answered.** D5 is retracted and the matched
contrast (grid 12/k38 ≈ 64.6k edges vs grid 16/k16 ≈ 65.5k edges) has never been run.
Any statement about corvid-style density scaling is currently **untested**.

**X5 — the task may not require what is being measured.** MC, expansion, reach and
memory horizon all quantify long-range temporal integration; next-character prediction
may need little of it. **Partially resolved, and it weakens the diagnostic frame.** The
model sits at roughly four-gram level (G2), and the shipped baseline is beaten by a
four-character lookup table (G3) — so whatever is costing accuracy here is not
obviously long-range memory, since a model with no memory at all is competitive.
Whether context beyond ~4 characters would help remains **unanswerable by n-gram
counting** (G4): the estimator collapses to 74% unseen contexts by order 20, long
before the question is reached. Settling it needs a model that can use long context —
the LSTM's advantage over the brain at a longer budget is the nearest available
evidence, and it points at optimisation rather than at the substrate's memory.

**X6 — the project has never reported a trivial baseline.** The paper compares against
a matched LSTM and dense RNN. It has never compared against an n-gram, which is the
comparison that says whether any of the biological machinery earns its keep at all.
On this corpus the answer is: barely. Any rewrite should carry this baseline.
