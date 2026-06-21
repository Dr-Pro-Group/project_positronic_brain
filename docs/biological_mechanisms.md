# Biological mechanisms & fidelity flags

This project is a **testbed**, so every brain-inspired mechanism is added as an
**off-by-default, ablatable flag** that is *byte-identical to the previous model
when off*. That keeps the science honest: each mechanism can be measured on
held-out data against the same baseline, and reported on the axis it actually
moves — **biological fidelity**, **stability**, or **language-modeling quality**
(these often diverge; a fidelity win is frequently quality-neutral, and that is
stated rather than hidden).

All flags live on `BrainConfig` (`positronic_brain/model.py`) and are surfaced on
the trainer CLI (`train_language.py`). They are saved into the checkpoint config
and the `.meta.json` provenance sidecar.

## Honesty instrumentation (always-on capability)

Before any mechanism can be *claimed* to help, the methodology has to make help
measurable. These fixes are the foundation:

| Concern | Fix | Where |
|---|---|---|
| No held-out split (in-distribution numbers) | content-disjoint **train/val/test** split + **bits-per-char** logging + best-val checkpoint | `corpus.load_corpus_splits`, `train_language.py` |
| Ablation harness val leak (repeated dialogues straddle the split) | partition dialogue **groups before** the `repeats` duplication | `corpus.load_corpus_splits`, `matched_experiment.py` |
| OOV chars silently dropped; index 0 aliases a real char | real **UNK** token at index 0, OOV mapped not dropped, UNK-rate logged | `language.CharTokenizer` |
| Dying-ReLU weight trap (`clamp(min=0)`) | smooth non-negative `abs()` reparam (checkpoint-safe) | `model.signed_weights` |
| Two divergent integration depths | single shared `integrate()` helper | `model.integrate`, `language._reverberate` |
| "Is the recurrent core inert?" | `--frozen-reservoir` baseline + per-group grad-norm / saturation / settling / memory-horizon diagnostics (`--diagnostics`) | `diagnostics.py`, `train_language.py` |

Run `python train_language.py --diagnostics` to print the settling residual (how
far from a fixed point the few integration steps actually get) and the empirical
memory horizon (how many characters a perturbation survives) — the measurements
behind the README's honest-scope caveats.

## Implemented mechanisms (Phase 1 + cheap Phase 2)

| Flag | Mechanism | Proven basis | Axis | Notes |
|---|---|---|---|---|
| `--persistent-state` | carry membrane state V across contiguous windows (TBPTT across batches) | the recurrent-state-as-memory premise | **quality + fidelity** | makes the headline "state IS context" claim actually *trained*; warm state provably lowers next-char loss (`test_persistent_state_lowers_loss_on_periodic_text`) |
| `--divnorm` | divisive normalization as a zone-pooled **shunting conductance** | Carandini & Heeger 2012; inhibition-stabilized nets | **stability** + fidelity | bounds activity under strong drive; gives the Voronoi zones a real role |
| `--stp` | Tsodyks–Markram **short-term plasticity** (facilitation u, depression x) | Mongillo et al. 2008 (activity-silent WM) | **fidelity** (2nd timescale) | per-synapse `w·u·x`; threaded through `step`/reverberation/TBPTT |
| `--homeostasis` | per-neuron **intrinsic-gain set-point** controller | Turrigiano 2008 (synaptic scaling) | **stability** | slow gain toward target rate; lets grad-clip loosen at scale |
| `--oscillation` | theta-like **pacemaker** drive on inhibitory cells | Buzsáki; Lisman & Jensen 2013 | **fidelity** | a temporal clock / phase substrate |
| `--dendrites` | per-branch **NMDA** thresholded nonlinearity | Poirazi 2003; Beniaguev et al. 2021 | **fidelity** | per-neuron nonlinear depth without growing the head |
| `--sparse-weight W` | metabolic **sparse-coding** penalty toward a firing set-point | Olshausen & Field 1996 | fidelity / anti-dead-unit | per-neuron mean-rate penalty |
| `--learning-rule eprop` | forward-only **eligibility-trace** learning (vs BPTT) | Bellec et al. 2020 | **fidelity** + online learning | gated by a gradient-agreement test (cosine to BPTT > 0.2); includes the conductance driving-force term |
| `--laminar` | canonical **laminar microcircuit**: L4→L2/3→L5/6 connectivity bias + spatially-even inhibition | Douglas & Martin 2004; Bastos et al. 2012 | **fidelity** | reshapes local wiring (same edge count/k_max); also fixes the random-inhibition zero-gap problem |
| `--lr-schedule warmcos` | LR warmup + cosine decay | standard | quality/stability | tied to step count |

Every flag has a test asserting **off == baseline** and **on is finite/trainable**
(`tests/test_divnorm.py`, `test_stp.py`, `test_homeostasis.py`, `test_eprop.py`,
`test_mechanisms.py`). The combined-stack test (`test_all_mechanisms_together…`)
guards against interaction bugs (it already caught an autograd in-place hazard in
the homeostasis update).

## How to run an ablation

```bash
# baseline vs a mechanism, content-disjoint held-out bits-per-char, 3 seeds
for seed in 42 43 44; do
  python train_language.py --grid-size 12 --steps 400 --seq-len 48 \
      --device mps --seed $seed --eval-every 100 --out runs/base_$seed.pt
  python train_language.py --grid-size 12 --steps 400 --seq-len 48 \
      --device mps --seed $seed --eval-every 100 --divnorm --out runs/divnorm_$seed.pt
done
# the classic biology ablations + matched LSTM/RNN baselines (multi-seed):
python research_paper/matched_experiment.py --mode all --grid-size 12 \
    --steps 400 --seeds 42,43,44 --json runs/matched_fixed.json
```

A mechanism is reported as a **quality win** only if it lowers held-out
bits-per-char beyond ±1 seed-std, and a **stability win** only if it reduces
divergence (NaN/silence) or permits looser grad-clip / larger grids at equal val.
Otherwise it is reported, honestly, as a fidelity feature with neutral quality.

## Deferred (scoped, not yet built)

These are larger and are intentionally left as the next tranche so they get the
same off-by-default, tested, measured treatment rather than a rushed one:

- **Inter-area hierarchy + full PV/SST/VIP interneuron classes.** `--laminar`
  already adds the canonical laminar *connectivity motif* (L4→L2/3→L5/6) and
  spatially-even inhibition; the remaining work is distinct interneuron *types*
  with their own targeting (PV perisomatic, SST dendritic, VIP disinhibitory) and
  directed feedforward/feedback pathways between zones treated as areas.
- **Predictive coding** (Rao & Ballard 1999): top-down prediction + precision-
  weighted inter-zone error as a self-supervised auxiliary loss. Only justifiable
  as an anti-overfit regularizer once the held-out split is in routine use; use
  low-rank heads so it does not dwarf the model or confound the conductance-cost
  ablation.
- **Complementary Learning Systems replay** (`lm_replay.py`): prioritized replay +
  offline ("sleep") consolidation for the LM. Only pays off on a sequential /
  domain-shift corpus where catastrophic forgetting actually occurs — that
  benchmark must be built first; couples tightly with `--persistent-state` (the
  buffer must store or recompute V).
- **Structural plasticity / pruning** via an `edge_active` mask (keeps `edge_index`
  shape and optimizer state coherent): a fidelity + inference-FLOP win, default-off
  so existing checkpoints load.
- **Byte-level / BPE tokenizer + a standard char benchmark** (text8 / enwik8 /
  wikitext) to eliminate OOV and produce literature-comparable bits-per-byte.
