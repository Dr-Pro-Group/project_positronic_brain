# Matched experiment, corrected (content-disjoint held-out split)

**Date:** 2026-06-25 · grid-12 (1,728 neurons) · seq_len 48 · batch 16 · lr 8e-4 ·
400 steps · seeds 42/43/44 · MPS.

The original matched-experiment table was scored on a **leaked** validation split:
the built-in corpus repeats ~50 fixed dialogues, so a positional tail-split put the
same sentences in train and val. `load_corpus_splits` now partitions material into
**content-disjoint** train/val/test sets (dialogue groups split *before* the
`repeats` duplication; tokenizer built from train only), so the held-out numbers
measure **generalization**, not memorization.

Two corpora are reported. The **SODA run is the authoritative one** — it is large
and diverse (1.5M train / 137k val tokens, 0% UNK), so its held-out number is a real
generalization signal. The built-in run is kept only to show *why corpus choice
matters*.

## Authoritative: SODA, content-disjoint (`runs/matched_soda.json`)

2,000 streamed `allenai/soda` conversations → 1.5M train / **137k val** tokens.

| Model | Params | Held-out perplexity (mean ± std) |
|---|---:|---:|
| **Dense RNN (matched)** | 168,403 | **4.85 ± 0.13** |
| Brain − conductance | 169,826 | **5.53 ± 0.13** |
| LSTM (matched) | 169,131 | 6.30 ± 0.08 |
| Brain (full biology) | 169,826 | 6.66 ± 0.14 |
| Brain − spatial wiring | 169,826 | 7.42 ± 0.22 |
| Brain − Dale's law | 169,826 | 8.41 ± 0.40 |

**The paper's qualitative findings survive honest held-out evaluation** (error bands
do not overlap unless noted):

1. **Conductance is the costly ingredient — CONFIRMED.** Removing it improves the
   brain from 6.66 → 5.53.
2. **No-conductance beats the matched LSTM — CONFIRMED.** 5.53 vs 6.30. The paper's
   most interesting finding holds on a real held-out corpus.
3. **Dale's law and spatial wiring both help — CONFIRMED.** Removing Dale (8.41) or
   spatial structure (7.42) is clearly worse than the full brain (6.66).
4. **The matched dense RNN is the strongest baseline (4.85).** Biology *does* cost
   accuracy at this tiny budget — exactly as the paper honestly states. The brain
   (and LSTM) carry recurrent/biological machinery the dense RNN does not need at
   1.7k units / 400 steps.

So the leak did **not** invent the story — it just meant the numbers had never been
shown to be generalization numbers. They now are, and the conclusions hold.

## Built-in corpus, content-disjoint (`runs/matched_fixed.json`) — why corpus choice matters

| Model | Old (leaked) | Built-in (held-out) | SODA (held-out) |
|---|---:|---:|---:|
| Brain (full) | 4.35 | 5.50 ± 0.34 | 6.66 ± 0.14 |
| LSTM | 3.11 | 5.16 ± 0.30 | 6.30 ± 0.08 |
| Dense RNN | 1.57 | **6.88 ± 0.59** | **4.85 ± 0.13** |
| Brain − Dale | 5.61 | 7.02 | 8.41 |
| Brain − conductance | 2.57 | 5.23 | 5.53 |
| Brain − spatial | 4.80 | 6.23 | 7.42 |

On the **built-in** corpus the dense RNN inverts to *worst* (6.88) — but that is
**also unreliable**: the built-in corpus is tiny and repetitive (only ~50 unique
dialogues), and its disjoint val is only ~380 tokens, so a high-capacity dense RNN
overfits the repeated phrasing and its held-out score is dominated by that, not by
language structure. The leak inflated it (1.57); removing the leak on too-small data
deflated it (6.88); only a **large diverse corpus** (SODA, 4.85) gives a trustworthy
generalization number.

## Takeaways

- **Always evaluate on the SODA (or larger) content-disjoint split**, never the
  built-in corpus or a tail-split — the built-in corpus cannot measure
  generalization (too small/repetitive), and a tail-split leaks.
- The paper's headline claims (conductance cost, no-conductance > LSTM, Dale &
  spatial help, dense-RNN-strongest / "biology costs accuracy at this budget") are
  **confirmed** on the authoritative held-out data; only the *evaluation rigor*
  needed fixing, not the conclusions.
- `paper_final.md` / `RESULTS.md` should adopt the SODA held-out numbers above and
  note the methodology fix.
