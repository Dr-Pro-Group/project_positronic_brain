# Intelligence program — first principles (realign)

**Status:** governing protocol after 2026-07-31 critique.  
Hard synthetic tasks (Track C) are **not** the primary intelligence claim.

---

## 1. What went wrong

We treated **delayed_copy / addition / associative** as “intelligence probes.”
On addition, a **causal CNN** reached ~0.99 answer accuracy at matched ~220k
params. That does **not** mean CNNs are more intelligent than brains or LSTMs.

It means the task was **local pattern completion** (digit strings under a fixed
grammar), which is exactly what convolutions do well. A metric that a CNN can
dominate is a **poor operationalization of intelligence** for a biomimetic LM.

Useful secondary diagnostics remain (memory stress, binding), but they are
**not** the headline “how intelligent is the positronic brain?” scoreboard.

---

## 2. What “intelligence” means *here* (comparable claim)

Today’s default operational definition of model intelligence in AI research is:

1. **Train** on large **public natural-language** corpora (the same family used
   for LLMs: web text, educational text, books-like mixes).
2. **Score** with **standard LM metrics** (held-out bits-per-char / bits-per-byte,
   perplexity) and, when budget allows, **public downstream** suites
   (WikiText, LAMBADA-style, simple zero-shot templates).
3. **Compare architectures at matched budget** (params, tokens, steps, data).

We are **not** claiming human/monkey AGI. We claim a **biomimetic sparse 3D
recurrent model** either does or does not match **LLM-family inductive bias**
(tiny GPT) and classical sequence models (LSTM) under that protocol.

| Axis | LLM world | This project |
|---|---|---|
| Data | FineWeb / C4 / OpenWebText / … | same family, capped for Mini |
| Tokenization | BPE / SentencePiece | BPE (`subword.py`) + optional char |
| Train objective | next-token CE | same |
| Metrics | ppl, bpb, gen samples, public evals | bpc/bpb/ppl + samples; grow evals |
| Scale story | bigger N + more tokens → better | same ladder, larva-scale N honest |

---

## 3. Three fair comparison frames (all required)

### Frame A — “LLM-class inductive bias at insect/larva N”

- Tiny **GPT** = compressed stand-in for Transformer LLM prior  
- **Brain / brain_wm** = biomimetic prior  
- Same params, same tokens, same public data  
- Question: *does biology win, lose, or draw on real language?*

### Frame B — “Classical sequence models”

- **LSTM / RNN** = strong recurrent baselines that already beat us on TinyStories  
- **CNN** = *local* floor — should **lose** on long-range language if the metric is real  
- If CNN wins open LM bpc, our data/protocol is still broken

### Frame C — “Toward biological intelligence” (interpretive, not a score)

- Neuron count ladder (G=12 → multi-area → …) vs larva / fly / mouse tables  
- Training length vs lifetime experience (orders of magnitude)  
- Mechanisms (Dale, conductance, WM, zones) ablated on **Frame A metrics only**

Never mix Frame C rhetoric with Frame A numbers without the ladder table.

---

## 4. Primary protocol (must run)

```
public corpus (FineWeb-Edu / TinyStories / WikiText)
  → disk shards + BPE (or char for continuity)
  → train: lstm, rnn, cnn, gpt, brain, brain_wm  [param-matched]
  → report: val/test bpc, ppl, bpb, wall, params, fixed-prompt samples
  → optional: train on FineWeb, eval zero-shot on WikiText holdout
```

Harnesses:

- Data plane: `positronic_brain/disk_data.py` + `experiments/scale_train.py prepare`
- Matched LM suite: `experiments/llm_public_benchmark.py` (full suite on disk store)
- Legacy char Track A: `experiments/public_lm_eval.py` (still valid for TinyStories/WikiText char tables)

### Secondary (optional, never headline alone)

- Hard tasks only with a **CNN-must-fail control**: if CNN ≥ LSTM on the task,
  task is discarded as intelligence evidence.
- Associative binding at long horizon after language pretrain (transfer), not from scratch on toys.

---

## 5. What we stop doing as primary

- Ranking “intelligence” by addition/copy answer accuracy when CNN wins  
- Expanding Track C before Track A / FineWeb is complete for full suite  
- Claiming bio-attention “intelligence gains” without public LM bpc movement  

---

## 6. Honesty constraints

- Tiny GPT ≠ GPT-4; matched from-scratch only  
- G=12 is larva-class N — results are **scale-conditional**  
- One seed unless stated; noise floors from replication study still apply  
- CNN is a **negative control** on language, not a competitor for “IQ”

---

## 7. Success criteria for the brain claim

| Claim | Pass condition |
|---|---|
| “Brain learns open language” | finite, improving held-out bpc on public data |
| “Competitive with LSTM at matched budget” | val bpc within noise of LSTM on same data/steps |
| “Bio machinery helps” | brain_wm or mechanism flag beats plain brain on **public LM bpc** |
| “Smarter than n-grams/CNN” | brain ≪ CNN bpc on public long-context text |
| “LLM-like” | approaches tiny GPT bpc as N/tokens grow (scaling law slope) |

Until those are measured, synthetic tasks are footnotes.
