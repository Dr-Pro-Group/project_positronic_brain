# Experiment queue — **LLM intelligence first** (realigned 2026-07-31)

## Correction

Hard-task / synthetic “intelligence probes” were **stopped** as the primary
program. CNN ≈ 0.99 on addition proved those tasks are not IQ tests.

**Primary:** public LLM-family data + standard LM metrics + full suite.  
**Doc:** [`docs/INTELLIGENCE_PROGRAM.md`](../docs/INTELLIGENCE_PROGRAM.md)

## Active queue — **COMPLETE** (2026-08-01 09:25-03)

Script: `experiments/queue_llm_intelligence.sh`  
Log: `logs/llm_intelligence.log`  
Markers: `runs/exp_markers_llm/` — **all 7 present**

| # | Stage | Output | Status |
|---|---|---|---|
| 1 | prepare TinyStories disk (BPE) | `data/llm_tinystories/` | **DONE** |
| 2 | full suite bench TinyStories | `runs/llm_bench_tinystories_g12.json` | **DONE** · RESULTS §13.4 |
| 3 | prepare WikiText | `data/llm_wikitext/` | **DONE** |
| 4 | full suite WikiText | `runs/llm_bench_wikitext_g12.json` | **DONE** · RESULTS §13.5 |
| 5 | prepare FineWeb-Edu (capped) | `data/llm_fineweb_edu/` | **DONE** |
| 6 | full suite FineWeb | `runs/llm_bench_fineweb_g12.json` | **DONE** · RESULTS §13.6 |
| 7 | char TinyStories full suite | `runs/public_tinystories_fullsuite_g12.json` | **DONE** · RESULTS §13.1b |

Last check (Mini 16:59-03): **all markers done** · primary queue exited · overfit **Phase A DONE** · **Phase B running** (secondary §13.7).  
**No further 15-min primary-loop action required** unless user asks.

### Fix 09:29 — Stage7 sample crash

brain_wm train+test completed (val **2.152** / test **2.173**) then `generate` crashed:
WM buffer left at train B=8 while generate uses B=1 (`index_add` shape error).  
JSON never got brain_wm (dump after samples). **Fixed:**
- `language.generate` → `_wm_reset(1)`
- `public_lm_eval` samples non-fatal + persist every model  
Metrics recovered from log → JSON + `char_tinystories_suite.done` marker.

### Stage 7 FINAL (char TinyStories continuity — not primary) → RESULTS §13.1b

| Rank | Model | best val bpc ↓ | test bpc | test ppl | status |
|---:|---|---:|---:|---:|---|
| 1 | **rnn** | **1.750** | **1.765** | **3.40** | done |
| 2 | cnn | 1.752 | 1.761 | 3.39 | done ⚠ ≈#1 char |
| 3 | lstm | 1.791 | 1.810 | 3.51 | done |
| 4 | **brain_wm** | **2.152** | **2.173** | **4.51** | done · beats brain/gpt |
| 5 | **brain** | **2.252** | **2.266** | **4.81** | done · stable |
| 6 | gpt | 2.328 | 2.343 | 5.07 | done |

⚠ Char only: CNN≈RNN beat LSTM — **not** IQ board. Primary = BPE §13.4–13.6.

### Stage 6 FINAL (BPE FineWeb-Edu) → RESULTS §13.6

| Rank | Model | best val bpc ↓ | test bpc | test ppl | test bpb | status |
|---:|---|---:|---:|---:|---:|---|
| 1 | **lstm** | **4.588** | **4.406** | **21.20** | **1.722** | done |
| 2 | **brain_wm** | **5.044** | **4.928** | **30.45** | **1.926** | done |
| 3 | **brain** | **5.063** | **4.969** | **31.31** | **1.942** | done |
| 4 | cnn | 5.103 | 4.992 | 31.82 | 1.951 | done ⚠ under-param |
| 5 | gpt | 5.181 | 5.080 | 33.83 | 1.986 | done ⚠ under-target |
| 6 | rnn | 5.948 | 5.823 | 56.60 | 2.276 | done |

### Stage 4 FINAL (BPE WikiText) → RESULTS §13.5

| Rank | Model | best val bpc ↓ | test bpc | test ppl | test bpb | status |
|---:|---|---:|---:|---:|---:|---|
| 1 | **lstm** | **3.976** | **4.083** | **16.95** | **1.742** | done |
| 2 | **brain_wm** | **4.346** | **4.483** | **22.36** | **1.912** | done |
| 3 | **brain** | **4.397** | **4.538** | **23.23** | **1.936** | done |
| 4 | gpt | 4.459 | 4.611 | 24.44 | 1.967 | done |
| 5 | cnn | 4.498 | 4.604 | 24.31 | 1.964 | done ⚠ under-param |
| 6 | rnn | 4.744 | 4.866 | 29.16 | 2.076 | done |

### Stage 2 FINAL (BPE TinyStories) → RESULTS §13.4

| Rank | Model | best val bpc ↓ | test bpc | status |
|---:|---|---:|---:|---|
| 1 | **lstm** | **2.224** | **2.297** | done |
| 2 | cnn | 2.731 | 2.764 | done ⚠ under-param |
| 3 | gpt | 2.750 | 2.768 | done |
| 4 | **brain_wm** | **2.786** | **2.761** | done |
| 5 | rnn | 2.911 | 2.955 | done |
| 6 | brain | ~2.886 | — | diverged@9878 |

**Primary protocol:** CNN must not own open LM vs LSTM — **holds on all BPE boards**.  
Do not rank intelligence by addition/copy.

Models every bench: **lstm, rnn, cnn, gpt, brain, brain_wm**  
Metrics: **val/test bpc, ppl, bpb, samples**

## Secondary (not primary IQ) — RESULTS §13.7

### Phase A FINAL (1M tok · 60k · 16:49-03) → `runs/overfit_public_tinystories_1M_60k.json`

| Rank (best val ↓) | Model | best val | final gap | notes |
|---:|---|---:|---:|---|
| 1 | gpt | **2.538** | 1.21 | full 60k · most stable val |
| 2 | lstm | **2.829** | **3.81** | full 60k · classic overfit |
| 3 | brain_wm | **3.158** | 0.57 | **DIVERGED@12782** |
| 4 | brain | **3.220** | 0.55 | **DIVERGED@12714** |
| 5 | cnn | 3.252 | **3.78** | full 60k · ⚠ under-param |

### Phase B RUNNING (256k tok · 40k · started 16:49) — update **20:31-03**

Live: `overfit_public_lm.py` pid 33275 · `runs/overfit_public_tinystories_256k_40k.json`  
Queue shell still wrapping; Phase A exit=127 was path line-split artifact only.

| Rank (best val ↓ so far) | Model | best val | @step | final train/val/gap | notes |
|---:|---|---:|---:|---|---|
| 1 | **gpt** | **3.577** | 7500 | 0.24 / 6.02 / **5.77** | full 40k · overfit_sig |
| 2 | **lstm** | **3.808** | 1500 | 0.12 / 5.99 / **5.87** | full 40k · classic overfit |
| 3 | **brain** | **3.956** | 4500 | 1.89 / 4.37 / **2.48** | **DIVERGED@12192** · wall 111m |
| 4 | cnn | 4.144 | 1500 | 0.21 / **17.99** / **17.78** | full 40k · ⚠ under-param · val exploded |
| — | **brain_wm** | — | — | — | **RUNNING** · started 20:28 · setup only @20:31 |

Pace (brain): ~4.5 min/500 · brain_wm will be slower → first 500 ≈ few min after start; full 40k if no diverge ≈ many hours. Phase A brain_wm diverged ~12.7k.

## Stopped (not primary)

- `queue_all_open_experiments.sh` (D6 hard-task match_wm, G16 hard, etc.) — killed  
- Hard tasks remain on disk as secondary diagnostics only

## 15-min reporting

**Primary queue complete.** Do not restart primary loops unless user asks.  
Secondary: monitor Phase B until DONE → rsync JSON + finalize §13.7.
