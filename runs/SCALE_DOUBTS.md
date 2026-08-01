# Scale path — doubt register

Track open questions until closed by code, test, or measured run.

| ID | Doubt | Status | Resolution |
|---|---|---|---|
| D1 | Can we train bigger by loading pieces one-by-one? | **closed** | Modular areas + `set_active_area` + disk save/load (`modular.py`) |
| D2 | Can HDD/SSD act as RAM? | **closed (partial)** | Yes for **data memmap + area checkpoints**; no for per-step edge paging |
| D3 | Public LLM datasets without OOM? | **closed** | `disk_data.stream_public_lm_to_shards` + memmap tokens |
| D4 | Need HuggingFace `tokenizers` package? | **closed** | Pure-Python BPE in `subword.py` |
| D5 | Does modular forward explode currents? | **closed** | Rebuild I each micro-step (token + pathways), no unbounded accumulate |
| D6 | Is brain_wm win just extra params? | **open** | Still need 220k-matched LSTM/GPT on Track C |
| D7 | Does modular beat single cube at equal total N? | **open** | Run `train-single` G≈∛(3×12³) vs `train-modular` 3×G12 |
| D8 | FineWeb on Mini overnight feasible? | **open** | prepare capped sample first; full 10BT not on laptop |
| D9 | Offload mid-forward (true layer streaming)? | **open / deferred** | Would need activation checkpoint across areas; save/load between *stages* is implemented |
| D10 | Subword vs char quality at same steps? | **open** | A/B on TinyStories after prepare |
| D11 | Readout `Linear(N,V)` at large N? | **mitigated** | `--readout-width` on single; modular head is `Linear(N_motor, V)` only |
| D12 | Grad checkpoint + modular? | **open** | Single cube supports `grad_checkpoint`; modular not yet |
| D13 | Multi-area all resident still uses sum of memories | **accepted** | Sequential *grad* freeze helps optimizer state; full RAM free needs offload+reload between stages only |
| D14 | Public data license / attribution for paper | **open** | Cite FineWeb / C4 / TinyStories per HF cards when publishing |
| D15 | MPS Embedding "Placeholder storage" on modular | **closed** | First smoke failed; fix = explicit `.to(device)` on embed/token_in/head/pathways after construction. Retry succeeded on Mini MPS. |

## Finalization checklist

- [x] Subword tokenizer offline tests  
- [x] Memmap store offline tests  
- [x] Modular forward + save/reload tests  
- [x] Mini: `prepare` tinystories — 3000 docs, 2.61M chars → 1,153,392 tokens (train 1,038,054), BPE vocab=2048, ~37s  
- [x] Mini: `train-modular` smoke — 3×G8, 400 steps/area, MPS; test_bpc=**5.864**, test_ppl=58.2, wall=**2.25 min**, params=1.27M; area best_val_bpc Sensory 6.405 → Assoc 5.926 → Motor 5.848 (`runs/scale_modular_g8_smoke.json`)  
- [x] Mini: `train-single` smoke — G=**12** (not G16; smoke budget), 600 steps, grad_ckpt, MPS; best_val_bpc=**4.741** @600, test_bpc=**4.718**, test_ppl=26.3, wall=**1.73 min**, params=3.75M (`runs/scale_single_g12_smoke.json`)  
- [x] Document RESULTS § scale path (smoke metrics)  
- [ ] Close D6–D8 with numbers when runs finish (optional follow-ups; not blocking path claim)

## Scale path status

**FINALIZED (infrastructure + Mini smoke green)** — 2026-07-31.

D1–D5 closed; Mini prepare + modular + single smokes complete with finite descending loss. Large FineWeb / equal-N modular-vs-single / param-matched Track C remain **open experiments** (D6–D8), not blockers for claiming the path exists.

No further 15-min scale-path review loops required unless queuing D6–D8.

## D9 — the public-benchmark `bpc` column is bits per TOKEN, not per character

**Status:** confirmed, corrected in the paper (2026-08-01).

`experiments/llm_public_benchmark.py` computes `bpc = ce / ln 2` without dividing
by characters. Under BPE that is bits per **token** (~2.34–2.56 bytes/token on
these corpora), so:

- the values are valid for **model-vs-model on one corpus** (identical tokenizer);
- they are **not** comparable across corpora;
- they are **not** comparable to the character-level bpc from `train_language.py`
  used everywhere else in RESULTS.md and the paper.

`bpb` (bits per byte) in the same records is correct and is what the paper now
reports. The `bpc` key is kept for compatibility with published run JSON; the
docstring and leaderboard header now say what it is.

**What this changed:** every margin quoted from these runs shrinks by the
bytes-per-token factor. In particular the brain's win over the causal CNN on
FineWeb-Edu is **0.009 bpb**, not the ~0.02 that the token-unit figure suggests —
i.e. a tie, and the negative control very nearly fired. Recorded here because the
first draft of the paper quoted these as "bits per character" and would have
overstated every margin by ~2.4×.
