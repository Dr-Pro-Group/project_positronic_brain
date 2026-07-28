# Positronic Brain v3 — Full Pipeline Run Report

**Date:** 2026-06-02 · **Machine:** Apple M1 Pro, 16 GB · **Operator:** automated run.
All console logs referenced below live in this `runs/` directory.

---

## 1. Environment & device

| Item | Value |
|---|---|
| Python | 3.11.9 (pyenv `positronic` → re-pinned `pyenv local 3.11.9`, fresh `python -m venv .venv`) |
| torch | 2.12.0 |
| Device | **Apple MPS** (`torch.backends.mps.is_available() == True`; CUDA `False`) |
| Key libs | numpy 2.4.6 · matplotlib 3.10.9 · plotly 6.7.0 · streamlit 1.58.0 · pandas 3.0.3 · pytest 9.0.3 |
| Optional extra installed | `datasets 4.8.5` (for `--hf-chat soda` streaming; commented-out optional in `requirements.txt`) |

The pre-existing `.venv` was Python **3.14.3**, which does not match the README's
3.11.9. It was discarded and rebuilt per the README. Provenance: `runs/env.txt`,
`runs/torch_check.txt`.

---

## 2. Sanity checks — PASS

| Check | Result | Log |
|---|---|---|
| `pytest -q tests/` | **35 passed in 7.06 s** | `runs/pytest.txt` |
| Import smoke test | `positronic_brain` version **3.3.0** | `runs/smoke.txt` |

---

## 3. Matched-budget experiment + ablations (multi-seed)

> **⚠️ Superseded — validation leak (corrected 2026-06-25).** The table in this
> section was scored on a positional tail-split of the **built-in** corpus, which
> repeats ~50 fixed dialogues, so train and val shared sentences and the numbers
> measured **memorisation, not generalisation**. The authoritative, content-disjoint
> **held-out** numbers (on SODA, 137k val tokens) are in **`runs/matched_fixed.md`**
> and `research_paper/paper_final.md` §6.2. Every qualitative finding below survives
> the correction (dense RNN strongest; no-conductance beats the LSTM; conductance is
> costly; Dale's law and spatial wiring help) — only the absolute perplexities and
> the evaluation rigor changed. Use the SODA held-out split for any future number.

Command (as specified):
```
python research_paper/matched_experiment.py --mode all \
    --grid-size 12 --steps 400 --seeds 42,43,44 --json runs/matched_results.json
```
Corpus: built-in offline conversational corpus (273,459 chars, vocab 51), `seq_len=64`,
`batch=16`, `lr=8e-4`, `grad_clip=0.5`, 400 steps, identical batches per model per seed.
Logs: `runs/matched.txt`, `runs/matched_results.json`.

### Multi-seed means vs. the single-seed reference table

| Model | Params | Single-seed ref | **Multi-seed mean ± std (42/43/44)** | Verdict |
|---|---:|---:|---:|---|
| Brain (full biology) | 169,826 | 4.50 | **4.35 ± 0.14** | confirmed |
| LSTM (matched) | 169,131 | 3.25 | **3.11 ± 0.12** | confirmed |
| Dense RNN (matched) | 168,403 | *(pending)* | **1.57 ± 0.01** | **new — strongest & most stable** |
| Brain − Dale's law | 169,826 | NaN | **5.61 ± 0.07 (1 of 3 seeds NaN)** | **partially overturned** |
| Brain − conductance | 169,826 | 2.67 | **2.57 ± 0.14** | confirmed |
| Brain − spatial wiring | 169,826 | 4.98 | **4.80 ± 0.16** | confirmed |

Per-seed perplexities (from the JSON):
- Brain: 4.50 / 4.31 / 4.22
- LSTM: 3.25 / 3.07 / 3.03
- Dense RNN: 1.57 / 1.58 / 1.58
- no-Dale: **NaN** / 5.67 / 5.56
- no-conductance: 2.72 / 2.44 / 2.56
- no-spatial: 4.99 / 4.76 / 4.67

### What the multi-seed run confirms or overturns

1. **"No-conductance beats the LSTM" — CONFIRMED, robustly.** no-conductance
   (2.57 ± 0.14) beats the LSTM (3.11 ± 0.12) on **every seed**; the bands do not
   overlap (no-cond 2.44–2.72 vs LSTM 3.03–3.25). The paper's most interesting
   finding holds across seeds. (It is *not* the best model overall — the dense RNN
   is — but it is the best of the brain variants and the cleanest isolation of the
   conductance cost.)
2. **"No-Dale diverges to NaN" — PARTIALLY OVERTURNED.** Only seed 42 diverged;
   seeds 43 and 44 trained stably to 5.67 and 5.56. So removing Dale's law is
   *unstable* (it can diverge) but does **not always** diverge — the outcome is
   seed-dependent. When it does train, it is the **least accurate** variant
   (5.61, worse than the full brain's 4.35), so Dale's law buys **both stability
   and accuracy**, a slightly stronger statement than the original "stability only."
3. **Dense RNN is the new best baseline (1.57 ± 0.01).** The simplest dense
   recurrent net — previously left "pending" in the paper — is the strongest and
   most stable model at this tiny char-level budget, beating the LSTM and every
   brain variant. This sharpens the central honest result.
4. **Brain (4.35), LSTM (3.11), spatial-wiring cost (→4.80) — all confirmed**
   close to the single-seed reference; the slightly better means reflect averaging
   over 3 seeds rather than 1.

**Docs updated** (measured numbers inserted, marked multi-seed; qualitative claims
unchanged except where the new measurements required it — the dense-RNN row is now
filled and the "always NaN" / "best in table" statements were corrected to match
the data): `research_paper/paper_final.md` §6.2 + abstract, and `README.md`.

---

## 4. Generative brain training (scaled, MPS)

**Final command actually run:**
```
python train_language.py --hf-chat soda --hf-chat-limit 4000 \
    --seq-len 48 --lr 8e-4 --grad-clip 0.5 --sample-every 250 \
    --out trained_models/brain_lm.pt \
    --grid-size 32 --batch-size 4 --device mps --steps 3000
```

| Item | Value |
|---|---|
| Corpus | SODA (`allenai/soda`), 4,000 conversations streamed → 3,033,727 chars, vocab 104 |
| Brain | **32,768 neurons**, 524,288 synapses, 4,965,185 trainable params |
| Steps / batch | 3000 / **4** (see deviation below) |
| Rate | **0.7–0.8 it/s** (swap-free) |
| Wall clock | **3,873 s (~64.5 min)** |
| **Final training loss** | **1.4207** (cross-entropy; converging toward the paper's ~1.56 region and below it) |
| Output | `trained_models/brain_lm.pt` (31 MB) + `trained_models/brain_lm.pt.meta.json` provenance sidecar — **both present** |

**Sample generation @ step 3000:**
```
User: hello
Brain: Tone move way don't good don't was times anymorry, I'm and what have is.
User: Thanks, I want you will it we any work.
```
Real words, apostrophised contractions, and emergent `User:`/`Brain:` turn-taking —
a "small biological char-RNN," exactly as the README scopes it. Log: `runs/train_lm.txt`.

### ⚠️ Honest deviation from the documented config

The documented MPS "grok" recipe (`--grid-size 32 --target-mem-frac 0.9`) auto-tuned
to **batch = 8 (~14 GB peak)**. On this 16 GB machine that does **not** fit alongside
macOS: it drove **14.7 GB of swap** and crawled at **~0.05 it/s (~18 h projected)**.
The run was killed and restarted with a **fixed `--batch-size 4`** (same grid-32 /
32,768-neuron brain, same 3000 steps, same seq-len/lr/clip), which ran cleanly at
0.7–0.8 it/s with no hot-path thrash. **The README's "fills ~90 % of RAM at batch 6–8,
~14 GB" claim is optimistic once the OS footprint is counted on a 16 GB M1 Pro.**
The deviation (`batch_size: 4`) is recorded faithfully in the `.meta.json` `argv`.

---

## 5. Feature-path verification

| Path | Command | Result | Log |
|---|---|---|---|
| **5.1 Config-driven no-conductance ablation** | `train_language.py --additive-current --grid-size 12 --steps 200` | ✅ `ABLATION active: use_conductance`; loss 3.88 → **1.35**; saved | `runs/abl.txt`, `runs/abl.pt` |
| **5.2 Truncated BPTT** | `train_language.py --grid-size 12 --tbptt-chunk 32 --steps 200` | ✅ `truncated BPTT: chunk=32`; loss 3.81 → **1.59**; saved | `runs/tbptt.txt`, `runs/tbptt.pt` |
| **5.3 DPO hook** | `preference.dpo_finetune` on 4 toy triples (tiny grid-6 brain, CPU) | ✅ mean DPO loss **0.693 → 0.013**, finite & monotonically decreasing | `runs/dpo_smoke.py`, `runs/dpo_smoke.txt` |
| **5.4 Chat with trained model** | `interact.py --lm-path trained_models/brain_lm.pt --no-pretrained` | ✅ generates locally-coherent replies via the REPL | `runs/chat.txt` |

### ⚠️ Documentation bug found

The task and the README's reproduction section both invoke the trainer's
no-conductance ablation as **`--no-conductance`**, but `train_language.py` does
**not** define that flag — it errors with *"unrecognized arguments: --no-conductance"*.
The correct flag is **`--additive-current`** (→ `use_conductance=False`). The
`--no-conductance` spelling is only valid inside `matched_experiment.py` semantics,
not the trainer CLI. **README reproduction example fixed** to `--additive-current`.

---

## 6. Summary

| Step | Status |
|---|---|
| 1 · Environment (pyenv 3.11.9 + venv + deps) | ✅ |
| 2 · pytest (35/35) + import smoke | ✅ |
| 3 · Multi-seed matched experiment + doc updates | ✅ |
| 4 · Train generative brain (grid-32 SODA, MPS) | ✅ (final loss 1.42; config lowered, documented) |
| 5 · Feature paths (ablation, TBPTT, DPO, chat) | ✅ (with the two doc bugs above noted/fixed) |
| 6 · This report | ✅ |

**Bottom line:** the pipeline runs end-to-end on a 16 GB M1 Pro under MPS. The
multi-seed study **confirms the headline "conductance is the costly ingredient /
no-conductance beats the LSTM" across all three seeds**, adds a strong dense-RNN
baseline (1.57), and **softens the Dale's-law claim from "always diverges" to
"unstable and least-accurate, but seed-dependent."** Two documentation issues were
found and corrected (the 90 %-RAM batch claim, and the `--no-conductance` vs
`--additive-current` flag). No results were fabricated; every number here is from a
log in this directory.
