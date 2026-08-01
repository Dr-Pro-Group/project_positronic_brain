# How we train and scale brain models

This is the practical curriculum — not only “run an experiment”, but **learn the
lifecycle** of a biomimetic LM: pretrain → save → fine-tune → align → scale.

---

## 1. Lifecycle (what “done” means)

```
public data (disk + BPE)
    → pretrain full suite (lstm, cnn, gpt, brain, brain_wm)
    → SAVE best checkpoints
    → evaluate (bpc / ppl / bpb + samples)
    → fine-tune (more data / domain / longer on fixed set)
    → optional DPO / preference alignment
    → scale (larger G, modular areas, more tokens)
```

Intelligence claims live on **public data + standard LM metrics**  
([`INTELLIGENCE_PROGRAM.md`](INTELLIGENCE_PROGRAM.md)).  
Checkpoints make the rest of the lifecycle real.

---

## 2. Saving models (always)

Primary bench and overfit runs write:

```
checkpoints/<run_name>/
  meta.json
  tokenizer.json
  lstm.pt
  cnn.pt
  gpt.pt
  brain.pt
  brain_wm.pt
```

API: `positronic_brain/checkpoints.py`

```python
from positronic_brain.checkpoints import load_brain, save_brain, load_baseline

model, tok, extra = load_brain("checkpoints/llm_bench_wikitext_g12/brain_wm.pt", device="mps")
# extra["metrics"] has best_val_bpc, etc.
```

CLI flags:

```bash
python experiments/llm_public_benchmark.py \
  --work-dir data/llm_tinystories \
  --checkpoint-dir checkpoints/llm_bench_tinystories_g12 \
  --json runs/llm_bench_tinystories_g12.json
```

Default if `--checkpoint-dir` omitted: `checkpoints/<json basename>/`.

---

## 3. Fine-tune / continue training (SFT)

```bash
python experiments/finetune_from_checkpoint.py \
  --checkpoint checkpoints/llm_bench_wikitext_g12/brain_wm.pt \
  --work-dir data/llm_tinystories \
  --steps 5000 --lr 2e-4 \
  --out-checkpoint checkpoints/ft_wiki_then_stories/brain_wm.pt \
  --json runs/finetune_brain_wm.json
```

Use **lower LR** than pretrain. Same public-data protocol; only the weight init
and step budget change.

---

## 4. Preference / “RL-like” alignment (DPO now, PPO later)

**Today (runnable):** Direct Preference Optimisation — no reward model, frozen
reference copy, preference triples.

```bash
python experiments/dpo_from_checkpoint.py \
  --checkpoint checkpoints/llm_bench_wikitext_g12/brain_wm.pt \
  --pairs data/preferences_example.jsonl \
  --write-example-pairs \
  --epochs 2 --beta 0.1 --lr 1e-4 \
  --out-checkpoint checkpoints/dpo_brain_wm/brain_wm.pt
```

Core code: `positronic_brain/preference.py`  
- `dpo_finetune` — production path for alignment experiments  
- `BrainPolicy` — contract for a future PPO / actor-critic loop (membrane `V` =
  recurrent policy state)

**Honest scope:** DPO only pays after supervised LM quality is coherent. It is a
**hook on the training curriculum**, not a claim that the brain is RL-aligned.

---

## 5. Scaling ladder (how to grow)

| Stage | What | Tooling |
|---|---|---|
| S0 | G=12, short public LM | `llm_public_benchmark.py` |
| S1 | Longer / overfit on fixed slice | `overfit_public_lm.py` |
| S2 | More data (FineWeb caps ↑) | `scale_train.py prepare` |
| S3 | Larger cube G=16–24 + grad ckpt | `scale_train.py train-single` |
| S4 | Multi-area load-one-at-a-time | `modular.py` + `train-modular` |
| S5 | Fine-tune from S0–S4 ckpts | `finetune_from_checkpoint.py` |
| S6 | Preference align | `dpo_from_checkpoint.py` |

Neuron-count honesty (larva → monkey) stays in RESULTS “Scale & training
perspective”. Scaling **params/data** is not the same as scaling **biological N**.

---

## 6. What we measure at each stage

| Stage | Metric |
|---|---|
| Pretrain / fine-tune | val/test **bpc, ppl, bpb**, samples |
| Overfit stress | train_bpc, val_bpc, **gap**, val_rise_after_best |
| DPO | preference loss (and later win-rate vs reference) |
| Scale | same LM metrics at larger G / more tokens |

Never rank “intelligence” by toy addition if CNN wins — see intelligence program.

---

## 7. Mini queue order (current)

1. Public benches TinyStories / WikiText / FineWeb → **save ckpts going forward**
2. Char continuity suite  
3. Overfit phases A/B (many epochs, fixed data) → **save ckpts**  
4. Optional: finetune + DPO smoke on best `brain_wm.pt`

---

## 8. Learning goals (for us as builders)

1. **Reproduce** a public LM run from a checkpoint  
2. **Continue** training without destroying val bpc (LR discipline)  
3. **Transfer** across corpora (WikiText → TinyStories fine-tune)  
4. **Align** with preferences without full PPO  
5. **Scale** only after S0 metrics and ckpts exist  

The positronic brain is a **research substrate**: every run should leave weights
we can stand on for the next experiment.
