# Scale implementation — disk data, modular areas, public LM corpora

**Status:** implemented MVP (code + offline tests). Large FineWeb runs are
opt-in on Mini; TinyStories is the default smoke path.

## What was built

| Module | Role |
|---|---|
| `positronic_brain/subword.py` | Pure-Python BPE (no `tokenizers` dep) |
| `positronic_brain/disk_data.py` | Stream public LM → shards → memmap tokens |
| `positronic_brain/modular.py` | Multi-area brain + pathway + area disk save/load |
| `experiments/scale_train.py` | `prepare` / `train-single` / `train-modular` CLI |

## Can we load “layers” one by one?

**Yes, as areas — not GPT layers.** A Positronic Brain is a sparse 3D graph, not
a stack of Transformer blocks. Scaling loads:

1. One **area** (small cube) on MPS/CPU  
2. Freezes other areas  
3. Optionally **saves area weights to disk** and reloads later  

See `ModularBrainLM.set_active_area` / `offload_area_to_disk` / `reload_area_from_disk`.

## Can hard drive act as RAM?

| Use | Supported? |
|---|---|
| Corpus on SSD (shards + `numpy.memmap` tokens) | **Yes** — primary path |
| Area checkpoints on disk between stages | **Yes** |
| Page every `edge_index` scatter through SSD each step | **No** (too slow) |

Disk is for **data + cold weights**, not for HBM replacement on every `step()`.

## Public LLM datasets

Presets in `disk_data.PUBLIC_LM_PRESETS`:

- `tinystories`, `wikitext` — small / eval  
- `fineweb-edu`, `fineweb`, `c4`, `openwebtext` — pretrain-scale (cap with `--max-docs` / `--max-chars`)

```bash
python experiments/scale_train.py prepare \
  --dataset tinystories --max-docs 5000 --max-chars 5000000 \
  --work-dir data/scale_tinystories

python experiments/scale_train.py train-modular \
  --work-dir data/scale_tinystories --grid-size 12 --n-areas 3 \
  --steps-per-area 1000 --device mps --json runs/scale_modular.json
```

FineWeb example (overnight, bandwidth-heavy):

```bash
python experiments/scale_train.py prepare \
  --dataset fineweb-edu --max-docs 20000 --max-chars 50000000 \
  --work-dir data/scale_fineweb_edu --vocab-size 8192
```

## Honest limits

1. **Sequential BPTT** still dominates time — modular areas cap the *graph* size,
   not total FLOPs if you still step every area every token.  
2. **Param match / readout**: large `Linear(N, vocab)` remains a scaling landmine;
   use `--readout-width` on single-cube runs.  
3. **BPE quality**: pure-Python trainer is basic (good enough for science runs,
   not GPT-4 token parity).  
4. **Larva scale**: G=12 area × 3 ≈ 5k neurons — still invertebrate-class total.

## Tests

```bash
PYTHONPATH=. pytest tests/test_scale_path.py -q
```
