#!/usr/bin/env bash
# Public LM qualification queue (run on Mini or M1 when free).
# Phase 1: TinyStories matched eval. Phase 2: WikiText-2.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; fi
if [[ -x positronic_brain_v2/.venv/bin/python ]]; then PY=positronic_brain_v2/.venv/bin/python; fi

mkdir -p runs logs
LOG=logs/queue_public_lm.log
exec > >(tee -a "$LOG") 2>&1
echo "======== public LM queue start $(date) py=$PY ========"

echo "[queue] PHASE 1 — TinyStories"
"$PY" experiments/public_lm_eval.py \
  --hf tinystories --hf-limit 100000 \
  --grid-size 12 --steps 20000 --seq-len 128 --batch-size 16 \
  --eval-every 500 --models lstm,rnn,gpt,brain \
  --seed 42 --device mps \
  --json runs/public_tinystories_g12.json \
  --samples-json runs/public_tinystories_g12_samples.json

echo "[queue] PHASE 2 — WikiText-2"
"$PY" experiments/public_lm_eval.py \
  --hf wikitext --hf-limit 50000 \
  --grid-size 12 --steps 15000 --seq-len 128 --batch-size 16 \
  --eval-every 500 --models lstm,rnn,gpt,brain \
  --seed 42 --device mps \
  --json runs/public_wikitext2_g12.json \
  --samples-json runs/public_wikitext2_g12_samples.json

echo "======== public LM queue COMPLETE $(date) ========"
