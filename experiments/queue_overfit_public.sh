#!/usr/bin/env bash
# After primary LLM intelligence queue frees MPS: public-data overfit stress.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/logs/overfit_public.log"
mkdir -p logs runs

ts() { date '+%Y-%m-%d %H:%M:%S %z'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

primary_busy() {
  # Match any primary LLM-track holder of MPS
  ps -axo command 2>/dev/null | grep -v grep | grep -Eq \
    'experiments/(llm_public_benchmark|public_lm_eval)\.py|queue_llm_intelligence\.sh' \
    && return 0
  return 1
}

say "======== OVERFIT PUBLIC LM QUEUE ========"
while primary_busy; do
  say "waiting for primary LLM intelligence work to finish..."
  sleep 60
done

if [[ ! -f data/llm_tinystories/prepare_done.json ]]; then
  say "preparing tinystories store"
  $PY experiments/scale_train.py prepare \
    --dataset tinystories --max-docs 20000 --max-chars 20000000 \
    --vocab-size 4096 --work-dir data/llm_tinystories >>"$LOG" 2>&1
fi

# Phase A: aggressive overfit on 1M-token frozen slice (~60 epochs at 60k steps)
say "==== PHASE A: 1M-token cap, 60k steps (~60 epochs) suite ===="
$PY experiments/overfit_public_lm.py \
  --work-dir data/llm_tinystories \
  --train-tokens-cap 1000000 \
  --steps 60000 --eval-every 1000 \
  --batch-size 8 --seq-len 128 \
  --models lstm,cnn,gpt,brain,brain_wm \
  --grid-size 12 --device mps \
  --lr 8e-4 --grad-clip 0.5 \
  --json runs/overfit_public_tinystories_1M_60k.json \
  >>"$LOG" 2>&1
ec=$?
say "phase A exit=$ec"

# Phase B: harder overfit on 256k freeze
say "==== PHASE B: 256k-token cap, 40k steps (~200 epochs) suite ===="
$PY experiments/overfit_public_lm.py \
  --work-dir data/llm_tinystories \
  --train-tokens-cap 256000 \
  --steps 40000 --eval-every 500 \
  --batch-size 8 --seq-len 128 \
  --models lstm,cnn,gpt,brain,brain_wm \
  --grid-size 12 --device mps \
  --lr 1e-3 --grad-clip 0.5 \
  --json runs/overfit_public_tinystories_256k_40k.json \
  >>"$LOG" 2>&1
ec=$?
say "phase B exit=$ec"
say "======== OVERFIT QUEUE COMPLETE ========"
ls -la runs/overfit_public_tinystories_*.json 2>&1 | tee -a "$LOG" || true
