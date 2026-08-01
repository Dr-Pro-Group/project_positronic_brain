#!/usr/bin/env bash
# Wait for public LM WikiText job, re-run WikiText properly (fixed corpus loader),
# then start Track B brain-like training on TinyStories.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
elif [[ -x positronic_brain_v2/.venv/bin/python ]]; then PY=positronic_brain_v2/.venv/bin/python
else PY=python3
fi
LOG=logs/queue_finish_and_track_b.log
mkdir -p runs logs
exec > >(tee -a "$LOG") 2>&1
echo "======== finish+trackB start $(date) py=$PY ========"

# Wait until no public_lm_eval / old queue is running
while ps -axo command= | awk '/public_lm_eval\.py|queue_public_lm\.sh/ && !/awk/ && !/queue_finish/ {found=1} END{exit !found}'; do
  echo "[wait] public LM still running $(date)"
  sleep 120
done
echo "[wait] public LM free $(date)"

echo "[phase] WikiText-2 re-run with fixed non-streaming corpus + gpt"
"$PY" experiments/public_lm_eval.py \
  --hf wikitext --hf-limit 100000 \
  --grid-size 12 --steps 15000 --seq-len 128 --batch-size 16 \
  --eval-every 500 --models lstm,rnn,gpt,brain \
  --seed 42 --device mps \
  --json runs/public_wikitext2_g12.json \
  --samples-json runs/public_wikitext2_g12_samples.json

echo "[phase] Track B — brain-like training on TinyStories"
"$PY" experiments/brain_training_eval.py \
  --hf tinystories --hf-limit 100000 \
  --grid-size 12 --steps 15000 --seq-len 128 --batch-size 16 \
  --eval-every 500 \
  --regimes bptt,persistent,eprop \
  --with-lstm \
  --seed 42 --device mps \
  --json runs/brain_training_tinystories.json

echo "======== finish+trackB COMPLETE $(date) ========"
