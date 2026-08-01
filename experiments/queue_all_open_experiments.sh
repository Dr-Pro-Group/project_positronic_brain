#!/usr/bin/env bash
# Full open-experiment queue for Mac Mini (D6–D8 + related).
# Sequential: one GPU/MPS job at a time. Logs to logs/exp_queue.log.
# Resume-safe: skips stages whose DONE marker already exists.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/logs/exp_queue.log"
MARKER_DIR="${ROOT}/runs/exp_markers"
mkdir -p logs runs "$MARKER_DIR" data

ts() { date '+%Y-%m-%d %H:%M:%S %z'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
done_mark() { touch "${MARKER_DIR}/$1.done"; }
is_done() { [[ -f "${MARKER_DIR}/$1.done" ]]; }

say "======== OPEN EXPERIMENT QUEUE START ========"
say "host=$(hostname) cwd=$ROOT"

# --------------------------------------------------------------------------- D6
# Param-fair hard tasks: match LSTM/CNN/GPT (and brain_wm) to brain_wm budget.
# Merges with historical plain-brain table conceptually; new JSON is fair suite.
if ! is_done d6_hard_match_wm; then
  say "==== D6: hard tasks match_to=brain_wm (lstm,cnn,gpt,brain_wm) ===="
  $PY experiments/hard_tasks_eval.py \
    --tasks delayed_copy,addition,associative \
    --models lstm,cnn,gpt,brain_wm \
    --match-to brain_wm \
    --grid-size 12 \
    --steps 8000 --batch-size 32 --eval-every 500 --n-eval 256 \
    --seed 42 --device mps \
    --json runs/hard_tasks_g12_match_wm.json \
    >>"$LOG" 2>&1
  ec=$?
  say "D6 exit=$ec"
  if [[ $ec -eq 0 ]]; then done_mark d6_hard_match_wm; else say "D6 FAILED"; exit $ec; fi
else
  say "SKIP D6 (marker present)"
fi

# Also include plain brain at same match budget? brain is lighter — already in
# hard_tasks_g12.json. Optional: brain at G=12 under match_wm for side-by-side
# with heavier baselines only.

# --------------------------------------------------------------------------- D7
# Equal-ish neuron count: modular 3×G12 (N=5184) vs single G=17 (N=4913).
# Shared TinyStories disk store from scale path.
if ! is_done d7_equal_n; then
  say "==== D7a: modular 3×G12 equal-N path ===="
  if [[ ! -f data/scale_tinystories/prepare_done.json ]]; then
    say "preparing tinystories disk store first"
    $PY experiments/scale_train.py prepare \
      --dataset tinystories --max-docs 5000 --max-chars 5000000 \
      --vocab-size 2048 --work-dir data/scale_tinystories \
      >>"$LOG" 2>&1
  fi
  $PY experiments/scale_train.py train-modular \
    --work-dir data/scale_tinystories \
    --grid-size 12 --n-areas 3 \
    --steps-per-area 800 --seq-len 64 --batch-size 4 \
    --eval-every 200 --inner-steps 2 \
    --device mps \
    --json runs/scale_modular_g12_equaln.json \
    >>"$LOG" 2>&1
  ec=$?
  say "D7 modular exit=$ec"
  [[ $ec -eq 0 ]] || { say "D7 modular FAILED"; exit $ec; }

  say "==== D7b: single G=17 near-equal N (4913 vs 5184) ===="
  $PY experiments/scale_train.py train-single \
    --work-dir data/scale_tinystories \
    --grid-size 17 \
    --steps 2400 --seq-len 64 --batch-size 4 \
    --eval-every 200 --inner-steps 2 \
    --grad-checkpoint --readout-width 256 \
    --device mps \
    --json runs/scale_single_g17_equaln.json \
    >>"$LOG" 2>&1
  ec=$?
  say "D7 single exit=$ec"
  if [[ $ec -eq 0 ]]; then done_mark d7_equal_n; else say "D7 single FAILED — continue queue"; fi
else
  say "SKIP D7 (marker present)"
fi

# --------------------------------------------------------------------------- D8
# FineWeb-Edu capped prepare + short modular/single train (feasibility).
if ! is_done d8_fineweb; then
  say "==== D8a: prepare fineweb-edu (capped) ===="
  $PY experiments/scale_train.py prepare \
    --dataset fineweb-edu \
    --max-docs 8000 --max-chars 15000000 \
    --vocab-size 4096 \
    --work-dir data/scale_fineweb_edu \
    >>"$LOG" 2>&1
  ec=$?
  say "D8 prepare exit=$ec"
  if [[ $ec -ne 0 ]]; then
    say "D8 prepare FAILED (network/HF?) — mark partial and continue"
  else
    say "==== D8b: train-single G=12 on FineWeb store ===="
    $PY experiments/scale_train.py train-single \
      --work-dir data/scale_fineweb_edu \
      --grid-size 12 --steps 1500 \
      --seq-len 64 --batch-size 4 --eval-every 250 \
      --grad-checkpoint --device mps \
      --json runs/scale_fineweb_single_g12.json \
      >>"$LOG" 2>&1
    ec2=$?
    say "D8 train exit=$ec2"
    if [[ $ec2 -eq 0 ]]; then done_mark d8_fineweb; fi
  fi
else
  say "SKIP D8 (marker present)"
fi

# --------------------------------------------------------------------------- D10
# Subword (existing store) already measured; run short char public LM slice for
# comparison is heavy — instead document scale store vs prior char Track A.
# Optional light: hard-tasks G=16 scale for brain + lstm (memory vs N).
if ! is_done d_scale_hard_g16; then
  say "==== EXTRA: hard tasks G=16 brain,brain_wm,lstm (scale N) ===="
  $PY experiments/hard_tasks_eval.py \
    --tasks delayed_copy,addition,associative \
    --models lstm,brain,brain_wm \
    --match-to brain \
    --grid-size 16 \
    --steps 6000 --batch-size 16 --eval-every 500 --n-eval 128 \
    --seed 42 --device mps \
    --json runs/hard_tasks_g16_scale.json \
    >>"$LOG" 2>&1
  ec=$?
  say "G16 hard tasks exit=$ec"
  if [[ $ec -eq 0 ]]; then done_mark d_scale_hard_g16; fi
else
  say "SKIP G16 hard (marker present)"
fi

# --------------------------------------------------------------------------- brain_wm public LM short (Track A follow-up)
if ! is_done d_brain_wm_tinystories; then
  say "==== EXTRA: public LM TinyStories brain vs brain_wm vs lstm (short) ===="
  $PY experiments/public_lm_eval.py \
    --hf tinystories --hf-limit 20000 \
    --grid-size 12 --steps 5000 \
    --seq-len 128 --batch-size 8 --eval-every 500 \
    --models lstm,brain,brain_wm \
    --device mps \
    --json runs/public_tinystories_brain_wm_short.json \
    >>"$LOG" 2>&1
  ec=$?
  say "brain_wm TinyStories exit=$ec"
  if [[ $ec -eq 0 ]]; then done_mark d_brain_wm_tinystories; fi
else
  say "SKIP brain_wm tinystories (marker present)"
fi

say "======== OPEN EXPERIMENT QUEUE COMPLETE ========"
# status summary
for m in d6_hard_match_wm d7_equal_n d8_fineweb d_scale_hard_g16 d_brain_wm_tinystories; do
  if is_done "$m"; then say "MARKER $m DONE"; else say "MARKER $m MISSING"; fi
done
ls -la runs/hard_tasks_g12_match_wm.json runs/scale_modular_g12_equaln.json \
  runs/scale_single_g17_equaln.json runs/scale_fineweb_single_g12.json \
  runs/hard_tasks_g16_scale.json runs/public_tinystories_brain_wm_short.json \
  2>>"$LOG" | tee -a "$LOG"
say "queue process exiting 0"
exit 0
