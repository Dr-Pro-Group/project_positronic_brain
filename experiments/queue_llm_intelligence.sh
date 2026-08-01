#!/usr/bin/env bash
# Primary intelligence queue — public LLM data + standard metrics ONLY.
# Hard synthetic tasks are NOT in this queue (see docs/INTELLIGENCE_PROGRAM.md).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/logs/llm_intelligence.log"
MARK="${ROOT}/runs/exp_markers_llm"
mkdir -p logs runs "$MARK" data

ts() { date '+%Y-%m-%d %H:%M:%S %z'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
done_m() { touch "${MARK}/$1.done"; }
is_done() { [[ -f "${MARK}/$1.done" ]]; }

say "======== LLM INTELLIGENCE QUEUE START ========"
say "protocol: public data + bpc/ppl/bpb + samples | suite: lstm,rnn,cnn,gpt,brain,brain_wm"

# --- 1) TinyStories disk store (reuse if present) + full suite benchmark
if ! is_done tinystories_prepare; then
  say "==== prepare TinyStories disk (public LLM-family text) ===="
  $PY experiments/scale_train.py prepare \
    --dataset tinystories --max-docs 20000 --max-chars 20000000 \
    --vocab-size 4096 --work-dir data/llm_tinystories \
    >>"$LOG" 2>&1
  ec=$?
  say "tinystories prepare exit=$ec"
  [[ $ec -eq 0 ]] && done_m tinystories_prepare || exit $ec
else
  say "SKIP tinystories prepare"
fi

if ! is_done tinystories_bench; then
  say "==== FULL SUITE on TinyStories (primary metrics) ===="
  $PY experiments/llm_public_benchmark.py \
    --work-dir data/llm_tinystories \
    --models lstm,rnn,cnn,gpt,brain,brain_wm \
    --grid-size 12 --steps 10000 --seq-len 128 --batch-size 8 \
    --eval-every 500 --device mps \
    --json runs/llm_bench_tinystories_g12.json \
    >>"$LOG" 2>&1
  ec=$?
  say "tinystories bench exit=$ec"
  [[ $ec -eq 0 ]] && done_m tinystories_bench || say "tinystories bench FAILED ec=$ec"
else
  say "SKIP tinystories bench"
fi

# --- 2) WikiText via prepare path if available (Salesforce/wikitext through HF stream preset)
if ! is_done wikitext_prepare; then
  say "==== prepare WikiText-2 disk ===="
  $PY experiments/scale_train.py prepare \
    --dataset wikitext --max-docs 100000 --max-chars 15000000 \
    --vocab-size 4096 --work-dir data/llm_wikitext \
    >>"$LOG" 2>&1
  ec=$?
  say "wikitext prepare exit=$ec"
  [[ $ec -eq 0 ]] && done_m wikitext_prepare || say "wikitext prepare failed (continue)"
else
  say "SKIP wikitext prepare"
fi

if is_done wikitext_prepare && ! is_done wikitext_bench; then
  say "==== FULL SUITE on WikiText ===="
  $PY experiments/llm_public_benchmark.py \
    --work-dir data/llm_wikitext \
    --models lstm,rnn,cnn,gpt,brain,brain_wm \
    --grid-size 12 --steps 8000 --seq-len 128 --batch-size 8 \
    --eval-every 500 --device mps \
    --json runs/llm_bench_wikitext_g12.json \
    >>"$LOG" 2>&1
  ec=$?
  say "wikitext bench exit=$ec"
  [[ $ec -eq 0 ]] && done_m wikitext_bench
fi

# --- 3) FineWeb-Edu (real open LLM pretrain mix, capped)
if ! is_done fineweb_prepare; then
  say "==== prepare FineWeb-Edu (capped open LLM data) ===="
  $PY experiments/scale_train.py prepare \
    --dataset fineweb-edu --max-docs 20000 --max-chars 40000000 \
    --vocab-size 8192 --work-dir data/llm_fineweb_edu \
    >>"$LOG" 2>&1
  ec=$?
  say "fineweb prepare exit=$ec"
  [[ $ec -eq 0 ]] && done_m fineweb_prepare || say "fineweb prepare FAILED (HF/network?)"
else
  say "SKIP fineweb prepare"
fi

if is_done fineweb_prepare && ! is_done fineweb_bench; then
  say "==== FULL SUITE on FineWeb-Edu ===="
  $PY experiments/llm_public_benchmark.py \
    --work-dir data/llm_fineweb_edu \
    --models lstm,rnn,cnn,gpt,brain,brain_wm \
    --grid-size 12 --steps 12000 --seq-len 128 --batch-size 8 \
    --eval-every 500 --device mps \
    --json runs/llm_bench_fineweb_g12.json \
    >>"$LOG" 2>&1
  ec=$?
  say "fineweb bench exit=$ec"
  [[ $ec -eq 0 ]] && done_m fineweb_bench
fi

# --- 4) Legacy char-level Track A continuity (optional short, full suite including CNN)
if ! is_done char_tinystories_suite; then
  say "==== char-level public_lm TinyStories full suite (continuity with §13) ===="
  $PY experiments/public_lm_eval.py \
    --hf tinystories --hf-limit 30000 \
    --grid-size 12 --steps 8000 \
    --seq-len 128 --batch-size 8 --eval-every 500 \
    --models lstm,rnn,cnn,gpt,brain,brain_wm \
    --device mps \
    --json runs/public_tinystories_fullsuite_g12.json \
    >>"$LOG" 2>&1
  ec=$?
  say "char tinystories suite exit=$ec"
  [[ $ec -eq 0 ]] && done_m char_tinystories_suite
fi

say "======== LLM INTELLIGENCE QUEUE COMPLETE ========"
for m in tinystories_prepare tinystories_bench wikitext_prepare wikitext_bench fineweb_prepare fineweb_bench char_tinystories_suite; do
  if is_done "$m"; then say "MARKER $m DONE"; else say "MARKER $m MISSING"; fi
done
ls -la runs/llm_bench_*.json runs/public_tinystories_fullsuite_g12.json 2>>"$LOG" | tee -a "$LOG" || true
say "exit 0"
exit 0
