#!/usr/bin/env bash
# After matched_long on M1 finishes, kill M1 lever_seeds if the queue starts it.
# Mini owns lever multi-seed (runs/lever_seeds.json on mac-mini).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/runs/queue_m1_remaining.log"
echo "[watchdog] start $(date)" >>"$LOG"

# Wait until no matched_experiment.py is running
while true; do
  if ! ps -axo command | grep -F '[m]atched_experiment.py' >/dev/null 2>&1 \
     && ! ps -axo command | grep -E 'python.*matched_experiment\.py' | grep -v grep >/dev/null 2>&1; then
    # double-check with pgrep -x not available; use ps awk
    n=$(ps -axo pid=,command= | awk '/matched_experiment\.py/ && !/awk/ {print}' | wc -l | tr -d ' ')
    if [[ "$n" -eq 0 ]]; then
      break
    fi
  fi
  sleep 30
done
echo "[watchdog] matched_experiment finished $(date)" >>"$LOG"

for i in $(seq 1 180); do
  # Kill only lever_seeds.py on this host
  ps -axo pid=,command= | awk '/lever_seeds\.py/ && !/awk/ {print $1}' | while read -r pid; do
    kill "$pid" 2>/dev/null || true
    echo "[watchdog] killed M1 lever_seeds pid=$pid $(date)" >>"$LOG"
  done
  # If queue script gone, stop
  nq=$(ps -axo command= | awk '/queue_m1_remaining\.sh/ && !/awk/ {print}' | wc -l | tr -d ' ')
  nl=$(ps -axo command= | awk '/lever_seeds\.py/ && !/awk/ {print}' | wc -l | tr -d ' ')
  if [[ "$nq" -eq 0 && "$nl" -eq 0 ]]; then
    echo "[watchdog] queue idle, exit $(date)" >>"$LOG"
    exit 0
  fi
  sleep 5
done
echo "[watchdog] timeout $(date)" >>"$LOG"
