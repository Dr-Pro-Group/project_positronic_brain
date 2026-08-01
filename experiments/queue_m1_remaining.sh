#!/usr/bin/env bash
# Queue remaining laptop-scale science on this M1 Pro after replicate.py finishes.
# Does NOT start a second MPS trainer while PID REPLICATE_PID is alive.
#
#   bash experiments/queue_m1_remaining.sh
#   bash experiments/queue_m1_remaining.sh 91560   # explicit replicate PID
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/positronic_brain_v2/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG="$ROOT/runs/queue_m1_remaining.log"
REP_PID="${1:-91560}"

exec >>"$LOG" 2>&1
echo "======== queue start $(date) root=$ROOT py=$PY wait_pid=$REP_PID ========"

if kill -0 "$REP_PID" 2>/dev/null; then
  echo "[queue] waiting for replicate PID $REP_PID ..."
  while kill -0 "$REP_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[queue] replicate PID $REP_PID exited at $(date)"
else
  echo "[queue] PID $REP_PID not running — assuming replicate already finished"
fi

if [[ -f runs/replication.log ]]; then
  echo "[queue] --- last 15 lines of replication.log ---"
  tail -15 runs/replication.log || true
fi
if [[ -f runs/replication.json ]]; then
  "$PY" - <<'PY'
import json
d=json.load(open("runs/replication.json"))
print(f"[queue] replication.json: {len(d.get('runs',[]))} runs")
for r in d.get("runs", []):
    bpc = r.get("bpc")
    print(f"  {r.get('label')} s{r.get('seed')} bpc={bpc:.4f}" if bpc is not None else f"  {r}")
PY
fi

echo "[queue] PHASE 1 — matched_long seeds 43,44 (3000 steps, mode all)"
"$PY" experiments/matched_experiment.py \
  --mode all --grid-size 12 --steps 3000 --seq-len 48 --batch-size 16 \
  --seeds 43,44 --hf-chat soda --hf-chat-limit 4000 --repeats 60 \
  --device mps --json runs/matched_long_seeds43_44.json
echo "[queue] PHASE 1 done $(date)"

echo "[queue] PHASE 2 — lever seeds 43,44 (baseline / g_max 0.691 / stp @ G16)"
"$PY" experiments/lever_seeds.py \
  --seeds 43,44 --steps 3000 --grid 16 --device mps \
  --json runs/lever_seeds.json
echo "[queue] PHASE 2 done $(date)"

echo "======== QUEUE COMPLETE $(date) ========"
echo "[queue] artifacts:"
echo "  runs/replication.json"
echo "  runs/matched_long_seeds43_44.json"
echo "  runs/lever_seeds.json"
echo "  runs/queue_m1_remaining.log"
