#!/usr/bin/env bash
# Wait for M1 matched_long + Mini lever_seeds, then pull Mini results and summarize.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/runs/monitor_integrate.log"
echo "[mon] start $(date)" | tee -a "$LOG"

wait_m1() {
  while ps -axo command= | awk '/matched_experiment\.py/ && !/awk/ {found=1} END{exit !found}'; do
    sleep 60
  done
  echo "[mon] M1 matched done $(date)" | tee -a "$LOG"
}

wait_mini() {
  while ssh -o BatchMode=yes -o ConnectTimeout=15 mac-mini \
    'ps -axo command= | awk "/lever_seeds\\.py/ && !/awk/ {found=1} END{exit !found}"' 2>/dev/null; do
    sleep 120
  done
  echo "[mon] Mini lever done $(date)" | tee -a "$LOG"
}

wait_m1
wait_mini

echo "[mon] pulling Mini artifacts $(date)" | tee -a "$LOG"
rsync -az mac-mini:~/code/project_positronic_brain/runs/lever_seeds.json \
  "$ROOT/runs/lever_seeds.json" 2>&1 | tee -a "$LOG"
rsync -az mac-mini:~/code/project_positronic_brain/logs/lever_seeds.log \
  "$ROOT/runs/lever_seeds_mini.log" 2>&1 | tee -a "$LOG"

python3 - <<'PY' | tee -a "$LOG"
import json, statistics as st, pathlib
root = pathlib.Path("/Users/compte_27/code/LaurentAIA/Projects/project_positronic_brain")
print("=== INTEGRATION SUMMARY ===")
ml = root / "runs/matched_long_seeds43_44.json"
if ml.exists():
    d = json.loads(ml.read_text())
    print("matched_long seeds", d.get("seeds"), "n_results", len(d.get("results", d.get("runs", []))))
    res = d.get("results") or d.get("runs") or {}
    if isinstance(res, dict):
        for k,v in res.items():
            print(" ", k, v)
    else:
        for r in res[:20]:
            print(" ", r)
else:
    print("matched_long_seeds43_44.json MISSING")

lv = root / "runs/lever_seeds.json"
if lv.exists():
    d = json.loads(lv.read_text())
    print("lever runs", len(d.get("runs",[])))
    from collections import defaultdict
    by = defaultdict(list)
    for r in d.get("runs", []):
        if "bpc" in r:
            by[r["label"]].append(r["bpc"])
    # seed 42 from prior
    prior = {
        "baseline": [2.1524],
        "g_max 0.691": [2.0393],
        "stp": [2.0679],
    }
    for lab, xs in by.items():
        allx = prior.get(lab, []) + xs
        print(f"  {lab}: n={len(allx)} mean={st.mean(allx):.4f} values={[round(x,4) for x in allx]}")
else:
    print("lever_seeds.json MISSING")

# density already frozen
print("density 3-seed already in RESULTS.md")
print("=== END ===")
PY

echo "[mon] complete $(date)" | tee -a "$LOG"
