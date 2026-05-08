#!/usr/bin/env bash
# run_experiments_3.sh — Larger system sizes for Binder resolution.
#
# Experiment E: Fine Binder at L=384 (16 μ × 32 seeds, 200k frames)
# Experiment F: Fine Binder at L=512 (16 μ × 32 seeds, 200k frames)
#
# NO SNAPSHOTS — uses --save-seeds for per-seed CSV instead.
# 8 workers to avoid CPU contention at large L.
# 12-hour timeout per job.
#
# Usage: bash run_experiments_3.sh 2>&1 | tee run_experiments_3.log

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="venv/bin/python3"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

# 8 workers for large L — avoids oversubscription on 32-core machine
W="--workers 24"
DYN32="--binary torus_dyn --reinforce-mins 4 --seeds 32 $W --max-degree 8 --save-seeds"

run_step() {
    local name="$1"
    shift
    local logfile="$LOGDIR/${name}.log"
    echo ""
    echo "====== [$name] START: $(date) ======"
    echo "Command: $*"
    echo "Log: $logfile"
    "$@" > "$logfile" 2>&1
    local rc=$?
    echo "====== [$name] DONE:  $(date) (exit $rc) ======"
    return $rc
}

echo "============================================"
echo " Experiments E+F — $(date)"
echo " Logs:      $LOGDIR"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# Experiment E: Fine Binder at L=384
# ══════════════════════════════════════════════════════════════════════════════

run_step "E_binder_L384_fine" \
    $PY analysis/sweep.py \
    $DYN32 --frames 200000 --stats-interval 10000 \
    --width 384 --height 384 \
    --mutation-probs 0.340 0.341 0.342 0.343 0.344 0.345 0.346 0.347 0.348 0.349 0.350 0.351 0.352 0.353 0.354 0.355

# ══════════════════════════════════════════════════════════════════════════════
# Experiment F: Fine Binder at L=512
# ══════════════════════════════════════════════════════════════════════════════

run_step "F_binder_L512_fine" \
    $PY analysis/sweep.py \
    $DYN32 --frames 200000 --stats-interval 10000 \
    --width 512 --height 512 \
    --mutation-probs 0.340 0.341 0.342 0.343 0.344 0.345 0.346 0.347 0.348 0.349 0.350 0.351 0.352 0.353 0.354 0.355

# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo " All experiments complete — $(date)"
echo "============================================"
