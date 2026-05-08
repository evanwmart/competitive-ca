#!/usr/bin/env bash
# run_experiments_2.sh — Two experiments for paper strengthening.
#
# Experiment C: Fine Binder at L=256 (16 μ × 32 seeds, 100k frames)
# Experiment D: k_max=4 adaptive control (L=128, 64 seeds, 30k frames)
#
# Usage: bash run_experiments_2.sh 2>&1 | tee run_experiments_2.log

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="venv/bin/python3"
LOGDIR="$ROOT/logs"
SNAPDIR="$ROOT/snapshots"
mkdir -p "$LOGDIR"

W="--workers 28"

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
echo " Experiments C+D — $(date)"
echo " Logs:      $LOGDIR"
echo " Snapshots: $SNAPDIR"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# Experiment D: k_max=4 adaptive control (runs faster, do first)
#   Adaptive network with k_max=4 (matching fixed lattice degree)
#   Same μ grid as the main adaptive sweep, 64 seeds, L=128, 30k frames
#   If discontinuity persists → degree range isn't the driver
#   If it vanishes → degree cap is load-bearing
# ══════════════════════════════════════════════════════════════════════════════

run_step "D_kmax4_control" \
    $PY analysis/sweep.py \
    --binary torus_dyn --reinforce-mins 4 --seeds 64 $W --max-degree 4 \
    --frames 30000 --stats-interval 500 \
    --mutation-probs 0.15 0.17 0.19 0.20 0.21 0.22 0.23 0.25 0.27 0.30 0.33 0.35 0.37 0.40 0.45 0.50 \
    --snapshot-dir "$SNAPDIR/D_kmax4_control"

# ══════════════════════════════════════════════════════════════════════════════
# Experiment C: Fine Binder at L=256
#   Dense μ grid near pseudocritical point
#   μ ∈ [0.340, 0.355] step 0.001, 32 seeds, 100k frames
#   Snapshots for per-seed Binder cumulant computation
# ══════════════════════════════════════════════════════════════════════════════

run_step "C_binder_L256_fine" \
    $PY analysis/sweep.py \
    --binary torus_dyn --reinforce-mins 4 --seeds 32 $W --max-degree 8 \
    --frames 100000 --stats-interval 5000 \
    --width 256 --height 256 \
    --mutation-probs 0.340 0.341 0.342 0.343 0.344 0.345 0.346 0.347 0.348 0.349 0.350 0.351 0.352 0.353 0.354 0.355 \
    --snapshot-dir "$SNAPDIR/C_binder_L256_fine"

# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo " All experiments complete — $(date)"
echo " Logs:      $LOGDIR/"
echo " Snapshots: $SNAPDIR/"
echo "============================================"
