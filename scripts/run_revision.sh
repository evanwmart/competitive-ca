#!/usr/bin/env bash
# run_revision.sh — revision experiments for paper strengthening.
#
# Experiment A: Local-only edge formation control (L=128, 64 seeds)
# Experiment B: Fine μ Binder resolution at L=128 and L=256
#
# Usage: bash run_revision.sh 2>&1 | tee run_revision.log

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="venv/bin/python3"
LOGDIR="$ROOT/logs"
SNAPDIR="$ROOT/snapshots"
mkdir -p "$LOGDIR"

W="--workers 24"
DYN="--binary torus_dyn --reinforce-mins 4 --seeds 64 $W --max-degree 8"
DYN32="--binary torus_dyn --reinforce-mins 4 --seeds 32 $W --max-degree 8"

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
echo " Revision experiments — $(date)"
echo " Logs:      $LOGDIR"
echo " Snapshots: $SNAPDIR"
echo "============================================"

# ══════════════════════════════════════════════════════════════════════════════
# Experiment A: Local-only edge formation control
#   Same as exp 3+4 but with --local-formation
#   Coarse integer rates + fine fractional sweep near transition
#   64 seeds, L=128, 30k frames
# ══════════════════════════════════════════════════════════════════════════════

run_step "A_local_coarse" \
    $PY analysis/sweep.py \
    $DYN --frames 30000 --stats-interval 500 \
    --local-formation \
    --mutation-rates 1 2 3 4 5 6 7 8 9 \
    --snapshot-dir "$SNAPDIR/A_local_coarse"

run_step "A_local_fine" \
    $PY analysis/sweep.py \
    $DYN --frames 30000 --stats-interval 500 \
    --local-formation \
    --mutation-probs 0.33 0.34 0.345 0.35 0.355 0.36 0.37 0.40 0.45 0.50 \
    --snapshot-dir "$SNAPDIR/A_local_fine"

# ══════════════════════════════════════════════════════════════════════════════
# Experiment B: Fine μ resolution for Binder cumulant
#   Dense μ grid near pseudocritical point at each L
#   L=128: μ ∈ [0.345, 0.360] step 0.001, 64 seeds, 30k frames
#   L=256: μ ∈ [0.340, 0.355] step 0.001, 32 seeds, 100k frames
# ══════════════════════════════════════════════════════════════════════════════

run_step "B_binder_L128_fine" \
    $PY analysis/sweep.py \
    $DYN --frames 30000 --stats-interval 500 \
    --mutation-probs 0.345 0.346 0.347 0.348 0.349 0.350 0.351 0.352 0.353 0.354 0.355 0.356 0.357 0.358 0.359 0.360 \
    --snapshot-dir "$SNAPDIR/B_binder_L128_fine"

run_step "B_binder_L256_fine" \
    $PY analysis/sweep.py \
    $DYN32 --frames 100000 --stats-interval 5000 \
    --width 256 --height 256 \
    --mutation-probs 0.340 0.341 0.342 0.343 0.344 0.345 0.346 0.347 0.348 0.349 0.350 0.351 0.352 0.353 0.354 0.355 \
    --snapshot-dir "$SNAPDIR/B_binder_L256_fine"

# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo " All revision experiments complete — $(date)"
echo " Logs:      $LOGDIR/"
echo " Snapshots: $SNAPDIR/"
echo "============================================"
