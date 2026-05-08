#!/usr/bin/env bash
# run_all_replication.sh — reproduce ALL paper results with full snapshot data.
#
# Produces:
#   logs/          — per-experiment stdout/stderr logs
#   results/       — aggregated CSV + PNG from sweep.py
#   snapshots/     — per-seed node/edge CSV snapshots for post-hoc visualization
#
# Usage: bash run_all_replication.sh 2>&1 | tee run_all.log
#
# Seed counts:
#   64 seeds — all runs (statistically robust for physics)
#
# Frame counts:
#   50k  — fixed lattice phase diagram
#   30k  — adaptive network at L=64, L=128
#   100k — adaptive network at L=256 (longer equilibration)
#   30k  — τ at L=512 (was 10k, increased for equilibration)
#
# Estimated total runtime: ~12–16 hours (24 workers).
# Bottleneck: L=256 runs at 100k frames (~5–7 hours total).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="venv/bin/python3"
LOGDIR="$ROOT/logs"
SNAPDIR="$ROOT/snapshots"
mkdir -p "$LOGDIR"

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
echo " Full replication run — $(date)"
echo " Logs:      $LOGDIR"
echo " Snapshots: $SNAPDIR"
echo "============================================"

# ── Common flag sets ─────────────────────────────────────────────────────────
#
# All runs use 64 seeds for statistical robustness.

W="--workers 24"
SWEEP="--reinforce-mins 4 --seeds 64 $W"
DYN="--binary torus_dyn $SWEEP --max-degree 8"

# ══════════════════════════════════════════════════════════════════════════════
# Result 1: Fixed lattice phase diagram
#   Paper: Fig 1, Table 1, Sec III
#   64 seeds
#   50k frames, snapshot every 1000
# ══════════════════════════════════════════════════════════════════════════════

# DONE — completed Mon Mar 16 11:04:32 PM
# run_step "1_fixed_lattice_phase" \
#     $PY analysis/sweep.py \
#     $SWEEP --frames 50000 --stats-interval 1000 \
#     --mutation-rates 3 4 5 6 7 8 9 10 \
#     --snapshot-dir "$SNAPDIR/1_fixed_lattice_phase"

# ══════════════════════════════════════════════════════════════════════════════
# Result 2: Domain size exponent τ(L) — fixed lattice
#   Paper: Fig 2, Sec III
#   32 seeds per size (exponent fitting)
#   Parallelized with 24 workers
# ══════════════════════════════════════════════════════════════════════════════

# DONE — all completed by Tue Mar 17 12:54:11 AM
# run_step "2_tau_L64" \
#     $PY analysis/histogram.py \
#     --frames 5000 --seeds 32 --workers 24 --width 64 --height 64
#
# run_step "2_tau_L128" \
#     $PY analysis/histogram.py \
#     --frames 5000 --seeds 32 --workers 24 --width 128 --height 128
#
# run_step "2_tau_L256" \
#     $PY analysis/histogram.py \
#     --frames 10000 --seeds 32 --workers 24 --width 256 --height 256
#
# run_step "2_tau_L512" \
#     $PY analysis/histogram.py \
#     --frames 30000 --seeds 32 --workers 24 --width 512 --height 512

# ══════════════════════════════════════════════════════════════════════════════
# Result 3: Adaptive network phase diagram — coarse sweep
#   Paper: Fig 3, Table 2, Sec IV.A
#   64 seeds, integer rates
#   30k frames, snapshot every 500
# ══════════════════════════════════════════════════════════════════════════════

# DONE — completed Tue Mar 17 01:39:55 AM
# run_step "3_dyn_graph_phase" \
#     $PY analysis/sweep.py \
#     $DYN --frames 30000 --stats-interval 500 \
#     --mutation-rates 1 2 3 4 5 6 7 8 9 \
#     --snapshot-dir "$SNAPDIR/3_dyn_graph_phase"

# ══════════════════════════════════════════════════════════════════════════════
# Result 4: First-order transition probe — fine fractional sweep
#   Paper: Fig 3 (fine points), Fig 5 (bimodality), Fig 4 (variance), Sec IV
#   64 seeds for bimodality statistics at μ=0.35
#   Fine μ grid near transition: 0.33, 0.34, 0.345, 0.35, 0.355, 0.36, 0.37
#   Plus coarse points away from transition: 0.40, 0.45, 0.50
#   30k frames, snapshot every 500
# ══════════════════════════════════════════════════════════════════════════════

# DONE — completed Tue Mar 17 02:24:05 AM
# run_step "4_first_order_probe" \
#     $PY analysis/sweep.py \
#     $DYN --frames 30000 --stats-interval 500 \
#     --mutation-probs 0.33 0.34 0.345 0.35 0.355 0.36 0.37 0.40 0.45 0.50 \
#     --snapshot-dir "$SNAPDIR/4_first_order_probe"

# ══════════════════════════════════════════════════════════════════════════════
# Result 5: Hysteresis — random vs ordered init
#   Paper: Fig 6, Table 3, Sec IV.D
#   64 seeds — need good statistics on basin partitioning
#   Fine grid across bistable window: 0.34, 0.345, 0.35, 0.355, 0.36
#   30k frames, snapshot every 500
# ══════════════════════════════════════════════════════════════════════════════

# DONE — all 10 hysteresis runs completed by Tue Mar 17 03:13:20 AM
# for MU in 0.34 0.345 0.35 0.355 0.36; do
#     MU_TAG=$(echo "$MU" | tr -d '.')
#     run_step "5_hysteresis_${MU_TAG}_random" \
#         $PY analysis/sweep.py \
#         $DYN --frames 30000 --stats-interval 500 \
#         --mutation-probs $MU \
#         --snapshot-dir "$SNAPDIR/5_hysteresis/${MU_TAG}_random"
#
#     run_step "5_hysteresis_${MU_TAG}_ordered" \
#         $PY analysis/sweep.py \
#         $DYN --frames 30000 --stats-interval 500 \
#         --mutation-probs $MU --ordered-init \
#         --snapshot-dir "$SNAPDIR/5_hysteresis/${MU_TAG}_ordered"
# done

# ══════════════════════════════════════════════════════════════════════════════
# Result 6: Finite-size dependence — full μ sweep at L=64, L=256
#   Paper: Fig 7, Sec V
#   64 seeds at transition-relevant μ values
#   L=64: 30k frames, L=256: 100k frames (equilibration)
# ══════════════════════════════════════════════════════════════════════════════

PROBS_FINE="0.33 0.34 0.345 0.35 0.355 0.36 0.37 0.40 0.45 0.50"

# DONE — completed Tue Mar 17 03:23:06 AM
# run_step "6_size_dep_L64" \
#     $PY analysis/sweep.py \
#     $DYN --frames 30000 --stats-interval 500 \
#     --mutation-probs $PROBS_FINE \
#     --width 64 --height 64 \
#     --snapshot-dir "$SNAPDIR/6_size_dep_L64"

# PARTIAL — mr=0.33 (64 seeds), mr=0.34 (64 seeds), mr=0.345 (29 seeds) complete
# Remaining rates run with 32 seeds, stats-interval 5000
PROBS_L256_REMAINING="0.35 0.355 0.36 0.37 0.40 0.45 0.50"
DYN32="--binary torus_dyn --reinforce-mins 4 --seeds 32 $W --max-degree 8"

run_step "6_size_dep_L256_remaining" \
    $PY analysis/sweep.py \
    $DYN32 --frames 100000 --stats-interval 5000 \
    --mutation-probs $PROBS_L256_REMAINING \
    --width 256 --height 256 \
    --snapshot-dir "$SNAPDIR/6_size_dep_L256"

# ══════════════════════════════════════════════════════════════════════════════
# Result 7: FSS at μ=0.35 — per-seed scatter by system size
#   Paper: Fig 8, Table 4, Sec V
#   64 seeds per condition (random + ordered) at each L
#   L=64, 128: 30k frames; L=256: 100k frames
# ══════════════════════════════════════════════════════════════════════════════

for L in 64 128 256; do
    if [ "$L" -eq 256 ]; then
        FRAMES=100000
        SI=5000
    else
        FRAMES=30000
        SI=500
    fi

    SIZE_FLAG=""
    if [ "$L" -ne 128 ]; then
        SIZE_FLAG="--width $L --height $L"
    fi

    run_step "7_fss_035_L${L}_random" \
        $PY analysis/sweep.py \
        $DYN32 --frames $FRAMES --stats-interval $SI \
        --mutation-probs 0.35 $SIZE_FLAG \
        --snapshot-dir "$SNAPDIR/7_fss_035/L${L}_random"

    run_step "7_fss_035_L${L}_ordered" \
        $PY analysis/sweep.py \
        $DYN32 --frames $FRAMES --stats-interval $SI \
        --mutation-probs 0.35 $SIZE_FLAG --ordered-init \
        --snapshot-dir "$SNAPDIR/7_fss_035/L${L}_ordered"
done

# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo " All replication runs complete — $(date)"
echo " Logs:      $LOGDIR/"
echo " Results:   $ROOT/results/"
echo " Snapshots: $SNAPDIR/"
echo "============================================"
