#!/bin/bash
# Tier 2.1 — Additional 32 seeds at L=256 and L=384.
#
# L=256: 8 chunks of 4 seeds each (~5h per chunk)
# L=384: 32 chunks of 1 seed each (~7.3h per chunk)
#
# Usage:
#   ./run_tier2.sh <size> <chunk>
#
# Examples:
#   ./run_tier2.sh 256 1     # L=256, chunk 1/8 (seeds 32-35), ~5h
#   ./run_tier2.sh 256 8     # L=256, chunk 8/8 (seeds 60-63), ~5h
#   ./run_tier2.sh 384 1     # L=384, chunk 1/32 (seed 32), ~7.3h
#   ./run_tier2.sh 384 32    # L=384, chunk 32/32 (seed 63), ~7.3h
#
# Run chunks in any order. Each writes its own CSV to results/.
# When all chunks for a size are done, merge and recompute Binder:
#   venv/bin/python3 analysis/merge_seeds.py

set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=${1:?Usage: run_tier2.sh <256|384> <chunk>}
CHUNK=${2:?Usage: run_tier2.sh <256|384> <chunk>}

MU_GRID="0.340 0.341 0.342 0.343 0.344 0.345 0.346 0.347 0.348 0.349 0.350 0.351 0.352 0.353 0.354 0.355"
WORKERS=24

case $SIZE in
    256)
        FRAMES=100000
        SEEDS_PER_CHUNK=4
        MAX_CHUNKS=8
        ;;
    384)
        FRAMES=200000
        SEEDS_PER_CHUNK=1
        MAX_CHUNKS=32
        ;;
    *)
        echo "Error: size must be 256 or 384"
        exit 1
        ;;
esac

if [ "$CHUNK" -lt 1 ] || [ "$CHUNK" -gt "$MAX_CHUNKS" ]; then
    echo "Error: chunk must be 1-${MAX_CHUNKS} for L=${SIZE}"
    exit 1
fi

OFFSET=$(( 32 + (CHUNK - 1) * SEEDS_PER_CHUNK ))
LAST_SEED=$(( OFFSET + SEEDS_PER_CHUNK - 1 ))

LOGFILE="logs/tier2_L${SIZE}_chunk${CHUNK}.log"

echo "====== Tier 2.1: L=${SIZE}, chunk ${CHUNK}/${MAX_CHUNKS} ======"
echo "  Seeds: ${OFFSET}-${LAST_SEED} (${SEEDS_PER_CHUNK} seeds × 16 μ = $(( SEEDS_PER_CHUNK * 16 )) jobs)"
echo "  Frames: ${FRAMES}, Workers: ${WORKERS}"
echo "  Log: ${LOGFILE}"
echo "  Started: $(date)"
echo ""

venv/bin/python3 analysis/sweep.py \
    --binary torus_dyn \
    --reinforce-mins 4 \
    --seeds "$SEEDS_PER_CHUNK" \
    --seed-offset "$OFFSET" \
    --workers "$WORKERS" \
    --max-degree 8 \
    --save-seeds \
    --frames "$FRAMES" \
    --stats-interval 10000 \
    --width "$SIZE" \
    --height "$SIZE" \
    --mutation-probs $MU_GRID \
    2> "$LOGFILE"

echo ""
echo "====== Done: $(date) ======"
