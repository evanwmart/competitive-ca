#!/usr/bin/env bash
# run_overnight.sh — queue all outstanding measurements
#
# 1. Hysteresis at p=0.37 (random + ordered init)
# 2. Hysteresis at p=0.40 (random + ordered init)
# 3. Size dependence: fractional sweep at L=64 and L=256
#
# Estimated runtime: ~2.5 hours
# Run from project root: bash analysis/run_overnight.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="venv/bin/python3"
SWEEP="analysis/sweep.py"
BASE="--binary torus_dyn --reinforce-mins 4 --seeds 8 --frames 30000 --workers 4 --max-degree 8"

echo "============================================"
echo " Overnight run — $(date)"
echo "============================================"
echo ""

# ── 1. Hysteresis at p=0.37 ───────────────────────────────────────────────────

echo "--- Hysteresis p=0.37: random init ---"
$PY $SWEEP $BASE --mutation-probs 0.37

echo "--- Hysteresis p=0.37: ordered init ---"
$PY $SWEEP $BASE --mutation-probs 0.37 --ordered-init

# ── 2. Hysteresis at p=0.40 ───────────────────────────────────────────────────

echo "--- Hysteresis p=0.40: random init ---"
$PY $SWEEP $BASE --mutation-probs 0.40

echo "--- Hysteresis p=0.40: ordered init ---"
$PY $SWEEP $BASE --mutation-probs 0.40 --ordered-init

# ── 3. Size dependence: fractional sweep at L=64 and L=256 ───────────────────

PROBS="0.33 0.35 0.37 0.40 0.42 0.45 0.48 0.50"

echo "--- Size dependence: L=64 ---"
$PY $SWEEP $BASE --mutation-probs $PROBS --width 64 --height 64

echo "--- Size dependence: L=256 ---"
$PY $SWEEP $BASE --mutation-probs $PROBS --width 256 --height 256 --frames 10000

echo ""
echo "============================================"
echo " All runs complete — $(date)"
echo "============================================"
