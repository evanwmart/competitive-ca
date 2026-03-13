#!/usr/bin/env bash
# smoke_test.sh — replication check (~20 minutes)
# Verifies that key phase boundaries and the first-order coexistence signal
# are present. Run from the project root: bash analysis/smoke_test.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PASS=0; FAIL=0

check() {
    local label="$1" val="$2" op="$3" thresh="$4"
    if awk "BEGIN { exit !(${val} ${op} ${thresh}) }"; then
        printf "[PASS] %s: %s %s %s\n" "$label" "$val" "$op" "$thresh"
        PASS=$((PASS+1))
    else
        printf "[FAIL] %s: %s NOT %s %s\n" "$label" "$val" "$op" "$thresh"
        FAIL=$((FAIL+1))
    fi
}

# Run binary, return mean bd over the SECOND HALF of stats output (discard transient).
bd_mean() {
    "$@" 2>&1 \
        | grep -v '^seed=' | grep -v '^frame,' \
        | awk -F',' 'NF>=8 && $1+0==$1 { print $3 }' \
        | awk '{ vals[NR]=$1; n++ }
               END { start=int(n/2)+1; s=0; c=0;
                     for(i=start;i<=n;i++){s+=vals[i];c++}
                     if(c>0) printf "%.6f", s/c }'
}

echo "=== torus smoke test ==="
echo "(~20 minutes total)"
echo ""

echo "--- Fixed lattice (10k frames each) ---"

echo -n "  fixed lattice ordered (mr=9, p=0.111)... "
BD=$(bd_mean ./torus --headless --stats-interval 100 --mutation-rate 9 \
        --frames 10000 128 128 0 42 4)
echo "$BD"
check "fixed lattice ordered (mr=9)" "$BD" "<" "0.20"

echo -n "  fixed lattice disordered (mr=2, p=0.5)... "
BD=$(bd_mean ./torus --headless --stats-interval 100 --mutation-rate 2 \
        --frames 10000 128 128 0 42 4)
echo "$BD"
check "fixed lattice disordered (mr=2)" "$BD" ">" "0.50"

echo ""
echo "--- Adaptive network (15k frames each) ---"

echo -n "  dynamic ordered (p=0.25, mr=4)... "
BD=$(bd_mean ./torus_dyn --headless --stats-interval 100 --mutation-rate 4 \
        --max-degree 8 --frames 15000 128 128 0 42 4)
echo "$BD"
check "dynamic graph ordered (mr=4, p=0.25)" "$BD" "<" "0.15"

echo -n "  dynamic disordered (p=0.40)... "
BD=$(bd_mean ./torus_dyn --headless --stats-interval 100 --mutation-prob 0.40 \
        --max-degree 8 --frames 15000 128 128 0 42 4)
echo "$BD"
check "dynamic graph disordered (p=0.40)" "$BD" ">" "0.45"

echo ""
echo "--- First-order coexistence check (p=0.35, 6 seeds × 30k frames) ---"
echo "    (seeds should split between ordered bd≈0.15 and disordered bd≈0.50)"

BD_VALS=()
for seed in 0 1 2 3 4 5; do
    printf "  seed=%d... " "$seed"
    b=$(bd_mean ./torus_dyn --headless --stats-interval 100 \
            --mutation-prob 0.35 --max-degree 8 \
            --frames 30000 128 128 0 "$seed" 4)
    echo "$b"
    BD_VALS+=("$b")
done

# Inter-seed variance: should be large due to bimodality.
ISVAR=$(printf '%s\n' "${BD_VALS[@]}" | awk '
    { vals[NR]=$1; s+=$1; n++ }
    END { mean=s/n; sq=0
          for(i=1;i<=n;i++) sq+=(vals[i]-mean)^2
          printf "%.6f", sq/n }')

echo "  inter-seed variance: $ISVAR"
check "first-order coexistence: inter-seed var(bd) at p=0.35" "$ISVAR" ">" "0.010"

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "All ${PASS} checks passed."
else
    echo "${PASS} passed, ${FAIL} FAILED."
    exit 1
fi
