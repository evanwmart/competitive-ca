# Replication guide

This document gives the exact commands to reproduce every quantitative result in the
project. Run them in order. Each section states the expected output so you can verify
the reproduction is correct.

---

## Prerequisites

**C compiler:** any C11-compliant compiler (gcc ≥ 9, clang ≥ 11).

**Python:** 3.10 or newer.

```bash
# Build both binaries
make all

# Python environment
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Verify:
```bash
./torus --headless --frames 10 2>/dev/null && echo "torus OK"
./torus_dyn --headless --frames 10 2>/dev/null && echo "torus_dyn OK"
```

---

## Quick smoke test (~20 minutes)

Runs five pass/fail checks covering both phase boundaries and the first-order
coexistence signal.

```bash
bash analysis/smoke_test.sh
```

Runtime: ~20 minutes.

Expected output:
```
[PASS] fixed lattice ordered (mr=9): 0.0XX < 0.20
[PASS] fixed lattice disordered (mr=2): 0.XXX > 0.50
[PASS] dynamic graph ordered (mr=4, p=0.25): 0.0XX < 0.15
[PASS] dynamic graph disordered (p=0.40): 0.XXX > 0.45
[PASS] first-order coexistence: inter-seed var(bd) at p=0.35: 0.0XX > 0.010
All 5 checks passed.
```

Expect ~1 of 6 seeds to land in the disordered basin (bd≈0.50) and the rest in
the ordered basin (bd≈0.14–0.18). The exact seed that falls disordered varies
by run, but bimodality and high inter-seed variance are reproducible.

---

## Result 1 — Fixed lattice phase diagram

**Parameters:** 128×128, reinforce_min=4, seeds=8, frames=50000

```bash
venv/bin/python3 analysis/sweep.py \
    --mutation-rates 3 4 5 6 7 8 9 10 \
    --reinforce-mins 4 \
    --seeds 8 --frames 50000 --workers 4
```

Runtime: ~3 hours. Output saved to `results/sweep_fix_128x128_f50000_s8_*.{csv,png}`.

**Expected key values (±0.005):**

| mr | prob  | bd_mean | var peak? |
|----|-------|---------|-----------|
| 3  | 0.333 | ~0.48   | —         |
| 5  | 0.200 | ~0.43   | **yes**   |
| 7  | 0.143 | ~0.25   | —         |
| 9  | 0.111 | ~0.10   | —         |

Variance peaks at mr=5 (prob≈0.20). This identifies the critical point.

---

## Result 2 — Fixed lattice domain size exponent τ(L)

**Parameters:** reinforce_min=4, mr=5 (critical point), seeds=4

```bash
# L = 64
venv/bin/python3 analysis/histogram.py \
    --frames 5000 --seeds 4 --width 64 --height 64

# L = 128
venv/bin/python3 analysis/histogram.py \
    --frames 5000 --seeds 4 --width 128 --height 128

# L = 256
venv/bin/python3 analysis/histogram.py \
    --frames 10000 --seeds 4 --width 256 --height 256

# L = 512
venv/bin/python3 analysis/histogram.py \
    --frames 10000 --seeds 4 --width 512 --height 512
```

**Expected τ values (±0.05):**

| L   | τ     | R²    |
|-----|-------|-------|
| 64  | ~1.71 | >0.96 |
| 128 | ~1.89 | >0.96 |
| 256 | ~2.00 | >0.92 |
| 512 | ~1.92 | >0.94 |

τ converges toward 2.0 (2D percolation / voter model universality class: 187/91 ≈ 2.055).

---

## Result 3 — Dynamic graph phase diagram (mr=1–9)

**Parameters:** 128×128, reinforce_min=4, max_degree=8, seeds=8, frames=30000

```bash
venv/bin/python3 analysis/sweep.py \
    --binary torus_dyn \
    --mutation-rates 1 2 3 4 5 6 7 8 9 \
    --reinforce-mins 4 \
    --seeds 8 --frames 30000 --workers 4 \
    --max-degree 8
```

Runtime: ~90 minutes. `--max-degree 8` is required — without it, the ordered phase
produces runaway hub growth and timeouts.

**Expected key values:**

| mr | prob  | bd_mean | deg_mean | phase |
|----|-------|---------|----------|-------|
| 1  | 1.000 | ~0.658  | ~3.11    | disordered |
| 2  | 0.500 | ~0.581  | ~3.59    | disordered |
| 3  | 0.333 | ~0.104  | ~7.30    | ordered |
| 5  | 0.200 | ~0.051  | ~7.85    | ordered |
| 9  | 0.111 | ~0.027  | ~7.90    | ordered |

Note: at mr=3, seed-to-seed variance in bd is ~10× higher than at mr=2 or mr=4,
indicating proximity to the transition.

---

## Result 4 — First-order transition: fractional mutation probe

**Parameters:** 128×128, reinforce_min=4, max_degree=8, seeds=8, frames=30000

```bash
venv/bin/python3 analysis/sweep.py \
    --binary torus_dyn \
    --mutation-probs 0.33 0.35 0.37 0.40 0.42 0.45 0.48 0.50 \
    --reinforce-mins 4 \
    --seeds 8 --frames 30000 --workers 4 \
    --max-degree 8
```

Runtime: ~50 minutes.

**Expected output at prob=0.35 (the diagnostic):**

At prob=0.35, seeds should split between two states with no intermediate values:

- Ordered basin (≈7 of 8 seeds): bd ≈ 0.10–0.18, deg ≈ 5.8–6.7
- Disordered basin (≈1 of 8 seeds): bd ≈ 0.50, deg ≈ 3.9–4.0

The seed that lands in the disordered basin has variance ≈ 100× higher than at
prob=0.33 or 0.37. Exact basin assignment varies by seed due to stochastic dynamics,
but bimodality is reproducible.

At prob=0.33 and prob=0.37, all seeds should be tightly clustered (σ(bd) < 0.002).

---

## Notes on reproducibility

**Seeds and RNG:** The RNG uses xorshift64 with splitmix64 seed mixing. Each run is
fully deterministic given the same seed, binary, and parameters. Integer seeds 0–7
are used in all sweeps. Results shown are averages over these 8 seeds.

**Platform:** Results were obtained on a 4-core x86-64 Linux system. The `-march=native`
flag is used for performance; remove it if cross-compiling. Results are not sensitive
to floating-point rounding at this level of precision.

**Python version:** Tested with Python 3.12. Any 3.10+ should work.

**Parallelism:** `--workers 4` assumes 4 physical cores. Adjust to your hardware.
Using more workers than physical cores causes oversubscription and may trigger timeouts
on long runs.

**`max_degree` and the ordered phase:** Without `--max-degree 8`, the adaptive network
in the ordered phase exhibits runaway hub formation (mean degree grows without bound).
This makes long runs (30k+ frames) intractable at mutation rates below prob≈0.35.
The cap at 8 (2× initial degree) is part of the model definition for these measurements.
