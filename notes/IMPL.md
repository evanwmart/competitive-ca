# Implementation Tracking

Steps to get to NEXT_STEPS.md Stage 1 (instrumentation), with modular structure for later stages.

---

## Step A: Makefile
- [x] Create Makefile targeting `torus` binary from `sim.c stats.c main.c`

## Step B: Split main.c → sim.c/sim.h + main.c
- [x] Extract: Torus struct, init/free, color helpers, Rules struct, process_node → `sim.h` / `sim.c`
- [x] `main.c` retains: CLI parsing, run loop, write_frame, main()
- [x] Verify build + behavior identical to original

## Step C: Add stats.c / stats.h
- [x] Define `Stats` struct (boundary_density, domain_count, mean_domain_size, type_fractions[3])
- [x] `compute_stats(const Torus *t, Stats *out, uint8_t *scratch)` — scratch is pre-allocated visited[]
- [x] Flood fill (BFS) over nodes by dominant type → domain count, sizes
- [x] Boundary density: count null edges / total edges
- [x] Type fractions: count dominant type per node
- [x] Domain size histogram (log2-binned, HIST_BINS=24)

## Step D: Add headless mode + stats output to main.c
- [x] New CLI args: `--headless`, `--stats-interval N`
- [x] Pre-allocate scratch buffer (size n) alongside video buf
- [x] Each frame: conditionally write_frame, conditionally compute_stats + emit CSV to stderr
- [x] CSV format: `frame,step,boundary_density,domain_count,mean_domain_size,frac_r,frac_g,frac_b`
- [x] Headless mode skips isatty() check on stdout

## Step E: Python analysis stub
- [x] `analysis/analyse.py` — reads CSV from stdin or file
- [x] 6-panel plot: boundary density + rolling mean, domain count, mean domain size,
      type fractions, boundary density variance
- [x] Equilibrium detection via rolling mean relative-std threshold
- [x] `summarise()` prints post-equilibrium stats to stdout
- [x] venv at `venv/` with numpy + matplotlib

Note: equilibrium detection catches end of sharp transient; mean domain size
continues slow logarithmic coarsening well beyond that — consistent with
HYPOTHESES.md voter-model prediction. May need longer runs / tighter threshold
for Step 2 (true equilibrium mapping).

---

## Future (NEXT_STEPS.md Stage 2+)
- Stage 2: equilibrium detection, transient length vs grid size
- Stage 3: parameter sweep (reinforce_min continuous, mutation rate sweep), phase diagram heatmap
- Stage 4: critical exponents, finite-size scaling, Binder cumulant
- Stage 5: compare to voter model / Potts exponents
- Stage 6: generalize to N types (requires sim.h node representation change)
