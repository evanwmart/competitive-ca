# competitive-ca

A three-color competitive cellular automaton, studied in two configurations:

- **`torus`** — fixed 4-regular toroidal lattice
- **`torus_dyn`** — same local rules, edges coevolve with node states (adaptive network)

At the tested system size ($L = 128$), the two configurations produce qualitatively
different finite-size behavior: a smooth crossover on the fixed lattice, a sharp
coexistence-like jump on the adaptive network. The accompanying manuscript
(in `paper/`) presents the finite-size Binder analysis, mechanism controls, and a
degree-headroom sweep consistent with degree headroom as a contributing topological
ingredient.

---

## Paper

**Adaptive rewiring produces sharp finite-size bimodality in a conviction-weighted competitive network.**
Evan W. Martin, *under review at Physica A* (2026).

- Manuscript: [`paper/main.pdf`](paper/main.pdf)
- Supplementary material: [`paper/supplement.pdf`](paper/supplement.pdf)
- LaTeX sources: [`paper/main.tex`](paper/main.tex), [`paper/supplement.tex`](paper/supplement.tex)
- Backing data for figures: [`paper/cache/`](paper/cache/) (per-seed CSVs)

### Headline findings

| Aspect | Result |
|---|---|
| Fixed-lattice baseline (L = 128) | Smooth crossover-like loss of order |
| Adaptive-network finite-size transition | Sharp coexistence-like jump (ρ_b: 0.17 → 0.48 across μ ∈ [0.35, 0.355]) |
| Finite-size Binder analysis (L = 64, 128, 256, 384) | U_min moves from 0.26 to 0.52 between L = 128 and L = 256 (CIs non-overlapping), a substantial fraction of the way to the single-peak value 2/3; bimodal window narrows; disfavors conventional first-order scaling over the tested sizes, with a weakly first-order alternative beyond L = 384 not excluded |
| Mechanism (kmax sweep) | Consistent with a degree-headroom requirement; partly confounded by reinforcement-firing and formation-rate effects |

---

## Build

Requires a C11 compiler. No external dependencies.

```bash
make all          # builds both torus and torus_dyn
make torus        # fixed lattice only
make torus_dyn    # adaptive network only
```

---

## Quick demo

The simulators stream raw RGB24 frames to stdout for live viewing with
[ffplay](https://ffmpeg.org/ffplay.html). Use `--headless` for analysis runs.

**Fixed lattice — live view:**
```bash
./torus | ffplay -f rawvideo -pixel_format rgb24 -video_size 128x128 \
    -framerate 60 -vf scale=512:512:flags=neighbor -i pipe:0
```

**Adaptive network, ordered phase (μ = 0.20):**
```bash
./torus_dyn --mutation-prob 0.20 | ffplay -f rawvideo -pixel_format rgb24 \
    -video_size 128x128 -framerate 60 -vf scale=512:512:flags=neighbor -i pipe:0
```

**Adaptive network, disordered phase (μ = 0.40):**
```bash
./torus_dyn --mutation-prob 0.40 | ffplay -f rawvideo -pixel_format rgb24 \
    -video_size 128x128 -framerate 60 -vf scale=512:512:flags=neighbor -i pipe:0
```

---

## Model

Each node holds a three-channel RGB color. The dominant channel defines the node's
type. At each step a random node is selected and competes with each of its neighbors:

- **Compatible pair** (same dominant type): both nodes reinforce — shared dominant
  channel increases.
- **Incompatible pair**: the node with the smaller dominance margin loses; its
  dominant channel decreases and the winner's channel increases in the loser.
- **Self-reinforcement**: if at least `reinforce_min` of a node's edges agree on the
  same channel, the node boosts that channel.
- **Mutation**: each selected node resets to a uniformly random color with
  probability `mutation-prob`.

In `torus_dyn`, after each pairwise interaction between focal node `i` and neighbor `j`:
- Aligned pair → with prob `1/topo_rate`, `i` attempts to form a new edge to a
  uniformly random non-neighbor (focal node and target both gated by `max_degree`).
  The target is *not* required to be of matching type; homophilic clustering arises
  from compatibility-triggered formation plus selective severance.
- Conflicting pair → attempt to sever the edge (prob `1/topo_rate`, min degree 2).

---

## CLI

### `./torus` (fixed lattice)
```
./torus [options] [width] [height] [steps_per_frame] [seed] [reinforce_min]
  --headless             skip video output
  --stats-interval N     emit CSV stats to stderr every N frames
  --mutation-rate N      1-in-N mutation chance (default 2000)
  --frames N             stop after N frames
```

### `./torus_dyn` (adaptive network)
```
./torus_dyn [options] [width] [height] [steps_per_frame] [seed] [reinforce_min]
  --headless             skip video output
  --stats-interval N     emit CSV stats to stderr every N frames
  --mutation-rate N      1-in-N mutation chance (default 2000)
  --mutation-prob P      float mutation probability (overrides --mutation-rate)
  --topo-rate N          1-in-N topology change per competition (default n_nodes;
                         0 = frozen topology)
  --max-degree N         cap degree via edge formation (0 = unlimited)
  --local-formation      restrict edge formation to distance-2 neighbors
                         (control experiment)
  --frames N             stop after N frames
```

`torus_dyn` stats CSV includes `mean_degree, degree_variance, max_degree` columns
in addition to the fixed-lattice fields.

---

## Reproducing the paper's analyses

Requires Python 3.10+ with `numpy`, `matplotlib`, `scipy` (see [`requirements.txt`](requirements.txt)).

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
```

For an exact replication of the environment that produced the paper's PDFs, use
the pinned lock file:

```bash
venv/bin/pip install -r requirements-frozen.txt
```

The `analysis/` directory contains the scripts that produce every figure in the paper:

| Script | Produces |
|---|---|
| `analysis/sweep.py` | Parameter sweeps; the workhorse runner |
| `analysis/figures.py` | Most main-text figures from cached per-seed data |
| `analysis/binder.py`, `binder_bootstrap.py` | Binder cumulant + bootstrap CIs (Fig. 6) |
| `analysis/data_collapse_binder.py` | Two-parameter Binder collapse (Supplement S1) |
| `analysis/histograms_coex.py`, `histograms_L384.py` | ρ_b distribution histograms (Fig. 4, Supplement S2) |
| `analysis/convergence_plot.py` | Equilibration diagnostics (Supplement S3) |
| `analysis/merge_seeds.py` | Consolidates per-seed CSVs across chunked runs |

See [`scripts/run_tier2.sh`](scripts/run_tier2.sh),
[`scripts/run_revision.sh`](scripts/run_revision.sh), and the other runners
in [`scripts/`](scripts/) for the exact compute scripts used to generate the
data behind each figure. All scripts cd to the repo root before running, so
invoke them as `bash scripts/<name>.sh` from anywhere.

---

## Compute

All runs reported in the paper were produced on a single workstation:
AMD Ryzen 9 9950X (16 cores / 32 threads), 32 GB DDR5, 1.8 TB NVMe (btrfs),
openSUSE Tumbleweed. The simulator is single-threaded C; parallelism comes
from the Python sweep runner farming `(μ, seed)` jobs across processes via
`--workers N`. No GPU and no MPI are used. Worker counts in the run scripts
target 24–28 concurrent jobs; reduce `--workers` to fit smaller hardware.

| Experiment | Script | Workers | Wall time |
|---|---|---|---|
| Full replication of all main-text figures | `scripts/run_all_replication.sh` | 24 | ~12–16 h |
| L=128, L=256 fine Binder sweep + k_max=4 control | `scripts/run_experiments_2.sh` | 28 | ~6–8 h |
| L=384 fine Binder sweep (32 seeds, baseline) | `scripts/run_experiments_3.sh` | 8 | ~12 h (cap) |
| L=384 additional 32 seeds, chunked | `scripts/run_tier2.sh 384 N` | 24 | ~7.3 h per chunk × 32 |
| L=256 additional 32 seeds, chunked | `scripts/run_tier2.sh 256 N` | 24 | ~5 h per chunk × 8 |
| Local-formation control + L=128/256 Binder | `scripts/run_revision.sh` | 24 | ~8–10 h |

The principal compute investment is the L=384 Binder sweep with 64 seeds at
200 000 frames — roughly 250 core-hours. Smaller systems (L ≤ 128) and the
fixed-lattice phase diagram complete in under an hour. Disk usage is dominated
by `snapshots/` (raw per-seed node and edge CSVs) which can exceed 100 GB
during the full replication; pass `--no-snapshots` or `--save-seeds` to the
sweep runner to keep only aggregated CSVs.

---

## Directory structure

```
competitive-ca/
├── sim.c / sim.h           fixed-lattice CA core (RNG, color helpers)
├── stats.c / stats.h       fixed-lattice statistics
├── main.c                  fixed-lattice CLI
├── dgraph.c / dgraph.h     adaptive-network CA (adjacency lists, topology coevolution)
├── main_dyn.c              adaptive-network CLI
├── Makefile
├── analysis/               Python figure-generation and analysis scripts
├── scripts/                bash runners for the compute pipelines
├── paper/                  manuscript, supplement, figures, cache CSVs
├── notes/                  pre-experiment hypotheses, implementation notes
├── research-log/           dated entries documenting measurements and findings
├── results/                sweep CSVs and PNGs (gitignored, timestamped)
├── snapshots/              raw simulation snapshots (gitignored, large)
├── videos/                 demo MP4s (gitignored)
└── logs/                   runtime logs (gitignored)
```

---

## License

MIT © 2026 Evan W. Martin. See [LICENSE](LICENSE).
