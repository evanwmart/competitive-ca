# torus

A three-color competitive exclusion cellular automaton, studied in two configurations:

- **`torus`** — fixed 4-regular toroidal lattice
- **`torus_dyn`** — same rules, edges co-evolve with state (adaptive network)

The two configurations produce qualitatively different phase transitions, establishing
that topological freedom changes the fundamental nature of ordering in this class of
system.

---

## Key results

| system | transition type | critical mutation prob | order parameter |
|--------|-----------------|------------------------|-----------------|
| fixed lattice | second-order (continuous) | ≈ 0.20 | boundary density |
| adaptive network | **first-order (discontinuous)** | ≈ 0.35 | boundary density + degree |

On the fixed lattice, the transition is in the voter model / 2D percolation universality
class (domain size exponent τ → 2.0). On the adaptive network, topological co-evolution
produces spontaneous homophily — compatible nodes preferentially connect, incompatible
nodes disconnect — driving a first-order transition with simultaneous discontinuities in
state order and network topology.

Full results and analysis are in `research-log/`.

---

## Build

Requires a C11 compiler. No external dependencies.

```bash
make all          # builds both torus and torus_dyn
make torus        # fixed lattice only
make torus_dyn    # adaptive network only
```

**Live view** (optional): the demo commands below pipe raw video to
[ffplay](https://ffmpeg.org/ffplay.html) (part of ffmpeg). Not needed for
headless runs or analysis.

---

## Quick demo

**Fixed lattice — live view (requires ffplay):**
```bash
./torus | ffplay -f rawvideo -pixel_format rgb24 -video_size 128x128 \
    -framerate 60 -vf scale=512:512:flags=neighbor -i pipe:0
```

**Adaptive network — ordered phase (mutation prob 0.20):**
```bash
./torus_dyn --mutation-rate 5 | ffplay -f rawvideo -pixel_format rgb24 \
    -video_size 128x128 -framerate 60 -vf scale=512:512:flags=neighbor -i pipe:0
```

**Adaptive network — disordered phase (mutation prob 0.40):**
```bash
./torus_dyn --mutation-prob 0.40 | ffplay -f rawvideo -pixel_format rgb24 \
    -video_size 128x128 -framerate 60 -vf scale=512:512:flags=neighbor -i pipe:0
```

---

## The model

Each node holds an RGB color. The dominant channel (R, G, or B) defines the node's
"type". At each step a random node is selected and competes with each of its neighbors:

- **Compatible pair** (same dominant channel): both nodes reinforce — their shared
  dominant channel increases. The edge is labeled with that channel.
- **Incompatible pair**: the node with the smaller dominance margin loses. Its dominant
  channel decreases and the winner's channel increases in the loser. The edge is labeled
  null (boundary).
- **Self-reinforcement**: if `reinforce_min` or more of a node's edges agree on the same
  channel, the node boosts that channel.
- **Mutation**: each selected node resets to a random color with probability
  `1/mutation_rate` (or `--mutation-prob p`).

In `torus_dyn`, after each competition:
- Aligned pair → attempt to form a new global random edge (prob `1/topo_rate`)
- Conflicting pair → attempt to sever the edge (prob `1/topo_rate`, min degree 2)

Edge formation is capped at `max_degree` (default unlimited; use `--max-degree 8` for
long sweeps to prevent runaway hub formation in the ordered phase).

---

## `torus` usage

```
./torus [options] [width] [height] [steps_per_frame] [seed] [reinforce_min]

Options:
  --headless             skip video output
  --stats-interval N     emit CSV stats to stderr every N frames
  --mutation-rate N      1-in-N mutation chance (default 2000)
  --frames N             stop after N frames
```

Writes raw RGB24 frames to stdout. Must be piped.

---

## `torus_dyn` usage

```
./torus_dyn [options] [width] [height] [steps_per_frame] [seed] [reinforce_min]

Options:
  --headless             skip video output
  --stats-interval N     emit CSV stats to stderr every N frames
  --mutation-rate N      1-in-N mutation chance (default 2000)
  --mutation-prob P      float mutation probability, overrides --mutation-rate
  --topo-rate N          1-in-N topology change per competition (default n_nodes)
                         0 = frozen topology (fixed-graph mode)
  --max-degree N         cap degree growth via edge formation (0=unlimited)
  --frames N             stop after N frames
```

Stats CSV includes degree columns: `mean_degree, degree_variance, max_degree`.

---

## Analysis

Requires Python 3.10+ with `numpy`, `matplotlib`, `scipy` (see `requirements.txt`).

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
```

**Phase diagram sweep:**
```bash
# Fixed lattice
venv/bin/python3 analysis/sweep.py --frames 5000 --seeds 3

# Adaptive network (fractional mutation probe)
venv/bin/python3 analysis/sweep.py --binary torus_dyn \
    --mutation-probs 0.33 0.35 0.37 0.40 0.50 \
    --reinforce-mins 4 --seeds 8 --frames 30000 --max-degree 8
```

**Domain size distribution (τ measurement):**
```bash
venv/bin/python3 analysis/histogram.py --frames 5000 --seeds 4
```

See `REPLICATION.md` for the exact commands that reproduce all results.

---

## Directory structure

```
torus/
├── sim.c / sim.h          fixed-lattice CA core + RNG + color helpers
├── stats.c / stats.h      fixed-lattice statistics (boundary density, domains, histogram)
├── main.c                 fixed-lattice CLI
├── dgraph.c / dgraph.h    adaptive-network CA (adjacency lists, topology co-evolution)
├── main_dyn.c             adaptive-network CLI
├── Makefile
├── analysis/
│   ├── sweep.py           parameter sweep → phase diagram plots
│   ├── histogram.py       domain size distribution → τ measurement
│   └── analyse.py         single-run stats plot
├── research-log/          dated entries documenting all measurements and findings
├── results/               sweep CSVs and PNGs (timestamped, not tracked in git)
├── notes/                 pre-experiment hypotheses, implementation notes, and future directions
└── REPLICATION.md         exact commands to reproduce all key results
```

---

## License

MIT © 2026 Evan William Martin. See [LICENSE](LICENSE).
