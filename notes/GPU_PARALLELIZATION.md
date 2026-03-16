# GPU parallelization notes

## Current parallelism

The simulation is already embarrassingly parallel at the ensemble level —
independent seeds run simultaneously across CPU cores (`--workers N`). This
scales linearly with core count and is how all sweeps are run.

What is not yet parallelized is a single simulation instance. Each Monte Carlo
step selects a random node, modifies that node and its neighbours, and
potentially rewires edges. Concurrent updates on the same node create data
hazards, so steps run serially.

## Why GPU is non-trivial

The dynamic graph version has three GPU-hostile properties:

1. **Irregular memory access** — adjacency lists are dynamic arrays allocated
   per node. Pointer chasing through heap memory is slow on GPU.
2. **Global edge formation** — when a node forms a new edge, it picks a random
   target anywhere in the graph. This is a global, unpredictable memory write.
3. **Variable degree** — nodes have different numbers of edges, which maps
   poorly to the SIMT (single instruction, multiple threads) execution model.

The fixed-lattice version (`torus`) has none of these problems and is
straightforwardly GPU-compatible.

## Making the dynamic graph GPU-compatible

### 1. Dense edge storage

Replace the current dynamic adjacency list (`DEdge *edges, n_edges, cap_edges`)
with a fixed 2D array of shape `[n_nodes × max_degree]`. The degree cap is
already part of the model (max_degree=8 in all published runs). This eliminates
pointer chasing and gives coalesced GPU memory access.

### 2. Checkerboard / graph coloring updates

For the fixed lattice: color nodes like a checkerboard. All nodes of one color
share no edges (they only connect to the other color), so every node of that
color can update simultaneously on the GPU without data hazards. Two passes
(red then black) constitute one full sweep. Maps directly to GPU thread blocks.

For the dynamic graph: the graph coloring changes as edges rewire, making
static checkerboard inapplicable. Instead, use a **stochastic conflict
resolution** scheme: launch all N nodes as GPU threads simultaneously, use
atomic compare-and-swap for color updates, and discard or retry the small
fraction of steps where two threads write to the same node. At typical degrees
(4–8), the conflict rate is low (~degree/N) and the approximation is
well-established for this class of model.

### 3. Separate state and topology phases

Alternate between two passes per frame:

- **State pass** — all nodes update colors in parallel (atomic ops for
  conflicts). This is the competition + reinforcement logic.
- **Topology pass** — process edge rewiring events. Topology changes are
  infrequent (1/topo_rate per competition ≈ 4 events per frame at topo_rate=N),
  so this pass can be handled with a small queue of pending events processed
  serially or with fine-grained locking. It does not need to be fast.

The model semantics are preserved: competition probabilities and topo rates are
unchanged. The update scheme becomes weakly synchronous rather than strictly
asynchronous, which is a known acceptable deviation for nonequilibrium CA models
and does not affect universality class or phase boundary location.

## Estimated speedup

| version       | scheme                          | estimated speedup |
|---------------|---------------------------------|-------------------|
| fixed lattice | checkerboard, all parallel      | 100–500×          |
| dynamic graph | dense storage + atomic + phases | 20–100×           |

These are rough estimates based on comparable CA GPU implementations. Actual
speedup depends on GPU memory bandwidth, atomic contention rate, and occupancy.

## Impact on tractable system sizes

Current bottleneck: ordering time scales as O(L²) in frames, and time per frame
scales as O(L²), giving total time per seed ∝ L⁴.

With GPU parallelism the per-frame cost drops dramatically (closer to O(1) in
GPU threads, bounded by memory bandwidth). The L⁴ scaling becomes an L²
scaling in wall time. Tractable sizes:

| L    | CPU (4 cores) | GPU (est. 50× dynamic graph) |
|------|---------------|------------------------------|
| 128  | ~10 min       | ~12 sec                      |
| 256  | ~3.5 hrs      | ~4 min                       |
| 512  | ~2.5 days     | ~1.2 hrs                     |
| 1024 | months        | ~20 hrs                      |

L=512 becomes an afternoon run. The full finite-size scaling study (L=64, 128,
256, 512) that would definitively confirm first-order vs pseudo-first-order
becomes tractable on a single GPU workstation.

## Implementation priority

This is post-writeup work. The current CPU results at L=64 and L=128 are
sufficient for the core paper. GPU implementation would:

1. Enable the L=256 and L=512 size dependence study needed to fully confirm
   first-order behaviour
2. Open up exploration of larger vocabulary sizes (4, 6, 8 types) and other
   parameter regimes
3. Make real-time visualization at large L feasible

The fixed-lattice GPU version is straightforward and could be a useful
standalone contribution. The dynamic graph version is a more substantial
engineering project but follows from the same principles.
