#pragma once

#include "sim.h"   // Relationship, Node color helpers, RNG, Rules

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

// ── adjacency list types ───────────────────────────────────────────────────────

typedef struct {
    size_t  neighbor;   // index of the other endpoint
    uint8_t rel;        // Relationship value (2 bits used)
} DEdge;

typedef struct {
    uint32_t color;
    DEdge   *edges;     // dynamic array
    size_t   n_edges;   // current degree
    size_t   cap_edges; // allocated capacity
} DNode;

typedef struct {
    size_t  n;          // number of nodes
    DNode  *nodes;
} DGraph;

// ── lifecycle ──────────────────────────────────────────────────────────────────

// Initialize from a W×H torus topology (4-regular, all edges start REL_NULL).
bool dgraph_init(DGraph *g, size_t w, size_t h);
void dgraph_free(DGraph *g);

// ── simulation step ────────────────────────────────────────────────────────────

// Process one randomly-selected node: compete on all its edges, topology
// co-evolve, reinforce, mutate.
// topo_rate: 1-in-topo_rate chance of topology change per competition.
//   0 = topology frozen (fixed-graph mode).
//   g->n = one expected topology event per frame.
void dgraph_process_node(DGraph *g, size_t idx, const Rules *rules,
                         uint32_t topo_rate);

// ── stats ──────────────────────────────────────────────────────────────────────

#define DGRAPH_HIST_BINS 24

typedef struct {
    double boundary_density;          // null edges / total edges
    size_t domain_count;
    double mean_domain_size;
    double type_fractions[3];         // R, G, B
    size_t size_histogram[DGRAPH_HIST_BINS];
    // degree distribution
    double mean_degree;
    double degree_variance;
    size_t max_degree;
} DStats;

// scratch must be n bytes of caller-owned memory (zeroed internally).
void dgraph_compute_stats(const DGraph *g, DStats *out, uint8_t *scratch);
