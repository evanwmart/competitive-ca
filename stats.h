#pragma once

#include "sim.h"
#include <stddef.h>

// ── stats ─────────────────────────────────────────────────────────────────────

// Domain size histogram: fixed logarithmic bins, bin i covers sizes
// [2^i, 2^(i+1)).  Bin 0 covers size 1.  HIST_BINS covers up to 2^HIST_BINS.
#define HIST_BINS 24

typedef struct {
    double boundary_density;          // fraction of edges that are null
    size_t domain_count;              // number of connected same-type regions
    double mean_domain_size;          // average nodes per domain
    double type_fractions[3];         // fraction of nodes dominant R, G, B
    size_t size_histogram[HIST_BINS]; // log2-binned domain size counts
} Stats;

// Compute all observables for the current torus state.
// scratch must point to a caller-allocated uint8_t array of size t->width*t->height.
// It is zeroed internally; no other initialization required.
void compute_stats(const Torus *t, Stats *out, uint8_t *scratch);
