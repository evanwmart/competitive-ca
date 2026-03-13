#include "stats.h"

#include <string.h>
#include <stdlib.h>

// ── flood fill (iterative BFS) ────────────────────────────────────────────────

// BFS queue stored as a flat array of node indices.
// We reuse the scratch buffer for "visited" flags (0 = unvisited).
// A separate queue is allocated once per compute_stats call and reused.

static size_t bfs_fill(const Torus *t, size_t start,
                       Relationship type, uint8_t *visited,
                       size_t *queue, size_t queue_cap) {
    size_t w = t->width, h = t->height;
    size_t head = 0, tail = 0;
    size_t count = 0;

    visited[start] = 1;
    queue[tail++] = start;
    count++;

    while (head < tail) {
        size_t idx = queue[head++];
        size_t x = idx % w, y = idx / w;

        // four neighbors (wrapping)
        size_t neighbors[4] = {
            node_idx(t, (x + 1) % w,         y        ),  // right
            node_idx(t, (x + w - 1) % w,     y        ),  // left
            node_idx(t,  x,                  (y + 1) % h),  // down
            node_idx(t,  x,          (y + h - 1) % h  ),  // up
        };

        for (int d = 0; d < 4; d++) {
            size_t nb = neighbors[d];
            if (!visited[nb] && node_dominant(t->nodes[nb].color) == type) {
                visited[nb] = 1;
                // grow queue (shouldn't exceed grid size, queue_cap == n)
                if (tail < queue_cap) queue[tail++] = nb;
                count++;
            }
        }
    }

    return count;
}

// ── compute_stats ─────────────────────────────────────────────────────────────

void compute_stats(const Torus *t, Stats *out, uint8_t *scratch) {
    size_t n = t->width * t->height;

    memset(out, 0, sizeof(*out));
    memset(scratch, 0, n);

    // allocate BFS queue (worst case: all nodes in one domain)
    size_t *queue = malloc(n * sizeof(*queue));
    if (!queue) return;  // can't compute; caller gets zeroed Stats

    // ── boundary density ──────────────────────────────────────────────────────
    size_t null_edges = 0;
    for (size_t i = 0; i < n; i++) {
        if ((t->hedges[i].value & 0x3) == REL_NULL) null_edges++;
        if ((t->vedges[i].value & 0x3) == REL_NULL) null_edges++;
    }
    out->boundary_density = (double)null_edges / (double)(2 * n);

    // ── type fractions ────────────────────────────────────────────────────────
    size_t type_counts[3] = {0};
    for (size_t i = 0; i < n; i++) {
        Relationship d = node_dominant(t->nodes[i].color);
        if (d >= REL_RED && d <= REL_BLUE) type_counts[(int)d - 1]++;
    }
    for (int c = 0; c < 3; c++)
        out->type_fractions[c] = (double)type_counts[c] / (double)n;

    // ── flood fill: domain count, sizes, histogram ────────────────────────────
    size_t total_size = 0;

    for (size_t i = 0; i < n; i++) {
        if (scratch[i]) continue;  // already visited

        Relationship type = node_dominant(t->nodes[i].color);
        size_t domain_size = bfs_fill(t, i, type, scratch, queue, n);

        out->domain_count++;
        total_size += domain_size;

        // log2-bin the domain size
        size_t bin = 0;
        size_t sz = domain_size;
        while (sz > 1 && bin < HIST_BINS - 1) { sz >>= 1; bin++; }
        out->size_histogram[bin]++;
    }

    out->mean_domain_size = out->domain_count > 0
        ? (double)total_size / (double)out->domain_count
        : 0.0;

    free(queue);
}
