#include "dgraph.h"

#include <stdlib.h>
#include <string.h>

// ── adjacency list helpers ─────────────────────────────────────────────────────

static bool edge_push(DNode *node, size_t neighbor, uint8_t rel) {
    if (node->n_edges >= node->cap_edges) {
        size_t new_cap = node->cap_edges ? node->cap_edges * 2 : 4;
        DEdge *e = realloc(node->edges, new_cap * sizeof(*e));
        if (!e) return false;
        node->edges    = e;
        node->cap_edges = new_cap;
    }
    node->edges[node->n_edges++] = (DEdge){ .neighbor = neighbor, .rel = rel };
    return true;
}

// O(degree) find; returns pointer into node->edges or NULL.
static DEdge *edge_find(DNode *node, size_t neighbor) {
    for (size_t i = 0; i < node->n_edges; i++)
        if (node->edges[i].neighbor == neighbor) return &node->edges[i];
    return NULL;
}

// O(degree) swap-and-shrink removal; no-op if not found.
static void edge_remove(DNode *node, size_t neighbor) {
    for (size_t i = 0; i < node->n_edges; i++) {
        if (node->edges[i].neighbor == neighbor) {
            node->edges[i] = node->edges[node->n_edges - 1];
            node->n_edges--;
            return;
        }
    }
}

// Add undirected edge between a and b.
static bool edge_add(DGraph *g, size_t a, size_t b, uint8_t rel) {
    return edge_push(&g->nodes[a], b, rel)
        && edge_push(&g->nodes[b], a, rel);
}

// ── lifecycle ──────────────────────────────────────────────────────────────────

bool dgraph_init(DGraph *g, size_t w, size_t h) {
    size_t n = w * h;
    g->n     = n;
    g->nodes = calloc(n, sizeof(*g->nodes));
    if (!g->nodes) return false;

    for (size_t i = 0; i < n; i++)
        g->nodes[i].color = make_color(rng_u8(), rng_u8(), rng_u8());

    // 4-regular torus: each node connects right and down (wrapping).
    for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
            size_t idx   = y * w + x;
            size_t right = y * w + (x + 1) % w;
            size_t down  = ((y + 1) % h) * w + x;
            if (!edge_add(g, idx, right, (uint8_t)REL_NULL)) goto oom;
            if (!edge_add(g, idx, down,  (uint8_t)REL_NULL)) goto oom;
        }
    }
    return true;

oom:
    dgraph_free(g);
    return false;
}

void dgraph_free(DGraph *g) {
    if (!g->nodes) return;
    for (size_t i = 0; i < g->n; i++) free(g->nodes[i].edges);
    free(g->nodes);
    g->nodes = NULL;
    g->n = 0;
}

// ── competition ────────────────────────────────────────────────────────────────

// Compete nodes ai and bi on their shared edge.
// Returns true if they aligned (same dominant channel).
static bool compete(DGraph *g, size_t ai, size_t bi, const Rules *rules) {
    uint32_t ca = g->nodes[ai].color;
    uint32_t cb = g->nodes[bi].color;
    Relationship da = node_dominant(ca);
    Relationship db = node_dominant(cb);

    if (da == db) {
        // Aligned: reinforce both nodes.
        set_channel(&g->nodes[ai].color, da,
                    clamp_u8((int)get_channel(ca, da) + rules->neighbor_step));
        set_channel(&g->nodes[bi].color, da,
                    clamp_u8((int)get_channel(cb, da) + rules->neighbor_step));
        // Mark edge in both lists.
        DEdge *eab = edge_find(&g->nodes[ai], bi);
        DEdge *eba = edge_find(&g->nodes[bi], ai);
        if (eab) eab->rel = (uint8_t)da;
        if (eba) eba->rel = (uint8_t)da;
        return true;
    } else {
        // Conflicting: weaker node converts toward the stronger.
        uint8_t va = dominance_margin(ca, da);
        uint8_t vb = dominance_margin(cb, db);
        if (va <= vb) {
            set_channel(&g->nodes[ai].color, da,
                        clamp_u8((int)va - rules->neighbor_step));
            set_channel(&g->nodes[ai].color, db,
                        clamp_u8((int)get_channel(ca, db) + rules->neighbor_step));
        } else {
            set_channel(&g->nodes[bi].color, db,
                        clamp_u8((int)vb - rules->neighbor_step));
            set_channel(&g->nodes[bi].color, da,
                        clamp_u8((int)get_channel(cb, da) + rules->neighbor_step));
        }
        DEdge *eab = edge_find(&g->nodes[ai], bi);
        DEdge *eba = edge_find(&g->nodes[bi], ai);
        if (eab) eab->rel = (uint8_t)REL_NULL;
        if (eba) eba->rel = (uint8_t)REL_NULL;
        return false;
    }
}

// ── topology co-evolution ─────────────────────────────────────────────────────

// Try to sever edge (ai, bi) with probability 1/topo_rate.
// Minimum degree 2 preserved on both endpoints.
static void maybe_sever(DGraph *g, size_t ai, size_t bi, uint32_t topo_rate) {
    if (g->nodes[ai].n_edges <= 2 || g->nodes[bi].n_edges <= 2) return;
    if (rng_range(topo_rate) != 0) return;
    edge_remove(&g->nodes[ai], bi);
    edge_remove(&g->nodes[bi], ai);
}

// Try to form a new global random edge from ai with probability 1/topo_rate.
// Tries up to 8 random candidates to find a non-neighbor.
// Honours max_degree cap on both endpoints (0 = unlimited).
static void maybe_form_global(DGraph *g, size_t ai, size_t bi,
                              uint32_t topo_rate, uint32_t max_degree) {
    if (max_degree > 0 && g->nodes[ai].n_edges >= max_degree) return;
    if (rng_range(topo_rate) != 0) return;
    size_t n = g->n;
    for (int tries = 0; tries < 8; tries++) {
        size_t k = rng_range(n);
        if (k == ai || k == bi) continue;
        if (edge_find(&g->nodes[ai], k)) continue;
        if (max_degree > 0 && g->nodes[k].n_edges >= max_degree) continue;
        edge_add(g, ai, k, (uint8_t)REL_NULL);
        return;
    }
}

// Try to form a new LOCAL edge from ai: pick a random neighbour-of-neighbour.
// Restricts formation to distance-2 in the current graph.
static void maybe_form_local(DGraph *g, size_t ai, size_t bi,
                             uint32_t topo_rate, uint32_t max_degree) {
    if (max_degree > 0 && g->nodes[ai].n_edges >= max_degree) return;
    if (rng_range(topo_rate) != 0) return;
    DNode *node_a = &g->nodes[ai];
    for (int tries = 0; tries < 8; tries++) {
        // Pick a random neighbour of ai
        if (node_a->n_edges == 0) return;
        size_t mid = node_a->edges[rng_range(node_a->n_edges)].neighbor;
        // Pick a random neighbour of that neighbour
        DNode *node_m = &g->nodes[mid];
        if (node_m->n_edges == 0) continue;
        size_t k = node_m->edges[rng_range(node_m->n_edges)].neighbor;
        if (k == ai || k == bi) continue;
        if (edge_find(node_a, k)) continue;
        if (max_degree > 0 && g->nodes[k].n_edges >= max_degree) continue;
        edge_add(g, ai, k, (uint8_t)REL_NULL);
        return;
    }
}

// Dispatch: form edge globally or locally depending on rules.
static void maybe_form(DGraph *g, size_t ai, size_t bi,
                       uint32_t topo_rate, const Rules *rules) {
    if (rules->local_formation)
        maybe_form_local(g, ai, bi, topo_rate, rules->max_degree);
    else
        maybe_form_global(g, ai, bi, topo_rate, rules->max_degree);
}

// ── reinforce ─────────────────────────────────────────────────────────────────

// Self-reinforce node idx based on consensus among its current neighbors.
static void reinforce(DGraph *g, size_t idx, const Rules *rules) {
    DNode *node = &g->nodes[idx];
    int counts[4] = {0};
    for (size_t e = 0; e < node->n_edges; e++)
        counts[node->edges[e].rel & 0x3]++;

    Relationship best = REL_NULL;
    int best_count = 0;
    for (int r = 1; r <= 3; r++) {
        if (counts[r] > best_count) { best_count = counts[r]; best = (Relationship)r; }
    }

    if (best_count >= (int)rules->reinforce_min) {
        uint8_t ch = get_channel(node->color, best);
        set_channel(&node->color, best, clamp_u8((int)ch + rules->neighbor_step));
    }
}

// ── process node ──────────────────────────────────────────────────────────────

void dgraph_process_node(DGraph *g, size_t idx, const Rules *rules,
                         uint32_t topo_rate) {
    DNode *node = &g->nodes[idx];
    if (node->n_edges == 0) return;

    // Compete on all current edges (snapshot n_edges to handle list changes).
    // Process a copy of neighbor indices — the list may mutate during severance.
    size_t ne = node->n_edges;
    // Stack buffer for small degrees; heap for larger (hub case).
    size_t  stack_buf[16];
    size_t *neighbors = (ne <= 16) ? stack_buf : malloc(ne * sizeof(size_t));
    if (!neighbors) return;
    for (size_t e = 0; e < ne; e++) neighbors[e] = node->edges[e].neighbor;

    for (size_t e = 0; e < ne; e++) {
        size_t nb = neighbors[e];
        // Neighbor may have been severed by a prior iteration; skip if gone.
        if (!edge_find(node, nb)) continue;

        bool aligned = compete(g, idx, nb, rules);

        if (topo_rate > 0) {
            if (aligned)
                maybe_form(g, idx, nb, topo_rate, rules);
            else
                maybe_sever(g, idx, nb, topo_rate);
        }
    }

    if (neighbors != stack_buf) free(neighbors);

    reinforce(g, idx, rules);

    if (mutation_fires(rules))
        g->nodes[idx].color = make_color(rng_u8(), rng_u8(), rng_u8());
}

// ── stats ──────────────────────────────────────────────────────────────────────

// BFS flood fill over non-null edges to find domain size.
// Returns domain size.
static size_t bfs_fill(const DGraph *g, size_t start,
                       Relationship type, uint8_t *visited,
                       size_t *queue, size_t queue_cap) {
    size_t head = 0, tail = 0, count = 0;
    visited[start] = 1;
    queue[tail++] = start;
    count++;

    while (head < tail) {
        size_t cur = queue[head++];
        const DNode *node = &g->nodes[cur];
        for (size_t e = 0; e < node->n_edges; e++) {
            size_t nb = node->edges[e].neighbor;
            if (!visited[nb] &&
                node->edges[e].rel != REL_NULL &&
                node_dominant(g->nodes[nb].color) == type) {
                visited[nb] = 1;
                if (tail < queue_cap) queue[tail++] = nb;
                count++;
            }
        }
    }
    return count;
}

void dgraph_compute_stats(const DGraph *g, DStats *out, uint8_t *scratch) {
    size_t n = g->n;
    memset(out, 0, sizeof(*out));
    memset(scratch, 0, n);

    size_t *queue = malloc(n * sizeof(*queue));
    if (!queue) return;

    // ── boundary density ──────────────────────────────────────────────────────
    size_t total_edges = 0, null_edges = 0;
    for (size_t i = 0; i < n; i++) {
        const DNode *node = &g->nodes[i];
        for (size_t e = 0; e < node->n_edges; e++) {
            if (node->edges[e].neighbor > i) {   // count each undirected edge once
                total_edges++;
                if ((node->edges[e].rel & 0x3) == REL_NULL) null_edges++;
            }
        }
    }
    out->boundary_density = total_edges > 0
        ? (double)null_edges / (double)total_edges : 0.0;

    // ── type fractions ────────────────────────────────────────────────────────
    size_t type_counts[3] = {0};
    for (size_t i = 0; i < n; i++) {
        Relationship d = node_dominant(g->nodes[i].color);
        if (d >= REL_RED && d <= REL_BLUE) type_counts[(int)d - 1]++;
    }
    for (int c = 0; c < 3; c++)
        out->type_fractions[c] = (double)type_counts[c] / (double)n;

    // ── domain flood fill ─────────────────────────────────────────────────────
    size_t total_size = 0;
    for (size_t i = 0; i < n; i++) {
        if (scratch[i]) continue;
        Relationship type = node_dominant(g->nodes[i].color);
        size_t sz = bfs_fill(g, i, type, scratch, queue, n);
        out->domain_count++;
        total_size += sz;
        size_t bin = 0, s = sz;
        while (s > 1 && bin < DGRAPH_HIST_BINS - 1) { s >>= 1; bin++; }
        out->size_histogram[bin]++;
    }
    out->mean_domain_size = out->domain_count > 0
        ? (double)total_size / (double)out->domain_count : 0.0;

    // ── degree stats ──────────────────────────────────────────────────────────
    double deg_sum = 0, deg_sq = 0;
    size_t max_deg = 0;
    for (size_t i = 0; i < n; i++) {
        size_t d = g->nodes[i].n_edges;
        deg_sum += d;
        deg_sq  += (double)d * d;
        if (d > max_deg) max_deg = d;
    }
    out->mean_degree    = deg_sum / (double)n;
    out->degree_variance = deg_sq / (double)n - out->mean_degree * out->mean_degree;
    out->max_degree     = max_deg;

    free(queue);
}
