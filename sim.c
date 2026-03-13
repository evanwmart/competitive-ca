#include "sim.h"

#include <stdlib.h>

uint64_t g_rng = 1;

// ── torus ─────────────────────────────────────────────────────────────────────

bool torus_init(Torus *t, size_t w, size_t h) {
    t->width  = w;
    t->height = h;
    size_t n  = w * h;
    t->nodes  = malloc(n * sizeof(*t->nodes));
    t->hedges = malloc(n * sizeof(*t->hedges));
    t->vedges = malloc(n * sizeof(*t->vedges));

    if (!t->nodes || !t->hedges || !t->vedges) {
        free(t->nodes); free(t->hedges); free(t->vedges);
        return false;
    }

    for (size_t i = 0; i < n; i++) {
        t->nodes[i].color  = make_color(rng_u8(), rng_u8(), rng_u8());
        t->hedges[i].value = REL_NULL;
        t->vedges[i].value = REL_NULL;
    }
    return true;
}

void torus_free(Torus *t) {
    free(t->nodes); free(t->hedges); free(t->vedges);
    t->nodes = NULL; t->hedges = NULL; t->vedges = NULL;
}

// ── simulation ────────────────────────────────────────────────────────────────

static void compete_edge(Torus *t, size_t ai, size_t bi,
                         size_t eidx, Edge *edges, const Rules *rules) {
    uint32_t ca = t->nodes[ai].color;
    uint32_t cb = t->nodes[bi].color;
    Relationship da = node_dominant(ca);
    Relationship db = node_dominant(cb);

    if (da == db) {
        set_channel(&t->nodes[ai].color, da,
                    clamp_u8((int)get_channel(ca, da) + rules->neighbor_step));
        set_channel(&t->nodes[bi].color, da,
                    clamp_u8((int)get_channel(cb, da) + rules->neighbor_step));
        edges[eidx].value = (uint8_t)da;
    } else {
        uint8_t va = dominance_margin(ca, da);
        uint8_t vb = dominance_margin(cb, db);
        if (va <= vb) {
            set_channel(&t->nodes[ai].color, da,
                        clamp_u8((int)va - rules->neighbor_step));
            set_channel(&t->nodes[ai].color, db,
                        clamp_u8((int)get_channel(ca, db) + rules->neighbor_step));
        } else {
            set_channel(&t->nodes[bi].color, db,
                        clamp_u8((int)vb - rules->neighbor_step));
            set_channel(&t->nodes[bi].color, da,
                        clamp_u8((int)get_channel(cb, da) + rules->neighbor_step));
        }
        edges[eidx].value = (uint8_t)REL_NULL;
    }
}

static void reinforce_center(Torus *t, size_t x, size_t y, const Rules *rules) {
    size_t w = t->width, h = t->height;
    size_t xl = (x + w - 1) % w;
    size_t yu = (y + h - 1) % h;

    Relationship rels[4] = {
        (Relationship)(t->hedges[node_idx(t, x,  y )].value & 0x3),
        (Relationship)(t->hedges[node_idx(t, xl, y )].value & 0x3),
        (Relationship)(t->vedges[node_idx(t, x,  y )].value & 0x3),
        (Relationship)(t->vedges[node_idx(t, x,  yu)].value & 0x3),
    };

    int counts[4] = {0};
    for (int i = 0; i < 4; i++) counts[(int)rels[i]]++;

    Relationship best_rel = REL_NULL;
    int best = 0;
    for (int r = 1; r <= 3; r++) {
        if (counts[r] > best) { best = counts[r]; best_rel = (Relationship)r; }
    }

    if (best >= (int)rules->reinforce_min) {
        size_t idx = node_idx(t, x, y);
        uint8_t ch = get_channel(t->nodes[idx].color, best_rel);
        set_channel(&t->nodes[idx].color, best_rel,
                    clamp_u8((int)ch + rules->neighbor_step));
    }
}

void process_node(Torus *t, size_t idx, const Rules *rules) {
    size_t w = t->width, h = t->height;
    size_t x = idx % w, y = idx / w;
    size_t xr = (x + 1) % w, xl = (x + w - 1) % w;
    size_t yd = (y + 1) % h, yu = (y + h - 1) % h;

    compete_edge(t, idx, node_idx(t, xr, y),
                 node_idx(t, x,  y),  t->hedges, rules);
    compete_edge(t, idx, node_idx(t, xl, y),
                 node_idx(t, xl, y),  t->hedges, rules);
    compete_edge(t, idx, node_idx(t, x, yd),
                 node_idx(t, x,  y),  t->vedges, rules);
    compete_edge(t, idx, node_idx(t, x, yu),
                 node_idx(t, x,  yu), t->vedges, rules);

    reinforce_center(t, x, y, rules);

    if (rules->mutation_rate > 0 &&
            rng_range(rules->mutation_rate) == 0) {
        t->nodes[idx].color = make_color(rng_u8(), rng_u8(), rng_u8());
    }
}
