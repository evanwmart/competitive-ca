#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// ── fast RNG (xorshift64) ─────────────────────────────────────────────────────
// Replaces rand()/srand(). Each process has its own state — no locks, no calls.

extern uint64_t g_rng;

static inline void rng_seed(uint64_t seed) {
    // splitmix64 mixing: distinct seeds always produce distinct initial states
    seed += 0x9e3779b97f4a7c15ULL;
    seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
    g_rng = seed ^ (seed >> 31);
    if (!g_rng) g_rng = 1;
}

static inline uint64_t rng_next(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

// Use top 32 bits — high bits have better distribution in xorshift generators.
// Low bits are weaker due to LFSR structure; power-of-2 moduli (8, 16 …) would
// only sample them, biasing the effective rate.
static inline size_t rng_range(size_t n) {
    return (size_t)((rng_next() >> 32) % (uint64_t)n);
}

static inline uint8_t rng_u8(void) { return (uint8_t)(rng_next() >> 56); }

// ── types ─────────────────────────────────────────────────────────────────────

typedef enum {
    REL_NULL  = 0,
    REL_RED   = 1,
    REL_GREEN = 2,
    REL_BLUE  = 3
} Relationship;

typedef struct {
    uint32_t color;
} Node;

typedef struct {
    uint8_t value;  // low 2 bits
} Edge;

typedef struct {
    size_t width, height;
    Node  *nodes;   // [y * width + x]
    Edge  *hedges;  // horizontal: right edge of (x,y) at [y*width+x]
    Edge  *vedges;  // vertical:   down  edge of (x,y) at [y*width+x]
} Torus;

typedef struct {
    uint8_t  neighbor_step;
    uint8_t  reinforce_min;   // 2–4: neighbors that must agree to self-reinforce
    uint32_t mutation_rate;   // 0=off, N=1-in-N per node selection
    uint32_t mutation_thresh; // alternative: fire when (rng>>32) < thresh (0=use rate)
    uint32_t max_degree;      // dgraph only: cap on edge formation (0=unlimited)
    uint8_t  local_formation; // dgraph only: 1=restrict new edges to distance-2 (neighbour-of-neighbour)
} Rules;

// Convert float probability [0,1] to mutation_thresh.
static inline uint32_t prob_to_thresh(double p) {
    if (p <= 0.0) return 0;
    if (p >= 1.0) return UINT32_MAX;
    return (uint32_t)(p * 4294967296.0);
}

// Check mutation: returns true if node should mutate this step.
static inline bool mutation_fires(const Rules *rules) {
    if (rules->mutation_thresh > 0)
        return (uint32_t)(rng_next() >> 32) < rules->mutation_thresh;
    return rules->mutation_rate > 0 && rng_range(rules->mutation_rate) == 0;
}

// ── color helpers ─────────────────────────────────────────────────────────────

static inline uint8_t clamp_u8(int x) {
    return x < 0 ? 0 : x > 255 ? 255 : (uint8_t)x;
}

static inline uint8_t color_r(uint32_t c) { return (uint8_t)((c >> 16) & 0xFF); }
static inline uint8_t color_g(uint32_t c) { return (uint8_t)((c >>  8) & 0xFF); }
static inline uint8_t color_b(uint32_t c) { return (uint8_t)( c        & 0xFF); }

static inline uint32_t make_color(uint8_t r, uint8_t g, uint8_t b) {
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}

static inline uint8_t get_channel(uint32_t c, Relationship rel) {
    switch (rel) {
        case REL_RED:   return color_r(c);
        case REL_GREEN: return color_g(c);
        case REL_BLUE:  return color_b(c);
        case REL_NULL:  return 0;
    }
    return 0;
}

static inline void set_channel(uint32_t *c, Relationship rel, uint8_t v) {
    uint8_t r = color_r(*c), g = color_g(*c), b = color_b(*c);
    switch (rel) {
        case REL_RED:   r = v; break;
        case REL_GREEN: g = v; break;
        case REL_BLUE:  b = v; break;
        case REL_NULL:  break;
    }
    *c = make_color(r, g, b);
}

static inline Relationship node_dominant(uint32_t c) {
    uint8_t r = color_r(c), g = color_g(c), b = color_b(c);
    if (r > g && r > b) return REL_RED;
    if (g > b)          return REL_GREEN;
    if (b > r)          return REL_BLUE;
    switch (c & 0x3) {
        case 0: return REL_RED;
        case 1: return REL_GREEN;
        case 2: return REL_BLUE;
        default: return REL_RED;
    }
}

static inline uint8_t dominance_margin(uint32_t c, Relationship rel) {
    uint8_t dom = get_channel(c, rel);
    uint8_t r = color_r(c), g = color_g(c), b = color_b(c);
    uint8_t a, bb;
    switch (rel) {
        case REL_RED:   a = g;  bb = b; break;
        case REL_GREEN: a = r;  bb = b; break;
        case REL_BLUE:  a = r;  bb = g; break;
        default: return 0;
    }
    uint8_t mx = a > bb ? a : bb;
    return dom > mx ? dom - mx : 0;
}

// ── torus ─────────────────────────────────────────────────────────────────────

bool torus_init(Torus *t, size_t w, size_t h);
void torus_free(Torus *t);

static inline size_t node_idx(const Torus *t, size_t x, size_t y) {
    return y * t->width + x;
}

// ── simulation ────────────────────────────────────────────────────────────────

void process_node(Torus *t, size_t idx, const Rules *rules);
