#define _POSIX_C_SOURCE 200809L

#include "sim.h"
#include "dgraph.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>
#include <unistd.h>

static FILE *tty_open(void) { return fopen("/dev/tty", "w"); }

static void tty_progress(FILE *tty, uint64_t frame, uint64_t step,
                         double fps, const DStats *st) {
    if (!tty) return;
    if (st) {
        fprintf(tty,
            "\rframe %-7"PRIu64"  step %-12"PRIu64"  "
            "bd=%.3f  domains=%-6zu  deg=%.1f±%.1f  %.1f fr/s   ",
            frame, step,
            st->boundary_density, st->domain_count,
            st->mean_degree, st->degree_variance,
            fps);
    } else {
        fprintf(tty,
            "\rframe %-7"PRIu64"  step %-12"PRIu64"  %.1f fr/s   ",
            frame, step, fps);
    }
    fflush(tty);
}

// ── output ────────────────────────────────────────────────────────────────────

static void write_frame(FILE *fp, const DGraph *g, size_t w, size_t h,
                        uint8_t *buf, bool degree_viz, size_t degree_scale) {
    size_t n = w * h;
    for (size_t i = 0; i < n; i++) {
        if (degree_viz) {
            // heatmap: blue (low degree) → green → red (high degree)
            double t = (degree_scale > 0)
                ? (double)g->nodes[i].n_edges / (double)degree_scale
                : 0.0;
            if (t > 1.0) t = 1.0;
            buf[i*3+0] = (uint8_t)(255 * t);
            buf[i*3+1] = (uint8_t)(255 * (t < 0.5 ? 2*t : 2*(1-t)));
            buf[i*3+2] = (uint8_t)(255 * (1.0 - t));
        } else {
            uint32_t c = g->nodes[i].color;
            buf[i*3+0] = color_r(c);
            buf[i*3+1] = color_g(c);
            buf[i*3+2] = color_b(c);
        }
    }
    fwrite(buf, 1, n * 3, fp);
}

static void print_stats_header(void) {
    fprintf(stderr,
        "frame,step,boundary_density,domain_count,mean_domain_size,"
        "frac_r,frac_g,frac_b");
    for (int i = 0; i < DGRAPH_HIST_BINS; i++)
        fprintf(stderr, ",h%d", i);
    fprintf(stderr, ",mean_degree,degree_variance,max_degree\n");
}

static void print_stats(uint64_t frame, uint64_t step, const DStats *s) {
    fprintf(stderr, "%"PRIu64",%"PRIu64",%.6f,%zu,%.2f,%.4f,%.4f,%.4f",
        frame, step,
        s->boundary_density,
        s->domain_count,
        s->mean_domain_size,
        s->type_fractions[0],
        s->type_fractions[1],
        s->type_fractions[2]);
    for (int i = 0; i < DGRAPH_HIST_BINS; i++)
        fprintf(stderr, ",%zu", s->size_histogram[i]);
    fprintf(stderr, ",%.4f,%.4f,%zu\n",
        s->mean_degree, s->degree_variance, s->max_degree);
}

// ── run loop ──────────────────────────────────────────────────────────────────

#define PROGRESS_INTERVAL 5

static void run(DGraph *g, size_t w, size_t h,
                size_t steps_per_frame, const Rules *rules,
                uint32_t topo_rate,
                bool headless, size_t stats_interval, uint64_t max_frames,
                bool degree_viz, size_t degree_scale,
                FILE *out, FILE *tty) {
    size_t n = g->n;

    uint8_t *frame_buf = headless ? NULL : malloc(n * 3);
    uint8_t *scratch   = (stats_interval > 0) ? malloc(n) : NULL;

    if ((!headless && !frame_buf) || (stats_interval > 0 && !scratch)) {
        fprintf(stderr, "out of memory\n");
        free(frame_buf); free(scratch);
        return;
    }

    if (stats_interval > 0) print_stats_header();

    uint64_t frame = 0, step = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (; max_frames == 0 || frame < max_frames;) {
        for (size_t s = 0; s < steps_per_frame; s++, step++) {
            size_t i = rng_range(n);
            dgraph_process_node(g, i, rules, topo_rate);
        }

        if (!headless) {
            write_frame(out, g, w, h, frame_buf, degree_viz, degree_scale);
            if (ferror(out)) break;
        }

        DStats st_buf;
        DStats *st = NULL;
        if (stats_interval > 0 && frame % stats_interval == 0) {
            dgraph_compute_stats(g, &st_buf, scratch);
            print_stats(frame, step, &st_buf);
            st = &st_buf;
        }

        if (tty && frame % PROGRESS_INTERVAL == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec)
                           + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            double fps = elapsed > 0 ? (double)PROGRESS_INTERVAL / elapsed : 0.0;
            tty_progress(tty, frame, step, fps, st);
            t0 = t1;
        }

        frame++;
    }

    if (tty) fprintf(tty, "\n");
    free(frame_buf);
    free(scratch);
}

// ── main ──────────────────────────────────────────────────────────────────────

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s [options] [width] [height] [steps_per_frame] [seed] [reinforce_min]\n"
        "\n"
        "  width           grid width (default 128)\n"
        "  height          grid height (default 128)\n"
        "  steps_per_frame simulation steps between frames (default width*height)\n"
        "  seed            RNG seed (default time-based)\n"
        "  reinforce_min   2-4 neighbors that must agree (default 2)\n"
        "\n"
        "options:\n"
        "  --headless             skip video output\n"
        "  --stats-interval N     emit CSV stats to stderr every N frames (default 0)\n"
        "  --mutation-rate N      1-in-N mutation chance per node (default 2000)\n"
        "  --mutation-prob P      float mutation probability, overrides --mutation-rate\n"
        "  --topo-rate N          1-in-N topology change per competition (default n_nodes)\n"
        "                         0 = frozen topology (fixed-graph mode)\n"
        "  --max-degree N         cap on degree growth via edge formation (0=unlimited)\n"
        "  --ordered-init         start fully ordered (all nodes type R, all edges aligned)\n"
        "  --degree-viz           color by degree: blue=low, green=mid, red=high (scale=max-degree)\n"
        "  --frames N             stop after N frames\n",
        argv0);
}

int main(int argc, char **argv) {
    size_t       width           = 128;
    size_t       height          = 128;
    size_t       steps_per_frame = 0;
    unsigned int seed            = (unsigned int)time(NULL);
    uint8_t      reinforce_min   = 2;
    bool         headless        = false;
    size_t       stats_interval  = 0;
    uint32_t     mutation_rate   = 2000;
    double       mutation_prob   = 0.0; // 0 = use mutation_rate instead
    uint32_t     topo_rate       = 0;   // 0 = use n_nodes (set after parsing)
    bool         topo_rate_set   = false;
    uint32_t     max_degree      = 0;   // 0 = unlimited
    uint64_t     max_frames      = 0;
    bool         ordered_init    = false;
    bool         degree_viz      = false;

    int argi = 1;
    for (; argi < argc; argi++) {
        if (strcmp(argv[argi], "--headless") == 0) {
            headless = true;
        } else if (strcmp(argv[argi], "--stats-interval") == 0 && argi+1 < argc) {
            stats_interval = (size_t)strtoull(argv[++argi], NULL, 10);
        } else if (strcmp(argv[argi], "--mutation-rate") == 0 && argi+1 < argc) {
            mutation_rate = (uint32_t)strtoul(argv[++argi], NULL, 10);
        } else if (strcmp(argv[argi], "--mutation-prob") == 0 && argi+1 < argc) {
            mutation_prob = strtod(argv[++argi], NULL);
        } else if (strcmp(argv[argi], "--topo-rate") == 0 && argi+1 < argc) {
            topo_rate = (uint32_t)strtoul(argv[++argi], NULL, 10);
            topo_rate_set = true;
        } else if (strcmp(argv[argi], "--max-degree") == 0 && argi+1 < argc) {
            max_degree = (uint32_t)strtoul(argv[++argi], NULL, 10);
        } else if (strcmp(argv[argi], "--frames") == 0 && argi+1 < argc) {
            max_frames = (uint64_t)strtoull(argv[++argi], NULL, 10);
        } else if (strcmp(argv[argi], "--ordered-init") == 0) {
            ordered_init = true;
        } else if (strcmp(argv[argi], "--degree-viz") == 0) {
            degree_viz = true;
        } else {
            break;
        }
    }

    int pos = argc - argi;
    if (pos > 0) width           = (size_t)strtoull(argv[argi+0], NULL, 10);
    if (pos > 1) height          = (size_t)strtoull(argv[argi+1], NULL, 10);
    if (pos > 2) steps_per_frame = (size_t)strtoull(argv[argi+2], NULL, 10);
    if (pos > 3) seed            = (unsigned int)strtoul(argv[argi+3], NULL, 10);
    if (pos > 4) reinforce_min   = (uint8_t)strtoul(argv[argi+4], NULL, 10);

    if (pos > 5 || width < 2 || height < 2 ||
            reinforce_min < 2 || reinforce_min > 4) {
        usage(argv[0]);
        return 1;
    }

    if (steps_per_frame == 0) steps_per_frame = width * height;
    if (!topo_rate_set) topo_rate = (uint32_t)(width * height);

    if (!headless && isatty(STDOUT_FILENO)) {
        fprintf(stderr,
            "error: stdout is a terminal — pipe to ffplay/ffmpeg or use --headless\n\n");
        usage(argv[0]);
        return 1;
    }

    rng_seed((uint64_t)seed);

    DGraph g = {0};
    if (!dgraph_init(&g, width, height)) {
        fprintf(stderr, "failed to allocate graph\n");
        return 1;
    }

    if (ordered_init) {
        // Overwrite random init: all nodes strongly type R, all edges aligned.
        for (size_t i = 0; i < g.n; i++) {
            g.nodes[i].color = make_color(200, 50, 50);
            for (size_t e = 0; e < g.nodes[i].n_edges; e++)
                g.nodes[i].edges[e].rel = (uint8_t)REL_RED;
        }
    }

    Rules rules = {
        .neighbor_step   = 8,
        .reinforce_min   = reinforce_min,
        .mutation_rate   = (mutation_prob > 0.0) ? 0 : mutation_rate,
        .mutation_thresh = prob_to_thresh(mutation_prob),
        .max_degree      = max_degree,
    };

    FILE *tty = tty_open();

    {
        char info[320];
        double eff_prob = (mutation_prob > 0.0) ? mutation_prob
                        : (mutation_rate > 0 ? 1.0 / mutation_rate : 0.0);
        snprintf(info, sizeof(info),
            "seed=%u  %zux%zu  steps/frame=%zu  reinforce_min=%u  "
            "mutation_prob=%.6f  topo_rate=%u  max_degree=%u  "
            "ordered_init=%s  headless=%s  stats_interval=%zu\n",
            seed, width, height, steps_per_frame, reinforce_min,
            eff_prob, topo_rate, max_degree,
            ordered_init ? "yes" : "no",
            headless ? "yes" : "no", stats_interval);
        fprintf(stderr, "%s", info);
        if (tty) fprintf(tty, "%s", info);
    }

    size_t degree_scale = (max_degree > 0) ? max_degree : 8;
    run(&g, width, height, steps_per_frame, &rules, topo_rate,
        headless, stats_interval, max_frames,
        degree_viz, degree_scale,
        stdout, tty);

    if (tty) fclose(tty);
    dgraph_free(&g);
    return 0;
}
