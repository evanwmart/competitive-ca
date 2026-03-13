#define _POSIX_C_SOURCE 200809L

#include "sim.h"
#include "stats.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>
#include <unistd.h>

// Progress is written to /dev/tty so it reaches the terminal even when
// stderr is redirected to a file.  tty_open() returns NULL silently if
// no terminal is available (e.g. running in CI).
static FILE *tty_open(void) { return fopen("/dev/tty", "w"); }

static void tty_progress(FILE *tty, uint64_t frame, uint64_t step,
                         double fps, const Stats *st) {
    if (!tty) return;
    if (st) {
        fprintf(tty,
            "\rframe %-7"PRIu64"  step %-12"PRIu64"  "
            "bd=%.3f  domains=%-6zu  %.1f fr/s   ",
            frame, step,
            st->boundary_density, st->domain_count, fps);
    } else {
        fprintf(tty,
            "\rframe %-7"PRIu64"  step %-12"PRIu64"  %.1f fr/s   ",
            frame, step, fps);
    }
    fflush(tty);
}

// ── output ────────────────────────────────────────────────────────────────────

static void write_frame(FILE *fp, const Torus *t, uint8_t *buf) {
    size_t n = t->width * t->height;
    for (size_t i = 0; i < n; i++) {
        uint32_t c = t->nodes[i].color;
        buf[i*3+0] = color_r(c);
        buf[i*3+1] = color_g(c);
        buf[i*3+2] = color_b(c);
    }
    fwrite(buf, 1, n * 3, fp);
}

static void print_stats_header(void) {
    fprintf(stderr,
        "frame,step,boundary_density,domain_count,mean_domain_size,"
        "frac_r,frac_g,frac_b");
    for (int i = 0; i < HIST_BINS; i++)
        fprintf(stderr, ",h%d", i);
    fprintf(stderr, "\n");
}

static void print_stats(uint64_t frame, uint64_t step, const Stats *s) {
    fprintf(stderr, "%"PRIu64",%"PRIu64",%.6f,%zu,%.2f,%.4f,%.4f,%.4f",
        frame, step,
        s->boundary_density,
        s->domain_count,
        s->mean_domain_size,
        s->type_fractions[0],
        s->type_fractions[1],
        s->type_fractions[2]);
    for (int i = 0; i < HIST_BINS; i++)
        fprintf(stderr, ",%zu", s->size_histogram[i]);
    fprintf(stderr, "\n");
}

// ── run loop ──────────────────────────────────────────────────────────────────

#define PROGRESS_INTERVAL 5   // update tty every N frames

static void run(Torus *t, size_t steps_per_frame, const Rules *rules,
                bool headless, size_t stats_interval, uint64_t max_frames,
                FILE *out, FILE *tty) {
    size_t n = t->width * t->height;

    uint8_t *frame_buf = headless ? NULL : malloc(n * 3);
    uint8_t *scratch   = (stats_interval > 0) ? malloc(n) : NULL;

    if ((!headless && !frame_buf) || (stats_interval > 0 && !scratch)) {
        fprintf(stderr, "out of memory\n");
        free(frame_buf); free(scratch);
        return;
    }

    if (stats_interval > 0) print_stats_header();

    uint64_t frame = 0;
    uint64_t step  = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (; max_frames == 0 || frame < max_frames;) {
        for (size_t s = 0; s < steps_per_frame; s++, step++) {
            size_t i = rng_range(n);
            process_node(t, i, rules);
        }

        if (!headless) {
            write_frame(out, t, frame_buf);
            if (ferror(out)) break;
        }

        Stats st_buf;
        Stats *st = NULL;

        if (stats_interval > 0 && frame % stats_interval == 0) {
            compute_stats(t, &st_buf, scratch);
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
        "  width           grid width (default 256)\n"
        "  height          grid height (default 256)\n"
        "  steps_per_frame simulation steps between frames (default width*height)\n"
        "  seed            RNG seed (default time-based)\n"
        "  reinforce_min   2-4 neighbors that must agree (default 2)\n"
        "\n"
        "options:\n"
        "  --headless             skip video output (for stats-only runs)\n"
        "  --stats-interval N     emit CSV stats to stderr every N frames (default 0 = off)\n"
        "  --mutation-rate N      1-in-N mutation chance per node (default 2000, 0 = off)\n"
        "  --frames N             stop after N frames (default: run forever)\n"
        "\n"
        "  live:   %s | ffplay -f rawvideo -pixel_format rgb24 -video_size WxH -framerate 60 -i pipe:0\n"
        "  record: %s | ffmpeg -f rawvideo -pixel_format rgb24 -video_size WxH -framerate 60 -i pipe:0 out.mp4\n"
        "  stats:  %s --headless --stats-interval 100 > /dev/null\n",
        argv0, argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    size_t       width          = 256;
    size_t       height         = 256;
    size_t       steps_per_frame = 0;
    unsigned int seed           = (unsigned int)time(NULL);
    uint8_t      reinforce_min  = 2;
    bool         headless       = false;
    size_t       stats_interval = 0;
    uint32_t     mutation_rate  = 2000;
    uint64_t     max_frames     = 0;   // 0 = infinite

    // parse options first
    int argi = 1;
    for (; argi < argc; argi++) {
        if (strcmp(argv[argi], "--headless") == 0) {
            headless = true;
        } else if (strcmp(argv[argi], "--stats-interval") == 0 && argi + 1 < argc) {
            stats_interval = (size_t)strtoull(argv[++argi], NULL, 10);
        } else if (strcmp(argv[argi], "--mutation-rate") == 0 && argi + 1 < argc) {
            mutation_rate = (uint32_t)strtoul(argv[++argi], NULL, 10);
        } else if (strcmp(argv[argi], "--frames") == 0 && argi + 1 < argc) {
            max_frames = (uint64_t)strtoull(argv[++argi], NULL, 10);
        } else {
            break;  // positional args start here
        }
    }

    int pos = argc - argi;
    if (pos > 0) width          = (size_t)strtoull(argv[argi+0], NULL, 10);
    if (pos > 1) height         = (size_t)strtoull(argv[argi+1], NULL, 10);
    if (pos > 2) steps_per_frame = (size_t)strtoull(argv[argi+2], NULL, 10);
    if (pos > 3) seed           = (unsigned int)strtoul(argv[argi+3], NULL, 10);
    if (pos > 4) reinforce_min  = (uint8_t)strtoul(argv[argi+4], NULL, 10);

    if (pos > 5 || width < 2 || height < 2 ||
            reinforce_min < 2 || reinforce_min > 4) {
        usage(argv[0]);
        return 1;
    }

    if (steps_per_frame == 0) steps_per_frame = width * height;

    if (!headless && isatty(STDOUT_FILENO)) {
        fprintf(stderr, "error: stdout is a terminal — pipe to ffplay/ffmpeg, or use --headless\n\n");
        usage(argv[0]);
        return 1;
    }

    rng_seed((uint64_t)seed);

    Torus t = {0};
    if (!torus_init(&t, width, height)) {
        fprintf(stderr, "failed to allocate torus\n");
        return 1;
    }

    Rules rules = {
        .neighbor_step = 8,
        .reinforce_min = reinforce_min,
        .mutation_rate = mutation_rate,
    };

    FILE *tty = tty_open();

    // info line: always to stderr (captured in CSV redirect) AND tty
    {
        char info[256];
        snprintf(info, sizeof(info),
            "seed=%u  %zux%zu  steps/frame=%zu  reinforce_min=%u  "
            "mutation_rate=%u  headless=%s  stats_interval=%zu\n",
            seed, width, height, steps_per_frame, reinforce_min,
            mutation_rate, headless ? "yes" : "no", stats_interval);
        fprintf(stderr, "%s", info);
        if (tty) fprintf(tty, "%s", info);
    }

    run(&t, steps_per_frame, &rules, headless, stats_interval, max_frames, stdout, tty);
    if (tty) fclose(tty);

    torus_free(&t);
    return 0;
}
