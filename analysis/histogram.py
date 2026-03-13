#!/usr/bin/env python3
"""
histogram.py — domain size distribution analysis.

Runs ./torus at multiple mutation rates, averages the log2-binned domain size
histogram over the second half of each run, and plots P(s) vs s on log-log
axes.  At the critical point the distribution should follow a power law
P(s) ~ s^(-τ).  The script fits τ for each mutation rate and highlights the
flattest (most power-law-like) curve.

Usage:
    python3 analysis/histogram.py
    python3 analysis/histogram.py --mutation-rates 4 5 6 7 8 --frames 5000 -o hist.png
"""

import argparse
import subprocess
import sys
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress

ROOT  = Path(__file__).parent.parent
TORUS = ROOT / "torus"
HIST_BINS = 24

# ── simulation ─────────────────────────────────────────────────────────────────

def run_torus(mutation_rate: int, reinforce_min: int, seed: int,
              frames: int, width: int, height: int,
              stats_interval: int = 20) -> np.ndarray | None:
    """
    Run torus and return a (n_frames, HIST_BINS) array of histogram counts,
    covering the second half of the run only.
    """
    cmd = [
        str(TORUS),
        "--headless",
        "--stats-interval", str(stats_interval),
        "--mutation-rate",  str(mutation_rate),
        "--frames",         str(frames),
        str(width), str(height), "0", str(seed), str(reinforce_min),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT mr={mutation_rate}", file=sys.stderr)
        return None

    rows = []
    for line in result.stderr.splitlines():
        if line.startswith("seed=") or line.startswith("frame,"):
            continue
        parts = line.split(",")
        if len(parts) >= 8 + HIST_BINS:
            try:
                hist = [float(parts[8 + i]) for i in range(HIST_BINS)]
                rows.append(hist)
            except ValueError:
                continue

    if len(rows) < 4:
        return None

    arr  = np.array(rows)       # (n_frames, HIST_BINS)
    half = len(arr) // 2
    return arr[half:]           # second half only


def collect(mutation_rate: int, reinforce_min: int, seeds: int,
            frames: int, width: int, height: int) -> np.ndarray | None:
    """Average histogram across seeds; returns mean counts (HIST_BINS,)."""
    accum = []
    for seed in range(seeds):
        h = run_torus(mutation_rate, reinforce_min, seed, frames, width, height)
        if h is not None:
            accum.append(h.mean(axis=0))   # average over frames for this seed
    if not accum:
        return None
    return np.array(accum).mean(axis=0)   # average over seeds


# ── fitting ────────────────────────────────────────────────────────────────────

def fit_powerlaw(counts: np.ndarray, min_bin: int = 1,
                 max_bin: int = HIST_BINS - 4,
                 n_nodes: int | None = None) -> tuple[float, float, float]:
    """
    Fit log(P(s)) ~ -τ log(s) over bins [min_bin, max_bin].
    Returns (tau, r_squared, intercept).
    Only uses bins with non-zero counts.
    n_nodes is accepted for caller convenience but not used.
    """
    sizes = np.array([2**i for i in range(HIST_BINS)], dtype=float)
    mask  = (np.arange(HIST_BINS) >= min_bin) & \
            (np.arange(HIST_BINS) <= max_bin) & \
            (counts > 0)
    if mask.sum() < 3:
        return float("nan"), 0.0, float("nan")

    # Normalise counts to probability density P(s) = count / (bin_width * total)
    # bin i covers [2^i, 2^(i+1)), width = 2^i
    widths = sizes   # bin i has width 2^i
    total  = counts[mask].sum()
    p      = counts / (widths * total)

    logx = np.log10(sizes[mask])
    logy = np.log10(p[mask])

    slope, intercept, r, _, _ = linregress(logx, logy)
    return -slope, r**2, intercept


# ── plotting ───────────────────────────────────────────────────────────────────

def plot(results: dict[int, np.ndarray], reinforce_min: int,
         output_path: str | None, params_label: str = "",
         n_nodes: int | None = None) -> None:
    """
    results: {mutation_rate: mean_hist_counts}
    """
    rates  = sorted(results)
    colors = cm.plasma(np.linspace(0.1, 0.9, len(rates)))
    sizes  = np.array([2**i for i in range(HIST_BINS)], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
    title = f"Domain size distribution  (reinforce_min={reinforce_min})"
    if params_label:
        title += f"\n{params_label}"
    fig.suptitle(title, fontsize=11)

    ax_dist, ax_tau = axes

    taus, r2s = [], []
    for mr, color in zip(rates, colors):
        counts = results[mr]
        # Probability density
        widths = sizes
        total  = counts.sum()
        if total == 0:
            continue
        p = counts / (widths * total)
        mask = p > 0

        tau, r2, intercept = fit_powerlaw(counts, n_nodes=n_nodes)
        taus.append(tau)
        r2s.append(r2)

        label = f"mr={mr}  τ={tau:.2f}  R²={r2:.3f}"
        ax_dist.loglog(sizes[mask], p[mask], "o-", color=color,
                       label=label, linewidth=1.2, markersize=4)

        # Draw fit line
        if not np.isnan(tau):
            fit_x = sizes[mask]
            fit_y = 10**intercept * fit_x**(-tau)
            ax_dist.loglog(fit_x, fit_y, "--", color=color, alpha=0.4, linewidth=1)

    ax_dist.set_xlabel("domain size  s")
    ax_dist.set_ylabel("P(s)  (probability density)")
    ax_dist.set_title("P(s) ~ s^(−τ) at criticality")
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3, which="both")

    # τ and R² vs mutation rate
    probs = [1 / mr for mr in rates]
    ax_tau.plot(probs, taus, "o-", color="steelblue", label="τ (slope)", linewidth=1.5)
    ax_tau.set_xlabel("mutation probability (1 / mutation_rate)")
    ax_tau.set_ylabel("τ  (power-law exponent)", color="steelblue")
    ax_tau.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax_tau.twinx()
    ax2.plot(probs, r2s, "s--", color="firebrick", label="R² (fit quality)", linewidth=1.5)
    ax2.set_ylabel("R²  (fit quality, higher = more power-law)", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax2.set_ylim(0, 1.05)

    ax_tau.set_title("Power-law exponent and fit quality vs mutation rate\n"
                     "(R² peak locates the critical point)")
    ax_tau.grid(True, alpha=0.3)

    lines1, labs1 = ax_tau.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax_tau.legend(lines1 + lines2, labs1 + labs2, fontsize=9)

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"saved: {output_path}", file=sys.stderr)
    else:
        plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

DEFAULT_RATES = [3, 4, 5, 6, 7, 8, 9]

def main():
    parser = argparse.ArgumentParser(
        description="Domain size distribution analysis across mutation rates."
    )
    parser.add_argument("--frames",      type=int, default=3000)
    parser.add_argument("--seeds",       type=int, default=4)
    parser.add_argument("--width",       type=int, default=128)
    parser.add_argument("--height",      type=int, default=128)
    parser.add_argument("--reinforce-min", type=int, default=4,
                        help="reinforce_min value (default 4, sharpest signal)")
    parser.add_argument("--mutation-rates", type=int, nargs="+",
                        default=DEFAULT_RATES)
    parser.add_argument("-o", "--output", help="save plot to file")
    args = parser.parse_args()

    total = len(args.mutation_rates) * args.seeds
    print(f"collecting {len(args.mutation_rates)} rates × {args.seeds} seeds "
          f"= {total} runs  ({args.width}×{args.height}, {args.frames} frames)",
          file=sys.stderr)

    n_nodes = args.width * args.height
    results = {}
    for i, mr in enumerate(args.mutation_rates):
        print(f"  [{i+1}/{len(args.mutation_rates)}]  mr={mr} ...", end="  ",
              file=sys.stderr, flush=True)
        h = collect(mr, args.reinforce_min, args.seeds,
                    args.frames, args.width, args.height)
        if h is not None:
            tau, r2, _ = fit_powerlaw(h, n_nodes=n_nodes)
            print(f"τ={tau:.3f}  R²={r2:.4f}", file=sys.stderr)
            results[mr] = h
        else:
            print("FAILED", file=sys.stderr)

    if not results:
        print("no data collected", file=sys.stderr)
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    params_label = (f"{args.width}×{args.height}  frames={args.frames}  "
                    f"seeds={args.seeds}  rm={args.reinforce_min}  {ts}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    stem = (f"hist_{args.width}x{args.height}_f{args.frames}"
            f"_s{args.seeds}_rm{args.reinforce_min}_{ts}")

    output = args.output or str(results_dir / (stem + ".png"))
    plot(results, args.reinforce_min, output, params_label, n_nodes=n_nodes)


if __name__ == "__main__":
    main()
