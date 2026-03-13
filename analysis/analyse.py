#!/usr/bin/env python3
"""
analyse.py — read torus stats CSV and plot observables.

Usage:
    # collect stats then analyse
    ./torus --headless --stats-interval 10 2>stats.csv
    python3 analysis/analyse.py stats.csv

    # or pipe directly (no live plot, saves to file)
    ./torus --headless --stats-interval 10 2>&1 >/dev/null | python3 analysis/analyse.py -

The CSV format expected (from torus stderr):
    frame,step,boundary_density,domain_count,mean_domain_size,frac_r,frac_g,frac_b
"""

import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict[str, np.ndarray]:
    """Read torus stats CSV; skip non-CSV lines (e.g. the 'seed=...' info line)."""
    rows = []
    src = sys.stdin if path == "-" else open(path)
    reader = csv.DictReader(
        (line for line in src if not line.startswith("seed=")),
    )
    for row in reader:
        rows.append({k: float(v) for k, v in row.items()})
    if path != "-":
        src.close()
    if not rows:
        raise ValueError("no data rows found in input")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ── equilibrium detection ─────────────────────────────────────────────────────

def detect_equilibrium(series: np.ndarray, window: int = 20,
                        threshold: float = 0.005) -> int | None:
    """
    Find the first index where the rolling mean of `series` has stabilised.

    'Stabilised' means the standard deviation of the rolling mean over the
    last `window` points falls below `threshold` * (global mean of series).

    Returns the frame index of the detected equilibrium, or None if not found.
    """
    if len(series) < window * 2:
        return None

    rolling_mean = np.convolve(series, np.ones(window) / window, mode="valid")
    # rolling_mean[i] corresponds to original index i + window//2
    offset = window // 2

    for i in range(window, len(rolling_mean)):
        segment = rolling_mean[i - window : i]
        rel_std = np.std(segment) / (np.mean(series) + 1e-12)
        if rel_std < threshold:
            return i + offset  # back to original index

    return None


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(data: dict[str, np.ndarray], eq_idx: int | None,
         output_path: str | None) -> None:

    frames = data["frame"].astype(int)
    steps  = data["step"].astype(int)

    fig = plt.figure(figsize=(12, 9), tight_layout=True)
    fig.suptitle("Torus simulation observables", fontsize=13)
    gs = gridspec.GridSpec(3, 2, figure=fig)

    def eq_vline(ax):
        if eq_idx is not None and eq_idx < len(frames):
            ax.axvline(frames[eq_idx], color="red", linestyle="--",
                       linewidth=0.8, label=f"eq ≈ frame {frames[eq_idx]}")

    # ── boundary density ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.plot(frames, data["boundary_density"], linewidth=0.8, color="steelblue")
    window = min(20, len(frames) // 4)
    if window > 1:
        rm = np.convolve(data["boundary_density"],
                         np.ones(window) / window, mode="valid")
        rm_frames = frames[window // 2 : window // 2 + len(rm)]
        ax.plot(rm_frames, rm, color="navy", linewidth=1.5,
                label=f"rolling mean (w={window})")
    eq_vline(ax)
    ax.set_ylabel("boundary density")
    ax.set_xlabel("frame")
    ax.set_title("Boundary density (primary order parameter)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── domain count ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(frames, data["domain_count"], linewidth=0.8, color="darkorange")
    eq_vline(ax)
    ax.set_ylabel("domain count")
    ax.set_xlabel("frame")
    ax.set_title("Domain count")
    ax.grid(True, alpha=0.3)

    # ── mean domain size ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(frames, data["mean_domain_size"], linewidth=0.8, color="forestgreen")
    eq_vline(ax)
    ax.set_ylabel("mean domain size (nodes)")
    ax.set_xlabel("frame")
    ax.set_title("Mean domain size")
    ax.grid(True, alpha=0.3)

    # ── type fractions ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(frames, data["frac_r"], linewidth=0.8, color="red",   label="R")
    ax.plot(frames, data["frac_g"], linewidth=0.8, color="green", label="G")
    ax.plot(frames, data["frac_b"], linewidth=0.8, color="blue",  label="B")
    ax.axhline(1/3, color="black", linestyle=":", linewidth=0.6, label="1/3")
    eq_vline(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of nodes")
    ax.set_xlabel("frame")
    ax.set_title("Dominant-type fractions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── boundary density variance (rolling) ───────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    var_window = max(10, len(frames) // 20)
    if len(frames) > var_window:
        bd = data["boundary_density"]
        rolling_var = np.array([
            np.var(bd[max(0, i - var_window) : i + 1])
            for i in range(len(bd))
        ])
        ax.plot(frames, rolling_var, linewidth=0.8, color="purple")
        eq_vline(ax)
    ax.set_ylabel("rolling variance")
    ax.set_xlabel("frame")
    ax.set_title(f"Boundary density variance (w={var_window})")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"saved: {output_path}", file=sys.stderr)
    else:
        plt.show()


# ── summary ───────────────────────────────────────────────────────────────────

def summarise(data: dict[str, np.ndarray], eq_idx: int | None) -> None:
    n = len(data["frame"])
    print(f"frames loaded  : {n}")
    if eq_idx is not None:
        eq_frame = int(data["frame"][eq_idx])
        eq_step  = int(data["step"][eq_idx])
        print(f"equilibrium    : frame {eq_frame}  (step {eq_step},"
              f" {100*eq_idx/n:.0f}% of run)")
        post = slice(eq_idx, None)
    else:
        print("equilibrium    : not detected")
        post = slice(0, None)

    bd = data["boundary_density"][post]
    print(f"boundary density (post-eq): mean={bd.mean():.4f}  std={bd.std():.4f}")
    dc = data["domain_count"][post]
    print(f"domain count    (post-eq): mean={dc.mean():.1f}  std={dc.std():.1f}")
    ds = data["mean_domain_size"][post]
    print(f"mean domain size(post-eq): mean={ds.mean():.1f}  std={ds.std():.1f}")
    print("type fractions  (post-eq):", "  ".join(
        f"{c}={data[f'frac_{c}'][post].mean():.3f}"
        for c in ("r", "g", "b")
    ))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot and analyse torus simulation stats CSV."
    )
    parser.add_argument("input", nargs="?", default="-",
                        help="stats CSV file, or - for stdin (default: -)")
    parser.add_argument("-o", "--output",
                        help="save plot to file instead of displaying it")
    parser.add_argument("--eq-window", type=int, default=20,
                        help="rolling window for equilibrium detection (default 20)")
    parser.add_argument("--eq-threshold", type=float, default=0.005,
                        help="relative-std threshold for equilibrium (default 0.005)")
    args = parser.parse_args()

    data   = load_csv(args.input)
    eq_idx = detect_equilibrium(
        data["boundary_density"],
        window=args.eq_window,
        threshold=args.eq_threshold,
    )
    summarise(data, eq_idx)
    plot(data, eq_idx, args.output)


if __name__ == "__main__":
    main()
