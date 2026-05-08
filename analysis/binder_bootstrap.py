"""Bootstrap CIs for Binder cumulant using merged seed data.

Reads merged per-seed CSVs (produced by merge_seeds.py), computes bootstrap
95% CIs for U at each (L, mu), and regenerates fig_binder_sweep.pdf with
shaded CI bands.

Usage:
    venv/bin/python3 analysis/binder_bootstrap.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
CACHEDIR = ROOT / "paper" / "cache"
FIGDIR = ROOT / "paper"

SIZE_COLORS = {
    64:  "#4dac26",
    128: "#2166ac",
    256: "#b2182b",
    384: "#e6ab02",
}
SIZE_MARKERS = {64: "o", 128: "s", 256: "^", 384: "D"}

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)


def binder(rho):
    rho = np.asarray(rho, dtype=float)
    m2 = np.mean(rho**2)
    m4 = np.mean(rho**4)
    return 1.0 - m4 / (3.0 * m2**2) if m2 > 0 else np.nan


def bootstrap_binder(rho, n_boot=N_BOOTSTRAP):
    rho = np.asarray(rho, dtype=float)
    n = len(rho)
    us = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        us[i] = binder(rho[idx])
    return us


def load_merged_seeds(L):
    """Load merged per-seed CSV. Returns {mu: [bd_mean, ...]}."""
    path = RESULTS / f"binder_L{L}_merged_seeds.csv"
    if not path.exists():
        return None
    groups = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            mu = float(row["mutation_prob"])
            groups[mu].append(float(row["bd_mean"]))
    return groups


def load_cached_binder_data(L):
    """Load a cached Binder CSV from CACHEDIR (L=64 or L=128)."""
    for name in [f"binder_L{L}_fine.csv", f"fig7_L{L}.csv"]:
        path = CACHEDIR / name
        if path.exists():
            groups = defaultdict(list)
            with open(path) as f:
                for row in csv.DictReader(f):
                    mu_key = "mu" if "mu" in row else "mutation_prob"
                    groups[float(row[mu_key])].append(float(row["bd_mean"]))
            return groups
    return None


def compute_u_with_ci(groups, alpha=0.05):
    mus = sorted(groups.keys())
    U = []
    U_lo = []
    U_hi = []
    n_seeds = []
    for mu in mus:
        rhos = np.array(groups[mu])
        U.append(binder(rhos))
        bs = bootstrap_binder(rhos)
        lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
        U_lo.append(lo)
        U_hi.append(hi)
        n_seeds.append(len(rhos))
    return np.array(mus), np.array(U), np.array(U_lo), np.array(U_hi), np.array(n_seeds)


def main():
    data_sources = {}
    for L in [64, 128]:
        g = load_cached_binder_data(L)
        if g is not None:
            data_sources[L] = (g, "cache")
    for L in [256, 384]:
        g = load_merged_seeds(L)
        if g is not None:
            data_sources[L] = (g, "merged")

    if not data_sources:
        print("No data sources found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded data for L = {sorted(data_sources.keys())}\n")

    u_data = {}
    for L in sorted(data_sources):
        g, src = data_sources[L]
        mus, U, U_lo, U_hi, n = compute_u_with_ci(g)
        u_data[L] = (mus, U, U_lo, U_hi, n)
        i_min = int(np.argmin(U))
        print(f"L={L:>3d} ({src}): U_min = {U[i_min]:+.4f}  "
              f"[{U_lo[i_min]:+.4f}, {U_hi[i_min]:+.4f}]  "
              f"at mu = {mus[i_min]:.4f}  (n = {n[i_min]})")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for L in sorted(u_data):
        mus, U, U_lo, U_hi, n = u_data[L]
        color = SIZE_COLORS.get(L, "gray")
        marker = SIZE_MARKERS.get(L, "o")
        ax.fill_between(mus, U_lo, U_hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(mus, U, marker=marker, color=color, label=f"$L={L}$",
                linewidth=1.2, markersize=5, markeredgewidth=0.5,
                markeredgecolor="white")

    ax.axhline(2/3, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel(r"Mutation probability $\mu$", fontsize=11)
    ax.set_ylabel(r"Binder cumulant $U_L$", fontsize=11)
    ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.7",
              loc="lower left")
    ax.tick_params(labelsize=9)

    # Inset: zoom on L=128, 256, 384 dip region (L=64 dominates main y-axis)
    axins = ax.inset_axes([0.55, 0.18, 0.42, 0.42])
    for L in [128, 256, 384]:
        if L not in u_data:
            continue
        mus, U, U_lo, U_hi, n = u_data[L]
        color = SIZE_COLORS.get(L, "gray")
        marker = SIZE_MARKERS.get(L, "o")
        axins.fill_between(mus, U_lo, U_hi, color=color, alpha=0.18, linewidth=0)
        axins.plot(mus, U, marker=marker, color=color, linewidth=1.0,
                   markersize=3.5, markeredgewidth=0.4, markeredgecolor="white")
    axins.axhline(2/3, color="0.6", ls=":", lw=0.6)
    axins.set_xlim(0.343, 0.358)
    axins.set_ylim(0.20, 0.70)
    axins.tick_params(labelsize=7)
    axins.set_title(r"$L\in\{128,256,384\}$", fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = FIGDIR / f"fig_binder_sweep.{ext}"
        fig.savefig(path, dpi=300 if ext == "pdf" else 150, bbox_inches="tight")
        print(f"  saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
