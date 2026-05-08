"""Binder data-collapse analysis.

Attempts to collapse U_L(μ) curves for L ∈ {128, 256, 384} onto a single
master curve via the standard FSS scaling form

    U_L(μ) = U_tilde((μ - μ_c) · L^(1/ν))

with two free parameters (μ_c, 1/ν).

L=64 is excluded from the fit by default — the deep negative dip there is
anomalous (small-system fluctuations in only 4096 nodes; the paper already
flags this) and would dominate the cost function.

Outputs:
- Best-fit (μ_c, 1/ν) with bootstrap 95% CIs
- paper/fig_data_collapse.{pdf,png}: two-panel figure showing pre-collapse
  (raw U vs μ) and post-collapse (U vs (μ-μ_c)L^(1/ν)) curves

Usage:
    venv/bin/python3 analysis/data_collapse_binder.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
CACHEDIR = ROOT / "paper" / "cache"
FIGDIR = ROOT / "paper"

SIZE_COLORS = {64: "#4dac26", 128: "#2166ac", 256: "#b2182b", 384: "#e6ab02"}
SIZE_MARKERS = {64: "o", 128: "s", 256: "^", 384: "D"}

L_FIT_3 = [128, 256, 384]   # primary fit (3 sizes)
L_FIT_2 = [256, 384]        # secondary fit (2 largest, asymptotic regime)
N_BOOTSTRAP = 200
RNG = np.random.default_rng(42)


def binder(rho):
    rho = np.asarray(rho, dtype=float)
    m2 = np.mean(rho**2)
    m4 = np.mean(rho**4)
    return 1.0 - m4 / (3.0 * m2**2) if m2 > 0 else np.nan


def load_seeds(L):
    """Return {mu: np.array of bd_mean values}."""
    if L in (256, 384):
        path = RESULTS / f"binder_L{L}_merged_seeds.csv"
    else:
        # L=64, 128: cached fine-grid CSVs
        for name in [f"binder_L{L}_fine.csv", f"fig7_L{L}.csv"]:
            p = CACHEDIR / name
            if p.exists():
                path = p
                break
        else:
            return None

    if not path.exists():
        return None

    groups = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            mu_key = "mu" if "mu" in row else "mutation_prob"
            groups[float(row[mu_key])].append(float(row["bd_mean"]))
    return {mu: np.array(v) for mu, v in groups.items()}


def compute_U(seeds_by_mu):
    mus = np.array(sorted(seeds_by_mu.keys()))
    U = np.array([binder(seeds_by_mu[mu]) for mu in mus])
    return mus, U


def collapse_cost(params, curves, mu_window=0.012):
    """L2 inter-curve scatter on a common scaling-variable grid.

    params: (mu_c, nu_inv)
    curves: list of (L, mus, U) tuples
    mu_window: only use μ within ±mu_window of μ_c (the dip region;
               outside the dip all curves trivially collapse to 2/3 and
               drown out the discriminating signal)
    """
    mu_c, nu_inv = params
    if nu_inv <= 0:
        return 1e6

    # Transform each curve to scaling variable x = (μ-μ_c)·L^(1/ν)
    transformed = []
    for L, mus, U in curves:
        mask = np.abs(mus - mu_c) <= mu_window
        if mask.sum() < 3:
            return 1e6
        x = (mus[mask] - mu_c) * L**nu_inv
        transformed.append((x, U[mask]))

    # Common x range (intersection of all curves)
    x_min = max(t[0].min() for t in transformed)
    x_max = min(t[0].max() for t in transformed)
    if x_max <= x_min:
        return 1e6

    # Sample N points in common range, interpolate each curve, compute std
    N = 50
    x_grid = np.linspace(x_min, x_max, N)
    interp = np.array([np.interp(x_grid, x, U) for x, U in transformed])
    # Per-x std across curves; average over x
    return float(np.mean(np.std(interp, axis=0)))


def fit_collapse(curves, x0=(0.348, 1.0)):
    res = minimize(
        collapse_cost,
        x0=np.array(x0),
        args=(curves,),
        method="Nelder-Mead",
        options=dict(xatol=1e-5, fatol=1e-6, maxiter=2000),
    )
    return res.x, res.fun


def bootstrap_exponents(seeds_by_L, sizes, n_boot=N_BOOTSTRAP):
    """Resample seeds, refit (μ_c, 1/ν) per bootstrap iteration."""
    results = []
    for b in range(n_boot):
        curves = []
        for L in sizes:
            sd = seeds_by_L[L]
            mus = np.array(sorted(sd.keys()))
            U = np.empty_like(mus)
            for i, mu in enumerate(mus):
                rho = sd[mu]
                idx = RNG.integers(0, len(rho), len(rho))
                U[i] = binder(rho[idx])
            curves.append((L, mus, U))
        params, _ = fit_collapse(curves)
        results.append(params)
        if (b + 1) % 50 == 0:
            print(f"    bootstrap {b+1}/{n_boot}", file=sys.stderr)
    return np.array(results)


def report_fit(seeds_by_L, sizes, label):
    """Run central fit + bootstrap for a given size set; return best params."""
    print(f"\n=== {label}: sizes = {sizes} ===")
    curves = [(L, *compute_U(seeds_by_L[L])) for L in sizes]
    best, cost = fit_collapse(curves)
    mu_c_hat, nu_inv_hat = best
    print(f"  Central fit:")
    print(f"    μ_c  = {mu_c_hat:.5f}")
    print(f"    1/ν  = {nu_inv_hat:.4f}  (ν ≈ {1/nu_inv_hat:.3f})")
    print(f"    cost = {cost:.5f}  (mean inter-curve std of U)")

    print(f"  Bootstrap ({N_BOOTSTRAP} resamples):")
    boot = bootstrap_exponents(seeds_by_L, sizes)
    mu_c_lo, mu_c_hi = np.quantile(boot[:, 0], [0.025, 0.975])
    nu_lo, nu_hi = np.quantile(boot[:, 1], [0.025, 0.975])
    print(f"    μ_c  = {mu_c_hat:.5f}  [{mu_c_lo:.5f}, {mu_c_hi:.5f}]")
    print(f"    1/ν  = {nu_inv_hat:.4f}  [{nu_lo:.4f}, {nu_hi:.4f}]")
    print(f"    ν    = {1/nu_inv_hat:.3f}  [{1/nu_hi:.3f}, {1/nu_lo:.3f}]")
    return best, cost, boot


def main():
    seeds_by_L = {}
    for L in [64, 128, 256, 384]:
        s = load_seeds(L)
        if s is not None:
            seeds_by_L[L] = s
            print(f"L={L}: {len(s)} μ values, "
                  f"~{len(next(iter(s.values())))} seeds each")
        else:
            print(f"L={L}: no data found")

    if not all(L in seeds_by_L for L in L_FIT_3):
        print(f"ERROR: missing data for one of {L_FIT_3}", file=sys.stderr)
        sys.exit(1)

    # Two fits: 3-size (primary, includes L=128) and 2-size (largest only)
    best3, cost3, _ = report_fit(seeds_by_L, L_FIT_3, "Three-size collapse")
    best2, cost2, _ = report_fit(seeds_by_L, L_FIT_2, "Two-size collapse "
                                 "(largest only)")
    mu_c3, nu3 = best3
    mu_c2, nu2 = best2

    # Plot: 3-panel — raw curves, 3-size collapse, 2-size collapse
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4))

    # Pre-collapse: include L=64 for context
    for L in sorted(seeds_by_L.keys()):
        mus, U = compute_U(seeds_by_L[L])
        c = SIZE_COLORS.get(L, "gray")
        m = SIZE_MARKERS.get(L, "o")
        ax1.plot(mus, U, marker=m, color=c, label=f"$L={L}$",
                 lw=1.2, ms=5, mew=0.5, mec="white")
    ax1.axhline(2/3, color="0.6", ls=":", lw=0.8)
    ax1.set_xlabel(r"$\mu$", fontsize=11)
    ax1.set_ylabel(r"$U_L$", fontsize=11)
    ax1.set_title("Pre-collapse", fontsize=11)
    ax1.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.7")
    ax1.tick_params(labelsize=9)

    # Middle: 3-size collapse (showing failure)
    for L in L_FIT_3:
        mus, U = compute_U(seeds_by_L[L])
        x = (mus - mu_c3) * L**nu3
        c = SIZE_COLORS.get(L, "gray")
        m = SIZE_MARKERS.get(L, "o")
        ax2.plot(x, U, marker=m, color=c, label=f"$L={L}$",
                 lw=1.2, ms=5, mew=0.5, mec="white")
    ax2.axhline(2/3, color="0.6", ls=":", lw=0.8)
    ax2.set_xlabel(rf"$(\mu - \mu_c)\, L^{{1/\nu}}$  "
                   rf"($\mu_c={mu_c3:.4f}$, $1/\nu={nu3:.2f}$)",
                   fontsize=9.5)
    ax2.set_ylabel(r"$U_L$", fontsize=11)
    ax2.set_title(f"3-size collapse  (cost={cost3:.3f})", fontsize=11)
    ax2.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.7")
    ax2.tick_params(labelsize=9)
    half2 = 0.012 * max(L**nu3 for L in L_FIT_3)
    ax2.set_xlim(-half2, half2)

    # Right: 2-size collapse (L=256, 384) showing successful collapse
    for L in L_FIT_2:
        mus, U = compute_U(seeds_by_L[L])
        x = (mus - mu_c2) * L**nu2
        c = SIZE_COLORS.get(L, "gray")
        m = SIZE_MARKERS.get(L, "o")
        ax3.plot(x, U, marker=m, color=c, label=f"$L={L}$",
                 lw=1.2, ms=5, mew=0.5, mec="white")
    ax3.axhline(2/3, color="0.6", ls=":", lw=0.8)
    ax3.set_xlabel(rf"$(\mu - \mu_c)\, L^{{1/\nu}}$  "
                   rf"($\mu_c={mu_c2:.4f}$, $1/\nu={nu2:.2f}$)",
                   fontsize=9.5)
    ax3.set_ylabel(r"$U_L$", fontsize=11)
    ax3.set_title(f"2-size collapse, $L\\geq 256$  (cost={cost2:.3f})",
                  fontsize=11)
    ax3.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.7")
    ax3.tick_params(labelsize=9)
    half3 = 0.012 * max(L**nu2 for L in L_FIT_2)
    ax3.set_xlim(-half3, half3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = FIGDIR / f"fig_data_collapse.{ext}"
        fig.savefig(path, dpi=300 if ext == "pdf" else 150,
                    bbox_inches="tight")
        print(f"  saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
