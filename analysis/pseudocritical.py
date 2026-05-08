#!/usr/bin/env python3
"""
pseudocritical.py — Track μ_c(L) via peak seed-to-seed variance of ρ_b.

For each system size L, groups seeds by μ, computes var(bd_mean), finds the
peak, refines via parabolic interpolation, and plots:
  (a) var(ρ_b) vs μ for all L
  (b) μ_c(L) vs 1/L  (pseudocritical drift)

Usage:
    venv/bin/python3 analysis/pseudocritical.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figures import (C_L64, C_L128, C_L256, FONT_TITLE, FONT_LABEL,
                     FONT_TICK, FONT_LEGEND, FONT_ANNOT, FIGDIR, CACHEDIR,
                     save, plt, panel_label, load_cache)


# ── data sources ────────────────────────────────────────────────────────────

SOURCES = {
    64:  ('fig7_L64',      C_L64,  'D'),
    128: ('fig3_dyn_frac', C_L128, 'o'),
    256: ('fig7_L256',     C_L256, 's'),
}


def variance_curve(data):
    """Return sorted arrays of (mu, var(bd_mean), n_seeds) per μ group."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in data:
        groups[round(r['mu'], 6)].append(r['bd_mean'])

    mus, variances, counts = [], [], []
    for mu in sorted(groups):
        bds = np.array(groups[mu])
        mus.append(mu)
        variances.append(np.var(bds))
        counts.append(len(bds))
    return np.array(mus), np.array(variances), np.array(counts)


def parabolic_peak(mus, variances):
    """Refine peak location by fitting a parabola through peak ± 1 neighbor.

    Returns (mu_c, var_peak) at the vertex of the fitted parabola.
    """
    idx = np.argmax(variances)
    # Need at least one neighbor on each side
    if idx == 0 or idx == len(mus) - 1:
        return mus[idx], variances[idx]

    x = mus[idx - 1: idx + 2]
    y = variances[idx - 1: idx + 2]
    # Fit y = a*x^2 + b*x + c
    coeffs = np.polyfit(x, y, 2)
    a, b, c = coeffs
    if a >= 0:
        # Parabola opens upward — no local max, fall back to grid peak
        return mus[idx], variances[idx]
    mu_c = -b / (2 * a)
    var_peak = np.polyval(coeffs, mu_c)
    return mu_c, var_peak


# ── main ────────────────────────────────────────────────────────────────────

def main():
    FIGDIR.mkdir(exist_ok=True)

    # Collect per-L variance curves and pseudocritical points
    curves = {}   # L -> (mus, variances, counts)
    peaks = {}    # L -> (mu_c, var_peak)

    for L in sorted(SOURCES):
        cache_name, color, marker = SOURCES[L]
        data = load_cache(cache_name)
        if data is None:
            print(f'  WARNING: cache {cache_name}.csv not found, skipping L={L}')
            continue
        mus, variances, counts = variance_curve(data)
        mu_c, var_peak = parabolic_peak(mus, variances)
        curves[L] = (mus, variances, counts)
        peaks[L] = (mu_c, var_peak)

    if len(peaks) < 2:
        print('Not enough data to plot. Exiting.')
        return

    # ── print results ───────────────────────────────────────────────────────
    print()
    print('Pseudocritical points μ_c(L):')
    print(f'  {"L":>5s}  {"μ_c":>8s}  {"var_peak":>12s}')
    for L in sorted(peaks):
        mu_c, var_peak = peaks[L]
        print(f'  {L:5d}  {mu_c:8.5f}  {var_peak:12.4e}')

    # ── figure ──────────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

    # (a) var(ρ_b) vs μ
    for L in sorted(curves):
        cache_name, color, marker = SOURCES[L]
        mus, variances, counts = curves[L]
        mu_c, var_peak = peaks[L]
        ax_a.plot(mus, variances, f'{marker}-', color=color, lw=1.5, ms=6,
                  label=f'L = {L}', zorder=3)
        # Mark the refined peak
        ax_a.plot(mu_c, var_peak, '*', color=color, ms=14, zorder=5,
                  markeredgecolor='white', markeredgewidth=0.5)

    ax_a.set_xlabel(r'Mutation probability  $\mu$')
    ax_a.set_ylabel(r'Seed-to-seed variance  var($\rho_b$)')
    ax_a.legend(fontsize=FONT_LEGEND + 1)
    ax_a.set_title('Variance peak shifts with system size')
    panel_label(ax_a, 'a')

    # (b) μ_c(L) vs 1/L
    Ls = np.array(sorted(peaks))
    mu_cs = np.array([peaks[L][0] for L in Ls])
    inv_L = 1.0 / Ls

    colors_by_L = {64: C_L64, 128: C_L128, 256: C_L256}
    markers_by_L = {64: 'D', 128: 'o', 256: 's'}

    for L, iL, mc in zip(Ls, inv_L, mu_cs):
        ax_b.plot(iL, mc, markers_by_L[L], color=colors_by_L[L], ms=10,
                  zorder=4, markeredgecolor='white', markeredgewidth=0.8,
                  label=f'L = {L}')

    # Linear fit and extrapolation to 1/L -> 0
    coeffs = np.polyfit(inv_L, mu_cs, 1)
    x_fit = np.linspace(0, inv_L.max() * 1.15, 200)
    ax_b.plot(x_fit, np.polyval(coeffs, x_fit), '--', color='gray', lw=1.2,
              alpha=0.6, zorder=2)

    mu_inf = coeffs[1]
    ax_b.plot(0, mu_inf, 'o', color='black', ms=9, zorder=5,
              markerfacecolor='white', markeredgewidth=1.5)
    ax_b.annotate(rf'$\mu_c(\infty) \approx {mu_inf:.4f}$',
                  xy=(0, mu_inf), xytext=(30, -20), textcoords='offset points',
                  fontsize=FONT_ANNOT + 1,
                  arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax_b.set_xlabel(r'Inverse system size  $1/L$')
    ax_b.set_ylabel(r'Pseudocritical point  $\mu_c(L)$')
    ax_b.legend(fontsize=FONT_LEGEND + 1, loc='upper right')
    ax_b.set_title(r'Pseudocritical drift: $\mu_c(L) \to \mu_c(\infty)$')
    ax_b.set_xlim(-0.001, inv_L.max() * 1.25)
    panel_label(ax_b, 'b')

    save(fig, 'fig_pseudocritical')

    print(f'\n  Linear extrapolation: μ_c(∞) ≈ {mu_inf:.5f}')
    print(f'  Slope: {coeffs[0]:.4f}')


if __name__ == '__main__':
    main()
