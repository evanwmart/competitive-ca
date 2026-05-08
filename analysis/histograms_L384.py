#!/usr/bin/env python3
"""
histograms_L384.py — per-seed ρ_b histograms at L=384 across μ_c.

Reads results/binder_L384_merged_seeds.csv (64 seeds × 16 μ values).
Produces a 5-panel histogram strip at μ near the Binder minimum to
show whether the L=128 bimodality (Fig 4) persists at the largest size.

Usage:
    venv/bin/python3 analysis/histograms_L384.py
"""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

import sys; sys.path.insert(0, str(Path(__file__).parent))
from figures import C_DYN, FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_ANNOT, save, plt

ROOT = Path(__file__).parent.parent
SEEDS = ROOT / 'results' / 'binder_L384_merged_seeds.csv'

MU_VALS = [0.345, 0.347, 0.348, 0.349, 0.351]
PHASE_LABELS = [
    'ordered',
    'near $\\mu_c$',
    '$\\approx \\mu_c$',
    'near $\\mu_c$',
    'disordered',
]

XLIM = (0.0, 0.65)
NBINS = 18


def load_bd_by_mu():
    data = defaultdict(list)
    with open(SEEDS) as f:
        for row in csv.DictReader(f):
            data[float(row['mutation_prob'])].append(float(row['bd_mean']))
    return data


def main():
    data = load_bd_by_mu()

    fig, axes = plt.subplots(1, 5, figsize=(12, 2.6), sharey=True)
    bins = np.linspace(XLIM[0], XLIM[1], NBINS + 1)

    for ax, mu, phase in zip(axes, MU_VALS, PHASE_LABELS):
        vals = np.array(data[mu])
        n = len(vals)
        mean = vals.mean()

        ax.hist(vals, bins=bins, color=C_DYN, alpha=0.75, edgecolor='white',
                linewidth=0.5)
        ax.axvline(mean, color='k', ls='--', lw=1.0, label=f'mean = {mean:.3f}')

        ax.set_xlim(XLIM)
        ax.set_title(f'$\\mu = {mu}$', fontsize=FONT_TITLE)
        ax.set_xlabel('$\\rho_b$', fontsize=FONT_LABEL)

        if mean > 0.3:
            tx, ha = 0.05, 'left'
        else:
            tx, ha = 0.95, 'right'
        ax.text(tx, 0.95, f'$n = {n}$\n{phase}',
                transform=ax.transAxes, ha=ha, va='top',
                fontsize=FONT_ANNOT,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8,
                          edgecolor='0.7', lw=0.5))

        ax.tick_params(labelsize=FONT_TICK)

    axes[0].set_ylabel('Count (seeds)', fontsize=FONT_LABEL)

    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig_hist_L384')


if __name__ == '__main__':
    main()
