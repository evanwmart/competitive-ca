#!/usr/bin/env python3
"""
binder.py — compute Binder cumulants from cached per-seed boundary density data.

Binder cumulant:  U_L = 1 - <rho_b^4> / (3 <rho_b^2>^2)

For a first-order transition, U_L develops a negative dip that deepens with L.

Usage:
    venv/bin/python3 analysis/binder.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figures import (C_L64, C_L128, C_L256,
                     FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_ANNOT,
                     FIGDIR, CACHEDIR, SNAP, save, plt,
                     cached_experiment, load_sweep_csv)

# ── colour / marker mapping ────────────────────────────────────────────────

C_L384 = '#e6ab02'  # yellow-brown
C_L512 = '#666666'  # gray

SIZE_STYLE = {
    64:  dict(color=C_L64,  marker='o', label='$L=64$'),
    128: dict(color=C_L128, marker='s', label='$L=128$'),
    256: dict(color=C_L256, marker='^', label='$L=256$'),
    384: dict(color=C_L384, marker='D', label='$L=384$'),
    512: dict(color=C_L512, marker='v', label='$L=512$'),
}

ROOT = Path(__file__).parent.parent
RESULTSDIR = ROOT / 'results'

# ── helpers ────────────────────────────────────────────────────────────────

def binder_cumulant(rho):
    """Compute U_L = 1 - <rho^4> / (3 <rho^2>^2) from an array of per-seed values."""
    rho = np.asarray(rho, dtype=float)
    m2 = np.mean(rho ** 2)
    m4 = np.mean(rho ** 4)
    if m2 == 0:
        return np.nan
    return 1.0 - m4 / (3.0 * m2 ** 2)


def load_csv(name):
    """Load a cache CSV, return list of dicts with floats for numeric columns."""
    path = CACHEDIR / name
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_seeds_csv(path):
    """Load a per-seed sweep CSV (from --save-seeds). Returns list of dicts."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def group_bd_by_mu(rows):
    """Group bd_mean values by mu. Returns dict {mu: [bd_mean, ...]}."""
    groups = defaultdict(list)
    for r in rows:
        # Support both 'mu' (cache format) and 'mutation_prob' (seeds CSV format)
        mu_key = 'mu' if 'mu' in r else 'mutation_prob'
        groups[float(r[mu_key])].append(float(r['bd_mean']))
    return groups


def binder_by_mu(rows):
    """Compute U_L per mu group. Returns sorted arrays (mus, U_Ls)."""
    groups = group_bd_by_mu(rows)
    items = sorted(groups.items())
    mus = np.array([mu for mu, _ in items])
    uls = np.array([binder_cumulant(vals) for _, vals in items])
    return mus, uls


# ── load sweep data ───────────────────────────────────────────────────────

# Fine grids (0.001 step): per-seed data from snapshots
binder_L128_fine = cached_experiment('binder_L128_fine',
                                     SNAP / 'B_binder_L128_fine')
binder_L256_fine = cached_experiment('binder_L256_fine',
                                     SNAP / 'C_binder_L256_fine')

sweep_data = {
    64:  load_csv('fig7_L64.csv'),
    128: binder_L128_fine,        # fine-grid replaces coarse
    256: binder_L256_fine,        # fine-grid replaces sparse
}

# Add L=384 and L=512 from --save-seeds CSVs if available
for L, pattern in [(384, 'sweep_dyn_384x384_*_seeds.csv'),
                    (512, 'sweep_dyn_512x512_*_seeds.csv')]:
    matches = sorted(RESULTSDIR.glob(pattern))
    if matches:
        sweep_data[L] = load_seeds_csv(matches[-1])
        print(f'  loaded seeds: {matches[-1].name} ({len(sweep_data[L])} rows)')

# ── load FSS data at mu=0.35 ──────────────────────────────────────────────

fss_data = {}
for L in [64, 128, 256]:
    combined = []
    for ic in ['rand', 'ord']:
        fname = f'fig8_fss_L{L}_{ic}.csv'
        combined.extend(load_csv(fname))
    fss_data[L] = combined

# ── compute ────────────────────────────────────────────────────────────────

# Sweep: U_L(mu) for each L
sweep_results = {}
for L, rows in sweep_data.items():
    sweep_results[L] = binder_by_mu(rows)

# FSS: U_L at mu=0.35 for each L
fss_results = {}
for L, rows in fss_data.items():
    rho = np.array([float(r['bd_mean']) for r in rows])
    fss_results[L] = binder_cumulant(rho)

# ── print summary ─────────────────────────────────────────────────────────

all_sizes = sorted(sweep_results.keys())

print('\n=== Binder cumulant U_L  sweep ===')
print(f'{"mu":>8s}', end='')
for L in all_sizes:
    print(f'  {"L="+str(L):>10s}', end='')
print()

all_mus = sorted(set().union(*(sweep_results[L][0].tolist() for L in all_sizes)))
for mu in all_mus:
    print(f'{mu:8.4f}', end='')
    for L in all_sizes:
        mus, uls = sweep_results[L]
        idx = np.where(np.isclose(mus, mu))[0]
        if len(idx):
            print(f'  {uls[idx[0]]:10.6f}', end='')
        else:
            print(f'  {"---":>10s}', end='')
    print()

print('\n=== Binder minimum per size ===')
for L in all_sizes:
    mus, uls = sweep_results[L]
    i = np.argmin(uls)
    print(f'  L={L:>3d}:  U_min = {uls[i]:.4f}  at mu* = {mus[i]:.3f}')

print('\n=== Binder cumulant U_L  at mu=0.35 (FSS, both ICs) ===')
for L in [64, 128, 256]:
    if L in fss_data:
        n = len(fss_data[L])
        print(f'  L={L:>3d}:  U_L = {fss_results[L]:.6f}   (n_seeds = {n})')

# ── figure 1: sweep ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4.5, 3.2))

for L in all_sizes:
    mus, uls = sweep_results[L]
    sty = SIZE_STYLE[L]
    ax.plot(mus, uls, marker=sty['marker'], color=sty['color'],
            label=sty['label'], linewidth=1.2, markersize=4.5, zorder=3)

ax.axhline(2/3, color='0.6', ls=':', lw=0.8, zorder=1)
ax.text(0.50, 2/3 + 0.005, '$2/3$', fontsize=FONT_ANNOT, color='0.4',
        ha='right', va='bottom')

ax.set_xlabel(r'Mutation probability $\mu$', fontsize=FONT_LABEL)
ax.set_ylabel(r'Binder cumulant $U_L$', fontsize=FONT_LABEL)
ax.set_title('Binder cumulant vs mutation rate', fontsize=FONT_TITLE)
ax.legend(fontsize=FONT_LEGEND, frameon=True, fancybox=False, edgecolor='0.7')
ax.tick_params(labelsize=FONT_TICK)

fig.tight_layout()
save(fig, 'fig_binder_sweep')

# ── figure 2: FSS bar chart at mu=0.35 ────────────────────────────────────

fig, ax = plt.subplots(figsize=(3.5, 3.0))

Ls = [64, 128, 256]
ul_vals = [fss_results[L] for L in Ls]
colors  = [SIZE_STYLE[L]['color'] for L in Ls]
labels  = [f'$L={L}$' for L in Ls]

bars = ax.bar(range(len(Ls)), ul_vals, color=colors, edgecolor='0.3',
              linewidth=0.6, width=0.55, zorder=3)

# annotate values on bars
for i, (b, v) in enumerate(zip(bars, ul_vals)):
    yoff = -0.015 if v < 0 else 0.008
    va   = 'top'  if v < 0 else 'bottom'
    ax.text(b.get_x() + b.get_width() / 2, v + yoff, f'{v:.4f}',
            ha='center', va=va, fontsize=FONT_ANNOT, color='0.15')

ax.axhline(0, color='0.5', ls='-', lw=0.5)
ax.axhline(2/3, color='0.6', ls=':', lw=0.8, zorder=1)
ax.text(len(Ls) - 0.6, 2/3 + 0.005, '$2/3$', fontsize=FONT_ANNOT,
        color='0.4', ha='right', va='bottom')

ax.set_xticks(range(len(Ls)))
ax.set_xticklabels(labels, fontsize=FONT_TICK)
ax.set_ylabel(r'Binder cumulant $U_L$', fontsize=FONT_LABEL)
ax.set_title(r'$U_L$ at $\mu = 0.35$ (both ICs)', fontsize=FONT_TITLE)
ax.tick_params(axis='y', labelsize=FONT_TICK)

fig.tight_layout()
save(fig, 'fig_binder_fss')

print('\nDone.')
