#!/usr/bin/env python3
"""
fdom.py — topology-independent order parameter f_dom = max(n_R, n_G, n_B) / N.

Computes f_dom from node snapshot CSVs (second-half time average), caches
results separately from the main figures.py cache, and generates fig10.

Usage:
    venv/bin/python3 analysis/fdom.py [--recompute]
"""

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).parent.parent
SNAP     = ROOT / 'snapshots'
FIGDIR   = ROOT / 'paper'
CACHEDIR = ROOT / 'paper' / 'cache'

# ── style (mirrors figures.py) ───────────────────────────────────────────────

C_RAND = '#2166ac'
C_DYN  = '#b2182b'

FONT_TITLE  = 11
FONT_LABEL  = 11
FONT_TICK   = 9
FONT_LEGEND = 8.5
FONT_ANNOT  = 8

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         FONT_TICK,
    'axes.labelsize':    FONT_LABEL,
    'axes.titlesize':    FONT_TITLE,
    'legend.fontsize':   FONT_LEGEND,
    'xtick.labelsize':   FONT_TICK,
    'ytick.labelsize':   FONT_TICK,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.15,
    'grid.linewidth':    0.4,
    'figure.dpi':        200,
})


def save(fig, name):
    path = FIGDIR / f'{name}.pdf'
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path.name}  +  .png')


# ── f_dom computation ────────────────────────────────────────────────────────

def compute_fdom_fast(node_path):
    """Compute f_dom = max(n_R, n_G, n_B) / N from a single node CSV.

    Only reads the 'dominant' column (7th field) — avoids loading full arrays.
    """
    counts = {'R': 0, 'G': 0, 'B': 0}
    total = 0
    with open(node_path) as f:
        f.readline()  # skip header
        for line in f:
            # dominant is the 7th column (index 6), degree is 8th
            dom = line.rstrip().rsplit(',', 2)[1]
            counts[dom] = counts.get(dom, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    return max(counts.values()) / total


def seed_fdom(seed_dir, second_half=True):
    """Time-averaged f_dom over (second half of) snapshots for one seed."""
    snapshots = sorted(seed_dir.glob('snapshot_*_nodes.csv'))
    if not snapshots:
        return None
    if second_half:
        snapshots = snapshots[len(snapshots) // 2:]
    fdoms = [compute_fdom_fast(p) for p in snapshots]
    return {'fdom_mean': np.mean(fdoms), 'fdom_std': np.std(fdoms)}


def parse_seed_dir_name(name):
    """Parse seed directory name — same logic as figures.py."""
    ordered = '_ordered' in name
    name = name.replace('_ordered', '')
    m = re.match(r'mr([0-9.]+)_rm(\d+)_s(\d+)', name)
    if not m:
        return None
    mu = float(m.group(1))
    if mu > 1.0:
        mu = 1.0 / mu
    return {'mu': mu, 'rm': int(m.group(2)), 'seed': int(m.group(3)),
            'ordered': ordered}


# ── caching ──────────────────────────────────────────────────────────────────

FDOM_FIELDS = ['mu', 'seed', 'ordered', 'fdom_mean', 'fdom_std']


def save_fdom_cache(name, records):
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    path = CACHEDIR / f'fdom_{name}.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FDOM_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)
    print(f'  cached: {path.name} ({len(records)} rows)')


def load_fdom_cache(name):
    path = CACHEDIR / f'fdom_{name}.csv'
    if not path.exists():
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            converted = {}
            for k, v in row.items():
                if v in ('True', 'False'):
                    converted[k] = v == 'True'
                else:
                    try:
                        converted[k] = int(v)
                    except ValueError:
                        try:
                            converted[k] = float(v)
                        except ValueError:
                            converted[k] = v
                rows.append(converted)
    print(f'  loaded cache: fdom_{name}.csv ({len(rows)} rows)')
    return rows


def collect_fdom(snap_path):
    """Compute f_dom for every seed directory under snap_path."""
    results = []
    for seed_dir in sorted(snap_path.iterdir()):
        if not seed_dir.is_dir():
            continue
        info = parse_seed_dir_name(seed_dir.name)
        if info is None:
            continue
        summary = seed_fdom(seed_dir)
        if summary is None:
            continue
        info.update(summary)
        results.append(info)
    return results


def cached_fdom(name, snap_path, recompute=False):
    if not recompute:
        cached = load_fdom_cache(name)
        if cached is not None:
            return cached
    print(f'  computing fdom for {name} from {snap_path}...')
    data = collect_fdom(snap_path)
    save_fdom_cache(name, data)
    return data


# ── grouping helper ──────────────────────────────────────────────────────────

def group_by_mu(results):
    groups = defaultdict(list)
    for r in results:
        groups[round(r['mu'], 6)].append(r)
    return sorted(groups.items())


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10: f_dom vs μ  (adaptive network + fixed lattice comparison)
# ══════════════════════════════════════════════════════════════════════════════

def fig10_fdom_phase(recompute=False):
    print('Figure 10: f_dom phase diagram...')

    # Adaptive network (combine coarse + fine-probe, same as fig3)
    dyn_int  = cached_fdom('dyn_int',  SNAP / '3_dyn_graph_phase',   recompute)
    dyn_frac = cached_fdom('dyn_frac', SNAP / '4_first_order_probe', recompute)
    dyn_data = dyn_int + dyn_frac

    # Fixed lattice
    fix_data = cached_fdom('fixed', SNAP / '1_fixed_lattice_phase', recompute)

    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)

    # --- plot adaptive network ---
    dyn_groups = group_by_mu(dyn_data)
    dyn_mus, dyn_means, dyn_errs = [], [], []
    for mu, seeds in dyn_groups:
        vals = [s['fdom_mean'] for s in seeds]
        dyn_mus.append(mu)
        dyn_means.append(np.mean(vals))
        dyn_errs.append(np.std(vals))

    ax.errorbar(dyn_mus, dyn_means, yerr=dyn_errs, fmt='o-', color=C_DYN,
                lw=1.5, ms=5, capsize=4, capthick=1,
                label='Adaptive network', zorder=3)

    # --- plot fixed lattice ---
    fix_groups = group_by_mu(fix_data)
    fix_mus, fix_means, fix_errs = [], [], []
    for mu, seeds in fix_groups:
        vals = [s['fdom_mean'] for s in seeds]
        fix_mus.append(mu)
        fix_means.append(np.mean(vals))
        fix_errs.append(np.std(vals))

    ax.errorbar(fix_mus, fix_means, yerr=fix_errs, fmt='s-', color=C_RAND,
                lw=1.2, ms=4, capsize=3, capthick=0.8,
                label='Fixed lattice', alpha=0.8)

    # Reference lines
    ax.axhline(1/3, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.text(0.01, 1/3 + 0.01, r'$f_{\rm dom} = 1/3$ (disordered)',
            fontsize=FONT_ANNOT, color='gray', va='bottom')
    ax.axvline(0.35, color='gray', ls='--', lw=0.8, alpha=0.4)

    ax.set_xlabel(r'Mutation probability  $\mu$')
    ax.set_ylabel(r'Dominant fraction  $f_{\rm dom} = \max(n_R, n_G, n_B) / N$')
    ax.set_ylim(0.30, 1.02)
    ax.set_xlim(0.0, 0.70)
    ax.legend(fontsize=FONT_LEGEND + 1, loc='upper right')
    ax.set_title('Topology-independent order parameter\n'
                 r'$128\times128$, $r_{\min}=4$, second-half time average')

    save(fig, 'fig10_fdom_phase')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute f_dom order parameter and generate fig10.')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute from snapshots (ignore cache)')
    args = parser.parse_args()

    FIGDIR.mkdir(exist_ok=True)
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    print(f'Output: {FIGDIR}/')
    if not args.recompute:
        print(f'Cache:  {CACHEDIR}/  (use --recompute to rebuild)')
    print()

    fig10_fdom_phase(recompute=args.recompute)

    print(f'\nDone. Figure written to {FIGDIR}/')


if __name__ == '__main__':
    main()
