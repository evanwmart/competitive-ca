#!/usr/bin/env python3
"""
degree_order.py — per-node degree vs local order correlation across the phase transition.

For each snapshot, computes per-node local order (fraction of compatible edges)
and the Pearson correlation between node degree and local order.

Usage:
    venv/bin/python3 analysis/degree_order.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from figures import (C_DYN, C_RAND, FONT_TITLE, FONT_LABEL, FONT_TICK,
                     FONT_LEGEND, FONT_ANNOT, FIGDIR, save, plt, panel_label,
                     C_PHASE_ORD, C_PHASE_DIS)

ROOT = Path(__file__).parent.parent
SNAP = ROOT / 'snapshots'

MAX_SEEDS = 8  # sample up to this many seeds per mu


def parse_seed_dir_name(name):
    """Parse seed directory name, return dict with mu, rm, seed."""
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


def last_snapshot(seed_dir):
    """Return (node_path, edge_path) for the last snapshot in a seed dir."""
    node_files = sorted(seed_dir.glob('snapshot_*_nodes.csv'))
    if not node_files:
        return None, None
    npath = node_files[-1]
    frame = npath.name.replace('_nodes.csv', '').replace('snapshot_', '')
    epath = npath.parent / f'snapshot_{frame}_edges.csv'
    if not epath.exists():
        return None, None
    return npath, epath


def compute_node_local_order(node_path, edge_path):
    """Compute per-node degree and local order from snapshot CSVs.

    Returns (degrees, local_orders) as numpy arrays indexed by node id.
    Local order = fraction of a node's edges that are compatible (rel != 'N').
    """
    # Read node degrees
    degrees = {}
    with open(node_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip().split(',')
            node_id = int(parts[0])
            deg = int(parts[7])
            degrees[node_id] = deg

    n_nodes = max(degrees.keys()) + 1
    deg_arr = np.zeros(n_nodes, dtype=int)
    compatible_count = np.zeros(n_nodes, dtype=int)
    total_count = np.zeros(n_nodes, dtype=int)

    for nid, d in degrees.items():
        deg_arr[nid] = d

    # Read edges and count compatible per node
    with open(edge_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip().split(',')
            src = int(parts[0])
            dst = int(parts[1])
            rel = parts[2]
            # Each edge is listed once; count for both endpoints
            total_count[src] += 1
            total_count[dst] += 1
            if rel != 'N':
                compatible_count[src] += 1
                compatible_count[dst] += 1

    # Local order = compatible / total (avoid div-by-zero for isolated nodes)
    mask = total_count > 0
    local_order = np.zeros(n_nodes)
    local_order[mask] = compatible_count[mask] / total_count[mask]

    return deg_arr, local_order, mask


def degree_order_correlation(node_path, edge_path):
    """Compute Pearson r between degree and local order for one snapshot."""
    deg, lo, mask = compute_node_local_order(node_path, edge_path)
    # Only include nodes with at least one edge
    d = deg[mask]
    o = lo[mask]
    if len(d) < 10 or np.std(d) == 0 or np.std(o) == 0:
        return np.nan, d, o
    r, _ = stats.pearsonr(d, o)
    return r, d, o


def collect_degree_order(snap_path, max_seeds=MAX_SEEDS):
    """Collect degree-order correlations for all mu values in an experiment."""
    # Group seed dirs by mu
    mu_dirs = defaultdict(list)
    for seed_dir in sorted(snap_path.iterdir()):
        if not seed_dir.is_dir():
            continue
        info = parse_seed_dir_name(seed_dir.name)
        if info is None:
            continue
        mu_dirs[round(info['mu'], 6)].append(seed_dir)

    results = []
    for mu in sorted(mu_dirs.keys()):
        dirs = mu_dirs[mu][:max_seeds]
        corrs = []
        for sd in dirs:
            npath, epath = last_snapshot(sd)
            if npath is None:
                continue
            r, _, _ = degree_order_correlation(npath, epath)
            if not np.isnan(r):
                corrs.append(r)
        if corrs:
            results.append({
                'mu': mu,
                'r_mean': np.mean(corrs),
                'r_std': np.std(corrs),
                'r_err': np.std(corrs) / np.sqrt(len(corrs)),
                'n_seeds': len(corrs),
            })
            print(f'  mu={mu:.3f}: r={np.mean(corrs):+.3f} +/- {np.std(corrs):.3f} '
                  f'({len(corrs)} seeds)')
    return results


def get_scatter_data(snap_path, target_mu, seed_idx=0):
    """Get degree and local order arrays for one seed at a given mu."""
    for seed_dir in sorted(snap_path.iterdir()):
        if not seed_dir.is_dir():
            continue
        info = parse_seed_dir_name(seed_dir.name)
        if info is None:
            continue
        if abs(info['mu'] - target_mu) < 0.005 and info['seed'] == seed_idx:
            npath, epath = last_snapshot(seed_dir)
            if npath is None:
                return None, None, None
            r, d, o = degree_order_correlation(npath, epath)
            return r, d, o
    return None, None, None


def main():
    FIGDIR.mkdir(exist_ok=True)

    # ── Adaptive network (exp 4) ────────────────────────────────────────────
    print('Degree-order correlation: adaptive network (exp 4)...')
    dyn_results = collect_degree_order(SNAP / '4_first_order_probe')

    # ── Fixed lattice (exp 1) ───────────────────────────────────────────────
    print('\nDegree-order correlation: fixed lattice (exp 1)...')
    fix_results = collect_degree_order(SNAP / '1_fixed_lattice_phase')

    # ── Scatter data at mu=0.35 ─────────────────────────────────────────────
    # Try seeds to find one in the ordered phase for a clear scatter
    print('\nGetting scatter data at mu=0.35...')
    scatter_r, scatter_d, scatter_o = None, None, None
    best_r, best_data = -1.0, (None, None, None)
    for seed_try in range(32):
        r, d, o = get_scatter_data(SNAP / '4_first_order_probe', 0.35, seed_try)
        if r is not None and r > best_r:
            best_r = r
            best_data = (r, d, o)
            if r > 0.15:
                break
    scatter_r, scatter_d, scatter_o = best_data
    print(f'  Best seed: r={scatter_r:.3f}' if scatter_r else '  No data found')

    # ── Figure ──────────────────────────────────────────────────────────────
    print('\nGenerating figure...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), tight_layout=True)

    # (a) Correlation vs mu
    if dyn_results:
        mus = [r['mu'] for r in dyn_results]
        means = [r['r_mean'] for r in dyn_results]
        errs = [r['r_err'] for r in dyn_results]
        ax1.errorbar(mus, means, yerr=errs, fmt='o-', color=C_DYN,
                     lw=1.5, ms=5, capsize=4, capthick=1,
                     label='Adaptive network', zorder=3)

    if fix_results:
        mus_f = [r['mu'] for r in fix_results]
        means_f = [r['r_mean'] for r in fix_results]
        errs_f = [r['r_err'] for r in fix_results]
        ax1.errorbar(mus_f, means_f, yerr=errs_f, fmt='s--', color=C_RAND,
                     lw=1.2, ms=4, capsize=3, capthick=0.8,
                     label='Fixed lattice ($k=4$)', alpha=0.7, zorder=2)
    else:
        # Fixed lattice: degree is constant (k=4), so Pearson r is undefined.
        # Add annotation explaining this.
        ax1.annotate('Fixed lattice: $r$ undefined\n(degree $\\equiv 4$ for all nodes)',
                     xy=(0.20, 0.0), xytext=(0.20, -0.10),
                     fontsize=FONT_ANNOT, color=C_RAND, ha='center',
                     arrowprops=dict(arrowstyle='->', color=C_RAND, lw=0.6))

    ax1.axhline(0, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax1.axvline(0.35, color='gray', ls='--', lw=0.8, alpha=0.4)
    ax1.set_xlabel(r'Mutation probability  $\mu$')
    ax1.set_ylabel(r'Pearson $r$ (degree, local order)')
    ax1.legend(fontsize=FONT_LEGEND, loc='best')
    ax1.set_ylim(-0.15, 0.85)
    if dyn_results:
        ax1.set_xlim(min(mus) - 0.02, max(mus) + 0.02)
    panel_label(ax1, 'a')

    # (b) Scatter at mu=0.35
    if scatter_d is not None and scatter_o is not None:
        # Subsample for plotting clarity
        rng = np.random.default_rng(42)
        n = len(scatter_d)
        if n > 2000:
            idx = rng.choice(n, 2000, replace=False)
            plot_d, plot_o = scatter_d[idx], scatter_o[idx]
        else:
            plot_d, plot_o = scatter_d, scatter_o

        # Add jitter to degree for visibility
        jitter = rng.uniform(-0.15, 0.15, len(plot_d))
        ax2.scatter(plot_d + jitter, plot_o, s=6, alpha=0.25, color=C_DYN,
                    edgecolors='none', rasterized=True)

        # Binned means
        unique_degs = np.unique(scatter_d)
        bin_means = []
        bin_degs = []
        for k in unique_degs:
            vals = scatter_o[scatter_d == k]
            if len(vals) >= 5:
                bin_degs.append(k)
                bin_means.append(np.mean(vals))
        if bin_degs:
            ax2.plot(bin_degs, bin_means, 'o-', color='black', lw=1.8,
                     ms=5, zorder=4, label='Binned mean')

        ax2.set_xlabel(r'Node degree  $k_i$')
        ax2.set_ylabel(r'Local order  $1 - \rho_{b,i}$')
        ax2.legend(fontsize=FONT_LEGEND, loc='best')
        ax2.text(0.97, 0.05, f'$r = {scatter_r:.2f}$',
                 transform=ax2.transAxes, ha='right', fontsize=FONT_ANNOT + 1,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray',
                           alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No scatter data available',
                 transform=ax2.transAxes, ha='center')

    panel_label(ax2, 'b')
    ax2.set_title(r'$\mu = 0.35$, adaptive network', fontsize=FONT_TITLE)
    ax1.set_title('Degree–local order correlation', fontsize=FONT_TITLE)

    save(fig, 'fig_degree_order')
    print(f'\nDone. Figure saved to {FIGDIR}/fig_degree_order.pdf')


if __name__ == '__main__':
    main()
