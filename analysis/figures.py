#!/usr/bin/env python3
"""
figures.py — produce all paper figures from snapshot data.

Reads CSVs from snapshots/ directory. Each seed directory contains
snapshot_NNNNNN_nodes.csv and snapshot_NNNNNN_edges.csv at every 500 frames.

Usage:
    venv/bin/python3 analysis/figures.py
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT    = Path(__file__).parent.parent
SNAP    = ROOT / 'snapshots'
FIGDIR  = ROOT / 'paper'
CACHEDIR = ROOT / 'paper' / 'cache'

# ── notation ─────────────────────────────────────────────────────────────────
#
#   ρ_b   boundary density (fraction of null/incompatible edges)
#   μ     mutation probability
#   ⟨k⟩   mean node degree
#   τ     domain-size power-law exponent
#   L     system linear size (L×L lattice)
#
# ── consistent palette ───────────────────────────────────────────────────────
#
# Data identity (what system/condition produced it):
#   C_RAND  = blue    — random initial conditions / fixed lattice
#   C_ORDI  = orange  — ordered initial conditions
#   C_DYN   = red     — adaptive network (when contrasted with fixed)
#
# Phase identity (what basin the data landed in):
#   C_PHASE_ORD = green  — ordered phase
#   C_PHASE_MIX = orange — coexistence / bistable
#   C_PHASE_DIS = red    — disordered phase
#
# Size identity:
#   C_L64  = green, C_L128 = blue, C_L256 = red
#

C_RAND  = '#2166ac'
C_ORDI  = '#e08214'
C_DYN   = '#b2182b'
C_DEG   = '#74a9cf'  # degree (lighter, secondary)

C_PHASE_ORD = '#4dac26'
C_PHASE_MIX = '#e08214'
C_PHASE_DIS = '#d7191c'

C_L64  = '#4dac26'
C_L128 = '#2166ac'
C_L256 = '#b2182b'

# ── unified style ────────────────────────────────────────────────────────────

FONT_TITLE = 11
FONT_LABEL = 11
FONT_TICK  = 9
FONT_LEGEND = 8.5
FONT_ANNOT = 8

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

def panel_label(ax, label, x=-0.08, y=1.05):
    """Add (a), (b), etc. label to a panel."""
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='bottom')

# ── snapshot loading ─────────────────────────────────────────────────────────

def compute_bd_fast(edge_path):
    """Boundary density = fraction of null edges."""
    with open(edge_path) as f:
        f.readline()
        n_total = 0
        n_null = 0
        for line in f:
            n_total += 1
            if line.rstrip().endswith(',N'):
                n_null += 1
    return n_null / n_total if n_total > 0 else 0.0


def compute_deg_fast(node_path):
    """Mean degree from nodes CSV (last field)."""
    total = 0
    count = 0
    with open(node_path) as f:
        f.readline()
        for line in f:
            total += int(line.rstrip().rsplit(',', 1)[1])
            count += 1
    return total / count if count > 0 else 0.0


def seed_summary(seed_dir, second_half=True):
    """Compute time-averaged bd and degree for one seed."""
    snapshots = sorted(seed_dir.glob('snapshot_*_nodes.csv'))
    if not snapshots:
        return None
    if second_half:
        snapshots = snapshots[len(snapshots) // 2:]
    bds, degs = [], []
    for npath in snapshots:
        frame = npath.name.replace('_nodes.csv', '').replace('snapshot_', '')
        epath = npath.parent / f'snapshot_{frame}_edges.csv'
        bds.append(compute_bd_fast(epath))
        degs.append(compute_deg_fast(npath))
    return {
        'bd_mean': np.mean(bds), 'bd_std': np.std(bds),
        'deg_mean': np.mean(degs), 'deg_std': np.std(degs),
    }


def parse_seed_dir_name(name):
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


def collect_experiment(snap_path):
    results = []
    for seed_dir in sorted(snap_path.iterdir()):
        if not seed_dir.is_dir():
            continue
        info = parse_seed_dir_name(seed_dir.name)
        if info is None:
            continue
        summary = seed_summary(seed_dir)
        if summary is None:
            continue
        info.update(summary)
        results.append(info)
    return results


def group_by_mu(results):
    groups = defaultdict(list)
    for r in results:
        groups[round(r['mu'], 6)].append(r)
    return sorted(groups.items())


# ── caching ──────────────────────────────────────────────────────────────────
#
# Each figure saves its per-seed data to paper/cache/<name>.csv on first run.
# Subsequent runs load from cache unless --recompute is passed.

def save_cache(name, records, fields):
    """Save list of dicts to CSV cache."""
    CACHEDIR.mkdir(parents=True, exist_ok=True)
    path = CACHEDIR / f'{name}.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)
    print(f'  cached: {path.name} ({len(records)} rows)')


def load_cache(name):
    """Load cached CSV as list of dicts with numeric conversion."""
    path = CACHEDIR / f'{name}.csv'
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
    print(f'  loaded cache: {path.name} ({len(rows)} rows)')
    return rows


SEED_FIELDS = ['mu', 'rm', 'seed', 'ordered', 'bd_mean', 'bd_std', 'deg_mean', 'deg_std']
RECOMPUTE = False  # set via --recompute flag
RESULTSDIR = ROOT / 'results'


def load_sweep_csv(path):
    """Load an aggregate sweep CSV (from sweep.py) as list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = int(v)
                except ValueError:
                    try:
                        converted[k] = float(v)
                    except ValueError:
                        converted[k] = v
            rows.append(converted)
    return rows


def cached_experiment(name, snap_path):
    """Load experiment data from cache, or compute and cache it."""
    if not RECOMPUTE:
        cached = load_cache(name)
        if cached is not None:
            return cached
    print(f'  computing {name} from {snap_path}...')
    data = collect_experiment(snap_path)
    save_cache(name, data, SEED_FIELDS)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Fixed lattice phase diagram
# ══════════════════════════════════════════════════════════════════════════════

def fig1_fixed_phase():
    print('Figure 1: fixed lattice phase diagram...')
    data = cached_experiment('fig1_fixed', SNAP / '1_fixed_lattice_phase')
    groups = group_by_mu(data)

    mus, bd_means, bd_errs, bd_vars = [], [], [], []
    for mu, seeds in groups:
        bds = [s['bd_mean'] for s in seeds]
        mus.append(mu)
        bd_means.append(np.mean(bds))
        bd_errs.append(np.std(bds) / np.sqrt(len(bds)))
        bd_vars.append(np.var(bds))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6.5), tight_layout=True,
        gridspec_kw={'height_ratios': [1, 1.3]})

    # (a) boundary density
    ax1.errorbar(mus, bd_means, yerr=bd_errs, fmt='o-', color=C_RAND,
                 lw=1.5, ms=5, capsize=4, capthick=1)
    ax1.set_ylabel(r'Boundary density  $\rho_b$')
    ax1.set_xlim(0.08, 0.38)
    ax1.set_xlabel(r'Mutation probability  $\mu$')
    panel_label(ax1, 'a')

    # (b) variance
    bd_vars_arr = np.array(bd_vars)
    ax2.semilogy(mus, bd_vars_arr + 1e-10, 'o-', color=C_RAND,
                 lw=1.5, ms=5)

    ax2.set_ylabel(r'Seed-to-seed variance  var($\rho_b$)')
    ax2.set_xlabel(r'Mutation probability  $\mu$')
    ax2.set_xlim(0.08, 0.38)
    panel_label(ax2, 'b')

    n_seeds = len(groups[0][1]) if groups else '?'
    fig.suptitle('Fixed 4-regular toroidal lattice\n'
                 rf'$128\times128$, $r_{{\min}}=4$, {n_seeds} seeds, 50k frames',
                 fontsize=FONT_TITLE)

    save(fig, 'fig1_fixed_phase')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: τ(L) finite-size scaling
# ══════════════════════════════════════════════════════════════════════════════

def fig2_tau():
    print('Figure 2: domain-size exponent τ(L)...')
    logdir = ROOT / 'logs'
    Ls, taus, r2s = [], [], []
    for logname in ['2_tau_L64.log', '2_tau_L128.log', '2_tau_L256.log', '2_tau_L512.log']:
        path = logdir / logname
        if not path.exists():
            continue
        text = path.read_text()
        for line in text.splitlines():
            if re.search(r'mr=\s*5\b', line):
                m_tau = re.search(r'[τt](?:au)?=([0-9.]+)', line)
                m_r2 = re.search(r'R[²2]=([0-9.]+)', line)
                if m_tau and m_r2:
                    m_L = re.search(r'L(\d+)', logname)
                    Ls.append(int(m_L.group(1)))
                    taus.append(float(m_tau.group(1)))
                    r2s.append(float(m_r2.group(1)))

    if len(Ls) < 2:
        print('  Not enough data, skipping')
        return

    Ls = np.array(Ls)
    taus = np.array(taus)
    r2s = np.array(r2s)
    inv_L = 1.0 / Ls

    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)

    ax.plot(inv_L, taus, 'o-', color=C_RAND, lw=1.5, ms=8, zorder=3,
            label=r'Measured $\tau$ at $\mu = 0.20$')

    # Annotations: pushed further from the data points
    offsets = {64: (18, 18), 128: (35, 20), 256: (18, 18), 512: (-60, -60)}
    for L, tau, r2 in zip(Ls, taus, r2s):
        ox, oy = offsets.get(L, (18, 12))
        label = f'L = {L},  R² = {r2:.3f}'
        ax.annotate(label, xy=(1/L, tau), xytext=(ox, oy),
                    textcoords='offset points', fontsize=FONT_ANNOT,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6))

    ax.axhline(2.0, color='#666', ls='--', lw=1.2,
               label=r'$\tau = 2.0$  (voter model)')
    tau_perc = 187 / 91
    ax.axhline(tau_perc, color='#aaa', ls=':', lw=1,
               label=r'$\tau = \frac{187}{91} \approx %.3f$  (2D percolation)' % tau_perc)

    # Linear extrapolation through L=64, 128, 256 only (exclude 512)
    coeffs = np.polyfit(inv_L[:3], taus[:3], 1)
    x_fit = np.linspace(0, inv_L[0] * 1.05, 200)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=C_RAND,
            lw=1, alpha=0.35,
            label=r'Linear fit (L=64–256) $\to \tau(\infty) \approx %.2f$' % coeffs[1])
    # Open circle at extrapolated point (more rigorous than star)
    ax.plot(0, coeffs[1], 'o', color=C_RAND, ms=8, zorder=4,
            markerfacecolor='white', markeredgewidth=1.5)

    ax.set_xlabel(r'Inverse system size  $1/L$')
    ax.set_ylabel(r'Domain-size exponent  $\tau$')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.set_xlim(-0.002, inv_L[0] * 1.4)
    ax.legend(fontsize=FONT_LEGEND, loc='upper right')
    ax.set_title(r'Fixed lattice: domain-size exponent $\tau(L)$ at criticality'
                 r'  ($\mu = 0.20$, $r_{\min} = 4$)')

    save(fig, 'fig2_tau')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Adaptive network phase diagram (bd + degree)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_dyn_phase():
    print('Figure 3: adaptive network phase diagram...')
    data_int = cached_experiment('fig3_dyn_int', SNAP / '3_dyn_graph_phase')
    data_frac = cached_experiment('fig3_dyn_frac', SNAP / '4_first_order_probe')
    data = data_int + data_frac
    groups = group_by_mu(data)

    mus, bd_means, bd_errs, deg_means, deg_errs = [], [], [], [], []
    for mu, seeds in groups:
        bds = [s['bd_mean'] for s in seeds]
        degs = [s['deg_mean'] for s in seeds]
        mus.append(mu)
        bd_means.append(np.mean(bds))
        bd_errs.append(np.std(bds) / np.sqrt(len(bds)))
        deg_means.append(np.mean(degs))
        deg_errs.append(np.std(degs) / np.sqrt(len(degs)))

    fig, ax1 = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)

    # bd is primary (bold), degree is secondary (lighter, thinner)
    l1 = ax1.errorbar(mus, bd_means, yerr=bd_errs, fmt='o-', color=C_DYN,
                       lw=1.8, ms=5, capsize=4, capthick=1,
                       label=r'Boundary density  $\rho_b$', zorder=3)
    l2 = ax2.errorbar(mus, deg_means, yerr=deg_errs, fmt='s-', color=C_DEG,
                       lw=1, ms=3.5, capsize=2, capthick=0.8,
                       label=r'Mean degree  $\langle k \rangle$', alpha=0.7)

    ax1.axvline(0.35, color='gray', ls='--', lw=0.8, alpha=0.4)

    ax1.set_xlabel(r'Mutation probability  $\mu$')
    ax1.set_ylabel(r'Boundary density  $\rho_b$', color=C_DYN)
    ax2.set_ylabel(r'Mean degree  $\langle k \rangle$', color=C_DEG)
    ax1.tick_params(axis='y', colors=C_DYN)
    ax2.tick_params(axis='y', colors=C_DEG)
    ax1.set_xlim(0.0, 0.70)

    ax1.legend(handles=[l1, l2], loc='center right', framealpha=0.9)
    n_seeds = len(groups[0][1]) if groups else '?'
    ax1.set_title(r'Adaptive network: coupled order parameters'
                  '\n'
                  rf'$128\times128$, $r_{{\min}}=4$, $k_{{\max}}=8$, {n_seeds} seeds, 30k frames')

    # --- Variance inset ---
    seed_vars = []
    for mu, seeds in groups:
        bds = [s['bd_mean'] for s in seeds]
        seed_vars.append(np.var(bds))
    ax_ins = ax1.inset_axes([0.52, 0.35, 0.38, 0.30])
    ax_ins.semilogy(mus, seed_vars, 'o-', color=C_DYN, lw=1, ms=3, zorder=3)
    ax_ins.set_xlabel(r'$\mu$', fontsize=FONT_ANNOT)
    ax_ins.set_ylabel(r'var($\rho_b$)', fontsize=FONT_ANNOT)
    ax_ins.tick_params(labelsize=FONT_ANNOT - 1)
    ax_ins.set_xlim(0.31, 0.52)
    ax_ins.set_title('Variance', fontsize=FONT_ANNOT)

    save(fig, 'fig3_dyn_phase')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Variance spike
# ══════════════════════════════════════════════════════════════════════════════

def fig4_variance():
    print('Figure 4: variance spike...')
    data = cached_experiment('fig3_dyn_frac', SNAP / '4_first_order_probe')
    groups = group_by_mu(data)

    mus, seed_vars = [], []
    for mu, seeds in groups:
        bds = [s['bd_mean'] for s in seeds]
        mus.append(mu)
        seed_vars.append(np.var(bds))

    fig, ax = plt.subplots(figsize=(7, 4.5), tight_layout=True)

    ax.semilogy(mus, seed_vars, 'o-', color=C_DYN, lw=1.5, ms=6, zorder=3)

    # Background variance band
    bg_vars = [v for m, v in zip(mus, seed_vars) if abs(m - 0.35) > 0.03]
    if bg_vars:
        bg_level = np.median(bg_vars)
        ax.axhspan(bg_level * 0.1, bg_level * 10, color='gray', alpha=0.06)
        ax.axhline(bg_level, color='gray', ls=':', lw=0.8, alpha=0.4)
        ax.text(0.47, bg_level * 0.3, 'Median background\nvariance (away from μ_c)',
                fontsize=FONT_ANNOT, color='gray', va='top')

    # Annotate peak
    peak_idx = np.argmax(seed_vars)
    ax.annotate(f'var = {seed_vars[peak_idx]:.1e}',
                xy=(mus[peak_idx], seed_vars[peak_idx]),
                xytext=(40, -25), textcoords='offset points', fontsize=FONT_ANNOT,
                arrowprops=dict(arrowstyle='->', color=C_DYN, lw=0.8))

    ax.set_xlabel(r'Mutation probability  $\mu$')
    ax.set_ylabel(r'Seed-to-seed variance  var($\rho_b$)')
    ax.set_xlim(0.31, 0.52)
    n_seeds = len(groups[0][1]) if groups else '?'
    ax.set_title(r'Adaptive network: variance spike at the discontinuous transition'
                 '\n'
                 rf'$128\times128$, {n_seeds} seeds, 30k frames')

    save(fig, 'fig4_variance')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Bimodality scatter
# ══════════════════════════════════════════════════════════════════════════════

def fig5_bimodality():
    print('Figure 5: bimodality scatter...')
    data = cached_experiment('fig3_dyn_frac', SNAP / '4_first_order_probe')

    target_mus = [0.33, 0.35, 0.37]
    colors = {0.33: C_PHASE_ORD, 0.35: C_PHASE_MIX, 0.37: C_PHASE_DIS}

    fig, ax = plt.subplots(figsize=(8, 5.5), tight_layout=True)
    rng = np.random.default_rng(42)

    spacing = 1.5  # wider horizontal spacing between groups

    for i, mu in enumerate(target_mus):
        cx = i * spacing
        seeds = [r for r in data if abs(r['mu'] - mu) < 0.005]
        bds = [s['bd_mean'] for s in seeds]
        jitter = rng.uniform(-0.3, 0.3, len(bds))
        ax.scatter(cx + jitter, bds, color=colors[mu], s=80, zorder=3,
                   edgecolors='white', linewidths=0.6)

        # Mean line — but NOT at μ=0.35 where bimodality makes mean misleading
        if mu != 0.35:
            ax.hlines(np.mean(bds), cx - 0.35, cx + 0.35,
                      color=colors[mu], lw=2.5, zorder=4)
        else:
            # Separate means for each basin
            ord_bds = [b for b in bds if b < 0.3]
            dis_bds = [b for b in bds if b >= 0.3]
            if ord_bds:
                ax.hlines(np.mean(ord_bds), cx - 0.35, cx + 0.35,
                          color=colors[mu], lw=2, zorder=4, linestyle='--')
            if dis_bds:
                ax.hlines(np.mean(dis_bds), cx - 0.35, cx + 0.35,
                          color=colors[mu], lw=2, zorder=4, linestyle='--')

    ax.set_xticks([i * spacing for i in range(3)])
    ax.set_xticklabels([r'$\mu = 0.33$', r'$\mu = 0.35$', r'$\mu = 0.37$'])
    ax.set_ylabel(r'Time-averaged boundary density  $\rho_b$')
    ax.set_ylim(-0.02, 0.65)
    ax.set_xlim(-0.7, spacing * 2 + 1.5)
    n_seeds = len([r for r in data if abs(r['mu'] - 0.35) < 0.005])
    ax.set_title('Per-seed boundary density at three mutation rates\n'
                 rf'Adaptive network, $128\times128$, {n_seeds} seeds, 30k frames')

    save(fig, 'fig5_bimodality')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6: Hysteresis
# ══════════════════════════════════════════════════════════════════════════════

def fig6_hysteresis():
    print('Figure 6: hysteresis...')
    hyst_dir = SNAP / '5_hysteresis'

    target_mus = [0.34, 0.35, 0.36]
    fig, axes = plt.subplots(1, 3, figsize=(9, 5), tight_layout=True, sharey=True)
    rng = np.random.default_rng(42)

    for idx, (ax, mu) in enumerate(zip(axes, target_mus)):
        mu_tag = f'{mu}'[2:].replace('.', '')  # 0.345 → '345', 0.34 → '34'
        rand_data = cached_experiment(f'fig6_hyst_0{mu_tag}_rand', hyst_dir / f'0{mu_tag}_random')
        ord_data = cached_experiment(f'fig6_hyst_0{mu_tag}_ord', hyst_dir / f'0{mu_tag}_ordered')

        rand_bds = [s['bd_mean'] for s in rand_data]
        ord_bds = [s['bd_mean'] for s in ord_data]

        jx = rng.uniform(-0.1, 0.1, len(rand_bds))
        ax.scatter(jx, rand_bds, color=C_RAND, s=45, zorder=3,
                   edgecolors='white', linewidths=0.4, label='Random init')
        if rand_bds:
            ax.hlines(np.mean(rand_bds), -0.3, 0.3, color=C_RAND, lw=2, zorder=4)

        jx_ord = rng.uniform(-0.1, 0.1, len(ord_bds))
        ax.scatter(1 + jx_ord, ord_bds, color=C_ORDI, s=45, zorder=3,
                   edgecolors='white', linewidths=0.4, label='Ordered init')
        if ord_bds:
            ax.hlines(np.mean(ord_bds), 0.7, 1.3, color=C_ORDI, lw=2, zorder=4)

        # Phase outcome label
        all_bds = rand_bds + ord_bds
        if all_bds:
            has_ord = any(b < 0.2 for b in all_bds)
            has_dis = any(b > 0.4 for b in all_bds)
            if has_ord and has_dis:
                phase_text = 'Bistable'
                phase_weight = 'bold'
            elif has_dis:
                phase_text = 'Disordered'
                phase_weight = 'normal'
            else:
                phase_text = 'Ordered'
                phase_weight = 'normal'
            ax.text(0.5, 0.60, phase_text, ha='center', fontsize=FONT_ANNOT,
                    color='gray', fontweight=phase_weight,
                    transform=ax.get_xaxis_transform())

        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Rand', 'Ord'], fontsize=FONT_TICK - 1)
        ax.set_title(r'$\mu = %.3f$' % mu, fontsize=FONT_TITLE)
        ax.set_ylim(-0.02, 0.65)
        ax.tick_params(axis='y', labelleft=(idx == 0), left=True)
        panel_label(ax, chr(ord('a') + idx))

    axes[0].set_ylabel(r'Time-averaged boundary density  $\rho_b$')
    axes[0].legend(loc='upper left', fontsize=FONT_LEGEND, framealpha=0.9)
    n_seeds = len(rand_bds)  # from last iteration, all conditions same
    fig.suptitle('Initial-condition dependence across the bistable window\n'
                 rf'Adaptive network, $128\times128$, $r_{{\min}}=4$, $k_{{\max}}=8$, '
                 f'{n_seeds} seeds per condition, 30k frames', fontsize=FONT_TITLE)

    save(fig, 'fig6_hysteresis')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7: Size dependence
# ══════════════════════════════════════════════════════════════════════════════

def fig7_size_dep():
    print('Figure 7: finite-size dependence...')

    sizes = {
        64:  SNAP / '6_size_dep_L64',
        128: SNAP / '4_first_order_probe',
        256: SNAP / '6_size_dep_L256',
    }
    cache_names = {64: 'fig7_L64', 128: 'fig3_dyn_frac', 256: 'fig7_L256'}
    colors = {64: C_L64, 128: C_L128, 256: C_L256}
    markers = {64: 'D', 128: 'o', 256: 's'}
    # Jitter to avoid overlapping error bars
    jitter = {64: -0.004, 128: 0.0, 256: 0.004}

    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)

    for L, snap_path in sizes.items():
        data = cached_experiment(cache_names[L], snap_path)
        groups = group_by_mu(data)

        mus, bd_means, bd_errs = [], [], []
        for mu, seeds in groups:
            bds = [s['bd_mean'] for s in seeds]
            mus.append(mu + jitter[L])
            bd_means.append(np.mean(bds))
            bd_errs.append(np.std(bds) / np.sqrt(len(bds)))

        ax.errorbar(mus, bd_means, yerr=bd_errs, fmt=f'{markers[L]}-',
                    color=colors[L], lw=1.5, ms=5, capsize=4, capthick=1,
                    label=f'L = {L}')

    ax.axvline(0.35, color='gray', ls='--', lw=0.8, alpha=0.4)
    ax.set_xlabel(r'Mutation probability  $\mu$')
    ax.set_ylabel(r'Boundary density  $\rho_b$')
    ax.legend(fontsize=FONT_LEGEND + 1)
    ax.set_title(r'Finite-size dependence of the phase transition'
                 '\n'
                 r'Adaptive network, $r_{\min}=4$, $k_{\max}=8$')

    save(fig, 'fig7_size_dep')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8: FSS at coexistence
# ══════════════════════════════════════════════════════════════════════════════

def fig8_fss():
    print('Figure 8: FSS at μ=0.35...')
    fss_dir = SNAP / '7_fss_035'

    sizes = [64, 128, 256]

    fig, axes = plt.subplots(1, 3, figsize=(11, 5), tight_layout=True, sharey=True)

    all_vars = {}
    for idx, (ax, L) in enumerate(zip(axes, sizes)):
        rng = np.random.default_rng(42)

        rand_data = cached_experiment(f'fig8_fss_L{L}_rand', fss_dir / f'L{L}_random')
        ord_data = cached_experiment(f'fig8_fss_L{L}_ord', fss_dir / f'L{L}_ordered')

        rand_bds = [s['bd_mean'] for s in rand_data]
        ord_bds = [s['bd_mean'] for s in ord_data]

        jx = rng.uniform(-0.1, 0.1, len(rand_bds))
        ax.scatter(jx, rand_bds, color=C_RAND, s=60, zorder=3,
                   edgecolors='white', linewidths=0.5, label='Random init')
        if rand_bds:
            ax.hlines(np.mean(rand_bds), -0.3, 0.3, color=C_RAND, lw=2, zorder=4)

        jx = rng.uniform(-0.1, 0.1, len(ord_bds))
        ax.scatter(1 + jx, ord_bds, color=C_ORDI, s=60, zorder=3,
                   edgecolors='white', linewidths=0.5, label='Ordered init')
        if ord_bds:
            ax.hlines(np.mean(ord_bds), 0.7, 1.3, color=C_ORDI, lw=2, zorder=4)

        combined_var = np.var(rand_bds + ord_bds) if (rand_bds and ord_bds) else 0
        all_vars[L] = combined_var

        # Scientific notation for variance
        var_str = f'{combined_var:.1e}'

        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Random\ninit', 'Ordered\ninit'])
        ax.set_title(r'$L = %d$' % L + f'\nvar = {var_str}', fontsize=FONT_TITLE)
        ax.set_ylim(-0.02, 0.65)
        ax.tick_params(axis='y', labelleft=(idx == 0), left=True)
        panel_label(ax, chr(ord('a') + idx))

        # Phase annotation
        all_bds = rand_bds + ord_bds
        mean_bd = np.mean(all_bds)
        if combined_var < 1e-3 and mean_bd < 0.2:
            phase_text = 'Ordered'
            phase_weight = 'normal'
        elif combined_var < 1e-3 and mean_bd > 0.4:
            phase_text = 'Disordered'
            phase_weight = 'normal'
        else:
            phase_text = 'Bistable'
            phase_weight = 'bold'
        ax.text(0.5, 0.58, phase_text,
                ha='center', fontsize=FONT_ANNOT, color='gray',
                fontweight=phase_weight, transform=ax.get_xaxis_transform())

    axes[0].set_ylabel(r'Time-averaged boundary density  $\rho_b$')
    axes[0].legend(loc='center left', fontsize=FONT_LEGEND, framealpha=0.9)
    n_seeds = len(rand_bds)  # from last iteration
    fig.suptitle(r'Finite-size scaling at the coexistence point  $\mu = 0.35$'
                 '\n'
                 rf'Adaptive network, $r_{{\min}}=4$, $k_{{\max}}=8$, '
                 f'{n_seeds} seeds per condition', fontsize=FONT_TITLE)

    save(fig, 'fig8_fss')

    for L, v in sorted(all_vars.items()):
        print(f'  L={L}: var(ρ_b) = {v:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9: r_min control — fixed lattice r_min=2 vs r_min=4
# ══════════════════════════════════════════════════════════════════════════════

def fig9_rmin_control():
    print('Figure 9: r_min=2 vs r_min=4 control...')
    data_rm4 = cached_experiment('fig1_fixed', SNAP / '1_fixed_lattice_phase')
    data_rm2 = cached_experiment('fig9_rmin2', SNAP / 'rmin2_control')

    if not data_rm2:
        print('  No r_min=2 data found, skipping')
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6.5), tight_layout=True,
        gridspec_kw={'height_ratios': [1, 1]})

    for data, rm, color, marker, label in [
        (data_rm4, 4, C_DYN,  'o', r'$r_{\min} = 4$'),
        (data_rm2, 2, C_RAND, 's', r'$r_{\min} = 2$'),
    ]:
        groups = group_by_mu(data)
        mus, bd_means, bd_errs, bd_vars = [], [], [], []
        for mu, seeds in groups:
            bds = [s['bd_mean'] for s in seeds]
            mus.append(mu)
            bd_means.append(np.mean(bds))
            bd_errs.append(np.std(bds) / np.sqrt(len(bds)))
            bd_vars.append(np.var(bds))

        n_seeds = len(groups[0][1]) if groups else '?'
        ax1.errorbar(mus, bd_means, yerr=bd_errs, fmt=f'{marker}-',
                     color=color, lw=1.5, ms=5, capsize=4, capthick=1,
                     label=f'{label} ({n_seeds} seeds)')
        ax2.semilogy(mus, np.array(bd_vars) + 1e-10, f'{marker}-',
                     color=color, lw=1.5, ms=5, label=label)

    # (a) boundary density
    ax1.set_ylabel(r'Boundary density  $\rho_b$')
    ax1.set_xlabel(r'Mutation probability  $\mu$')
    ax1.legend(fontsize=FONT_LEGEND + 1)
    panel_label(ax1, 'a')

    # (b) variance
    ax2.set_ylabel(r'Seed-to-seed variance  var($\rho_b$)')
    ax2.set_xlabel(r'Mutation probability  $\mu$')
    ax2.legend(fontsize=FONT_LEGEND + 1)
    panel_label(ax2, 'b')

    fig.suptitle(r'Reinforcement threshold control: fixed lattice $128\times128$, 50k frames',
                 fontsize=FONT_TITLE)

    save(fig, 'fig9_rmin_control')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10: Three-way comparison (fixed vs local-only vs global adaptive)
# ══════════════════════════════════════════════════════════════════════════════

C_FIXED = C_RAND    # blue — fixed lattice
C_LOCAL = '#7570b3'  # purple — local-only adaptive
C_GLOBAL = C_DYN    # red — global adaptive

def fig10_threeway():
    print('Figure 10: three-way comparison (fixed vs local-only vs global)...')

    # --- Fixed lattice (per-seed cache) ---
    fixed_data = cached_experiment('fig1_fixed', SNAP / '1_fixed_lattice_phase')
    fixed_groups = group_by_mu(fixed_data)
    fix_mus, fix_bds, fix_errs = [], [], []
    for mu, seeds in fixed_groups:
        bds = [s['bd_mean'] for s in seeds]
        fix_mus.append(mu)
        fix_bds.append(np.mean(bds))
        fix_errs.append(np.std(bds) / np.sqrt(len(bds)))

    # --- Global adaptive (per-seed cache) ---
    data_int = cached_experiment('fig3_dyn_int', SNAP / '3_dyn_graph_phase')
    data_frac = cached_experiment('fig3_dyn_frac', SNAP / '4_first_order_probe')
    global_data = data_int + data_frac
    global_groups = group_by_mu(global_data)
    glob_mus, glob_bds, glob_errs = [], [], []
    for mu, seeds in global_groups:
        bds = [s['bd_mean'] for s in seeds]
        glob_mus.append(mu)
        glob_bds.append(np.mean(bds))
        glob_errs.append(np.std(bds) / np.sqrt(len(bds)))

    # --- Local-only adaptive (aggregate sweep CSV) ---
    local_csv = RESULTSDIR / 'sweep_dyn_128x128_f30000_s64_20260319_100037.csv'
    local_rows = load_sweep_csv(local_csv)
    loc_mus = [r['mutation_rate'] for r in local_rows]
    loc_bds = [r['bd_mean'] for r in local_rows]
    loc_errs = [r['bd_mean_std'] / np.sqrt(r['n_seeds']) for r in local_rows]

    # --- Global adaptive k_max=4 (aggregate sweep CSV) ---
    C_KMAX4 = '#66a61e'  # green — k_max=4 control
    kmax4_csv = RESULTSDIR / 'sweep_dyn_128x128_f30000_s64_20260320_121212.csv'
    kmax4_data, kmax4_mus, kmax4_bds, kmax4_errs = [], [], [], []
    if kmax4_csv.exists():
        kmax4_data = load_sweep_csv(kmax4_csv)
        kmax4_mus = [r['mutation_rate'] for r in kmax4_data]
        kmax4_bds = [r['bd_mean'] for r in kmax4_data]
        kmax4_errs = [r['bd_mean_std'] / np.sqrt(r['n_seeds']) for r in kmax4_data]

    # --- Global adaptive k_max=5 and k_max=6 (intermediate-headroom controls) ---
    C_KMAX5 = '#fdae61'  # warm orange
    C_KMAX6 = '#d95f02'  # darker orange
    kmax5_csv = RESULTSDIR / 'sweep_dyn_128x128_f30000_s64_20260502_224313.csv'
    kmax6_csv = RESULTSDIR / 'sweep_dyn_128x128_f30000_s64_20260502_234631.csv'
    kmax5_mus = kmax5_bds = kmax5_errs = None
    kmax6_mus = kmax6_bds = kmax6_errs = None
    if kmax5_csv.exists():
        d = load_sweep_csv(kmax5_csv)
        kmax5_mus = [r['mutation_rate'] for r in d]
        kmax5_bds = [r['bd_mean'] for r in d]
        kmax5_errs = [r['bd_mean_std'] / np.sqrt(r['n_seeds']) for r in d]
    if kmax6_csv.exists():
        d = load_sweep_csv(kmax6_csv)
        kmax6_mus = [r['mutation_rate'] for r in d]
        kmax6_bds = [r['bd_mean'] for r in d]
        kmax6_errs = [r['bd_mean_std'] / np.sqrt(r['n_seeds']) for r in d]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)

    ax.errorbar(fix_mus, fix_bds, yerr=fix_errs, fmt='o-', color=C_FIXED,
                lw=1.5, ms=5, capsize=3, capthick=1,
                label='Fixed lattice', zorder=2)
    ax.errorbar(loc_mus, loc_bds, yerr=loc_errs, fmt='D-', color=C_LOCAL,
                lw=1.5, ms=5, capsize=3, capthick=1,
                label=r'Local-only adaptive ($k_{\max}=8$)', zorder=3)
    if kmax4_data:
        ax.errorbar(kmax4_mus, kmax4_bds, yerr=kmax4_errs, fmt='^-', color=C_KMAX4,
                    lw=1.5, ms=5, capsize=3, capthick=1,
                    label=r'Global adaptive ($k_{\max}=4$)', zorder=3)
    if kmax5_mus is not None:
        ax.errorbar(kmax5_mus, kmax5_bds, yerr=kmax5_errs, fmt='v-', color=C_KMAX5,
                    lw=1.5, ms=5, capsize=3, capthick=1,
                    label=r'Global adaptive ($k_{\max}=5$)', zorder=3)
    if kmax6_mus is not None:
        ax.errorbar(kmax6_mus, kmax6_bds, yerr=kmax6_errs, fmt='<-', color=C_KMAX6,
                    lw=1.5, ms=5, capsize=3, capthick=1,
                    label=r'Global adaptive ($k_{\max}=6$)', zorder=3)
    ax.errorbar(glob_mus, glob_bds, yerr=glob_errs, fmt='s-', color=C_GLOBAL,
                lw=1.8, ms=5, capsize=3, capthick=1,
                label=r'Global adaptive ($k_{\max}=8$)', zorder=4)

    ax.axvline(0.35, color='gray', ls='--', lw=0.8, alpha=0.4)

    ax.set_xlabel(r'Mutation probability  $\mu$')
    ax.set_ylabel(r'Boundary density  $\rho_b$')
    ax.set_xlim(0.08, 0.55)
    ax.set_ylim(-0.02, 0.60)
    ax.legend(fontsize=FONT_LEGEND, loc='upper left')
    ax.set_title('Control comparisons: formation range and degree cap\n'
                 r'$128\times128$, $r_{\min}=4$, 64 seeds, 30k frames')

    save(fig, 'fig_threeway')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 11: Effective free energy landscape
# ══════════════════════════════════════════════════════════════════════════════

def fig11_free_energy():
    print('Figure 11: effective free energy landscape...')
    from scipy.stats import gaussian_kde

    # --- L=128 fine grid: pick bimodal μ values ---
    fine_data = cached_experiment('binder_L128_fine', SNAP / 'B_binder_L128_fine')

    # --- L=128 coarse grid (has μ=0.355 with 64 seeds) ---
    coarse_data = cached_experiment('fig3_dyn_frac', SNAP / '4_first_order_probe')

    # --- L=256 FSS at μ=0.35 ---
    fss_rand = cached_experiment('fig8_fss_L256_rand', SNAP / '7_fss_035' / 'L256_random')
    fss_ord  = cached_experiment('fig8_fss_L256_ord',  SNAP / '7_fss_035' / 'L256_ordered')
    fss_256  = fss_rand + fss_ord

    # Combine fine + coarse for L=128 to get best coverage
    all_128 = fine_data + coarse_data

    # Pick μ values that show progression through the transition
    panels = [
        (128, 0.349, all_128, C_L128),
        (128, 0.351, all_128, C_L128),
        (128, 0.355, all_128, C_L128),
    ]

    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(12, 3.5),
                             tight_layout=True, sharey=True)

    x_grid = np.linspace(0.02, 0.62, 300)

    for idx, (L, mu, data, color) in enumerate(panels):
        ax = axes[idx]
        bds = np.array([s['bd_mean'] for s in data
                        if abs(s['mu'] - mu) < 0.0015])
        if len(bds) < 3:
            continue

        kde = gaussian_kde(bds, bw_method=0.35)
        density = kde(x_grid)
        density = np.maximum(density, density[density > 0].min() * 0.01)
        free_energy = -np.log(density)
        free_energy -= free_energy.min()

        ax.plot(x_grid, free_energy, '-', color=color, lw=1.5)
        ax.fill_between(x_grid, free_energy, alpha=0.12, color=color)
        ax.set_xlabel(r'$\rho_b$')
        ax.set_title(r'$L=%d$, $\mu=%.3f$' % (L, mu), fontsize=FONT_TITLE)
        ax.set_xlim(0, 0.65)
        ax.set_ylim(0, 8)
        if idx == 0:
            ax.set_ylabel(r'$-\ln\,P(\rho_b)$  (shifted)')
        panel_label(ax, chr(ord('a') + idx))

    # L=256 at μ=0.35
    ax = axes[-1]
    bds_256 = np.array([s['bd_mean'] for s in fss_256])
    kde = gaussian_kde(bds_256, bw_method=0.35)
    density = kde(x_grid)
    density = np.maximum(density, density[density > 0].min() * 0.01)
    free_energy = -np.log(density)
    free_energy -= free_energy.min()

    ax.plot(x_grid, free_energy, '-', color=C_L256, lw=1.5)
    ax.fill_between(x_grid, free_energy, alpha=0.12, color=C_L256)
    ax.set_xlabel(r'$\rho_b$')
    ax.set_title(r'$L=256$, $\mu=0.350$', fontsize=FONT_TITLE)
    ax.set_xlim(0, 0.65)
    panel_label(ax, chr(ord('a') + len(panels)))

    fig.suptitle(r'Effective free energy $-\ln\,P(\rho_b)$ across the transition',
                 fontsize=FONT_TITLE)

    save(fig, 'fig_free_energy')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global RECOMPUTE
    parser = argparse.ArgumentParser(description='Generate paper figures.')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute from snapshots (ignore cache)')
    parser.add_argument('figures', nargs='*',
                        help='Generate only these figures (e.g. fig1 fig5). '
                             'Default: all.')
    args = parser.parse_args()
    RECOMPUTE = args.recompute

    FIGDIR.mkdir(exist_ok=True)
    print(f'Output: {FIGDIR}/')
    if not RECOMPUTE:
        print(f'Cache:  {CACHEDIR}/  (use --recompute to rebuild)')
    print()

    all_figs = {
        'fig1': fig1_fixed_phase,
        'fig2': fig2_tau,
        'fig3': fig3_dyn_phase,
        'fig4': fig4_variance,
        'fig5': fig5_bimodality,
        'fig6': fig6_hysteresis,
        'fig7': fig7_size_dep,
        'fig8': fig8_fss,
        'fig9': fig9_rmin_control,
        'fig10': fig10_threeway,
        'fig11': fig11_free_energy,
    }

    targets = args.figures if args.figures else list(all_figs.keys())
    for name in targets:
        if name in all_figs:
            all_figs[name]()
        else:
            print(f'  unknown figure: {name}')

    print(f'\nFigures written to {FIGDIR}/')


if __name__ == '__main__':
    main()
