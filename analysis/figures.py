#!/usr/bin/env python3
"""
figures.py — publication figures for the torus adaptive-network CA paper.

Generates five figures to figures/:
  1. fig1_phase_diagram  — bd and variance vs mutation probability, both systems
  2. fig2_bimodality     — per-seed bd at p=0.33/0.35/0.37 (first-order evidence)
  3. fig3_hysteresis     — random vs ordered init at p=0.35/0.37/0.40
  4. fig4_tau            — domain-size exponent tau(L) finite-size scaling
  5. fig5_frames         — rendered frames, visual comparison

Usage:
    venv/bin/python3 analysis/figures.py            # full run (may run sims)
    venv/bin/python3 analysis/figures.py --no-sims  # skip sims, use cache/hardcoded
"""

import sys
import csv as csv_mod
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / 'results'
FIGS    = ROOT / 'figures'
BIN_FIX = ROOT / 'torus'
BIN_DYN = ROOT / 'torus_dyn'

# ── style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         10,
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'legend.fontsize':   9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
    'figure.dpi':        150,
})

C_FIX  = '#2166ac'   # fixed lattice: blue
C_DYN  = '#ca0020'   # dynamic graph: red
C_ORD  = '#4dac26'   # ordered phase: green
C_DIS  = '#d7191c'   # disordered phase: red
C_RAND = '#2166ac'   # random init: blue
C_ORDI = '#f4a442'   # ordered init: orange

# ── utilities ──────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as f:
        return list(csv_mod.DictReader(f))

def save_fig(fig, name):
    FIGS.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'{name}.{ext}', bbox_inches='tight', dpi=150)
    print(f'  saved: figures/{name}.pdf  +  .png')
    plt.close(fig)

# ── simulation runner ──────────────────────────────────────────────────────────

def _run_seed(args):
    prob, seed, frames, ordered_init = args
    cmd = [
        str(BIN_DYN), '--headless',
        '--stats-interval', '50',
        '--frames', str(frames),
        '--mutation-prob', f'{prob:.8f}',
        '--max-degree', '8',
    ]
    if ordered_init:
        cmd.append('--ordered-init')
    cmd += ['128', '128', '0', str(seed), '4']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None
    rows = []
    for line in r.stderr.splitlines():
        if line.startswith(('seed=', 'frame,')):
            continue
        parts = line.split(',')
        if len(parts) >= 8:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if len(rows) < 4:
        return None
    arr  = np.array(rows)
    post = arr[len(arr) // 2:]
    bd   = post[:, 2].mean()
    deg  = post[:, 32].mean() if arr.shape[1] > 32 else float('nan')
    return {'prob': prob, 'seed': seed, 'bd': bd, 'deg': deg,
            'ordered_init': ordered_init}

def run_parallel(specs, workers=4):
    """specs: list of (prob, seed, frames, ordered_init)."""
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_seed, s): s for s in specs}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                results.append(r)
                tag = 'ord' if r['ordered_init'] else 'rnd'
                print(f'    p={r["prob"]}  seed={r["seed"]}  {tag}  '
                      f'bd={r["bd"]:.4f}')
    return results

# ── frame capture ──────────────────────────────────────────────────────────────

def capture_frame(binary, extra_args, warmup=1000, width=128, height=128):
    """Run binary, skip warmup frames, return next frame as HxWx3 uint8."""
    cmd = [str(binary)] + extra_args + [str(width), str(height), '0', '42', '4']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    n = width * height * 3
    try:
        for _ in range(warmup):
            d = proc.stdout.read(n)
            if len(d) < n:
                return None
        data = proc.stdout.read(n)
    finally:
        proc.kill()
        proc.wait()
    if len(data) < n:
        return None
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)

# ── data loading ───────────────────────────────────────────────────────────────

def load_fix_phase():
    """Fixed lattice phase diagram (rm=4). Returns (x, bd, err, var)."""
    path = max(RESULTS.glob('sweep_128x128_f50000_s8_*.csv'),
               key=lambda p: p.stat().st_mtime)
    rows = sorted(
        [r for r in load_csv(path) if int(r['reinforce_min']) == 4],
        key=lambda r: float(r['mutation_rate'])
    )
    x   = np.array([1.0 / float(r['mutation_rate']) for r in rows])
    bd  = np.array([float(r['bd_mean'])              for r in rows])
    err = np.array([float(r['bd_mean_std'])           for r in rows])
    var = np.array([float(r['bd_var_mean'])           for r in rows])
    return x, bd, err, var

def load_dyn_phase():
    """Dynamic graph phase diagram (rm=4). Returns (x, bd, err, var, deg).
    Uses the three canonical phase-diagram CSVs only."""
    canonical = [
        'sweep_dyn_128x128_f30000_s8_20260310_202734.csv',
        'sweep_dyn_128x128_f30000_s8_20260310_212616.csv',
        'sweep_dyn_128x128_f30000_s8_20260311_210549.csv',
    ]
    merged = {}
    for name in canonical:
        path = RESULTS / name
        if not path.exists():
            print(f'  WARNING: {name} not found, skipping')
            continue
        for r in load_csv(path):
            if int(float(r['reinforce_min'])) != 4:
                continue
            prob = (round(float(r['mutation_prob']), 8)
                    if 'mutation_prob' in r
                    else round(1.0 / float(r['mutation_rate']), 8))
            merged[prob] = r   # last file wins on conflict

    pts = sorted(merged.values(),
                 key=lambda r: float(r.get('mutation_prob',
                                           1.0 / float(r['mutation_rate']))))
    def _prob(r):
        return float(r.get('mutation_prob', 1.0 / float(r['mutation_rate'])))

    x   = np.array([_prob(r)             for r in pts])
    bd  = np.array([float(r['bd_mean'])  for r in pts])
    err = np.array([float(r['bd_mean_std']) for r in pts])
    var = np.array([float(r['bd_var_mean']) for r in pts])
    deg = np.array([float(r['deg_mean']) if 'deg_mean' in r else float('nan')
                    for r in pts])
    return x, bd, err, var, deg

# ── per-seed data with caching ─────────────────────────────────────────────────

BIMOD_CACHE = RESULTS / 'perseeds_bimodality.csv'
HYST_CACHE  = RESULTS / 'perseeds_hysteresis.csv'

def load_or_run_bimodality(run_sims, workers=4):
    if BIMOD_CACHE.exists():
        print('  (loading bimodality cache)')
        rows = load_csv(BIMOD_CACHE)
        data = {}
        for r in rows:
            data.setdefault(float(r['prob']), []).append(float(r['bd']))
        return data

    if not run_sims:
        raise RuntimeError(
            'No bimodality cache found. Run without --no-sims first.')

    print('  running bimodality sims (p=0.33, 0.35, 0.37 × 8 seeds × 30k frames)...')
    probs = [0.33, 0.35, 0.37]
    specs = [(p, s, 30000, False) for p in probs for s in range(8)]
    results = run_parallel(specs, workers=workers)

    with open(BIMOD_CACHE, 'w', newline='') as f:
        w = csv_mod.DictWriter(f, fieldnames=['prob', 'seed', 'bd', 'deg'])
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: (r['prob'], r['seed'])))
    print(f'  cached: {BIMOD_CACHE.name}')

    data = {}
    for r in results:
        data.setdefault(r['prob'], []).append(r['bd'])
    return data

def load_or_run_hysteresis(run_sims, workers=4):
    if HYST_CACHE.exists():
        print('  (loading hysteresis cache)')
        rows = load_csv(HYST_CACHE)
        data = {}
        for r in rows:
            key = (round(float(r['prob']), 4), r['ordered_init'] == 'True')
            data.setdefault(key, []).append(float(r['bd']))
        return data

    if not run_sims:
        print('  (using hardcoded hysteresis values from research log)')
        return _hysteresis_hardcoded()

    print('  running hysteresis sims (p=0.35,0.37,0.40 × 2 inits × 8 seeds × 30k frames)...')
    probs = [0.35, 0.37, 0.40]
    specs = [(p, s, 30000, o)
             for p in probs for s in range(8) for o in (False, True)]
    results = run_parallel(specs, workers=workers)

    with open(HYST_CACHE, 'w', newline='') as f:
        w = csv_mod.DictWriter(f, fieldnames=['prob', 'seed', 'bd', 'deg', 'ordered_init'])
        w.writeheader()
        w.writerows(sorted(results,
                           key=lambda r: (r['prob'], r['ordered_init'], r['seed'])))
    print(f'  cached: {HYST_CACHE.name}')

    data = {}
    for r in results:
        key = (round(r['prob'], 4), r['ordered_init'])
        data.setdefault(key, []).append(r['bd'])
    return data

def _hysteresis_hardcoded():
    """Per-seed values from research-log/2026-03-12-a.md and overnight log."""
    return {
        (0.35, False): [0.5047, 0.1762, 0.1613, 0.1440,
                        0.1624, 0.1659, 0.1303, 0.1542],
        (0.35, True):  [0.1348, 0.1276, 0.2536, 0.1431,
                        0.1977, 0.1353, 0.1289, 0.2548],
        (0.37, False): [0.5511, 0.5508, 0.5513, 0.5513,
                        0.5514, 0.5510, 0.5514, 0.5517],
        (0.37, True):  [0.5508, 0.5511, 0.5513, 0.5514,
                        0.5513, 0.5509, 0.5510, 0.5516],
        (0.40, False): [0.5593, 0.5587, 0.5590, 0.5593,
                        0.5589, 0.5589, 0.5593, 0.5594],
        (0.40, True):  [0.5588, 0.5588, 0.5592, 0.5588,
                        0.5579, 0.5587, 0.5592, 0.5590],
    }

# ── Figure 1: phase diagram ────────────────────────────────────────────────────

def fig1_phase_diagram():
    print('Figure 1: phase diagram...')
    fx, fbd, ferr, fvar           = load_fix_phase()
    dx, dbd, derr, dvar, ddeg     = load_dyn_phase()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), tight_layout=True)

    # bd panel
    ax1.errorbar(fx, fbd, yerr=ferr, fmt='o-', color=C_FIX,
                 label='Fixed lattice', lw=1.5, ms=4, capsize=3, zorder=3)
    ax1.errorbar(dx, dbd, yerr=derr, fmt='s-', color=C_DYN,
                 label='Adaptive network', lw=1.5, ms=4, capsize=3, zorder=3)
    ax1.axvline(0.20, color=C_FIX, ls=':', lw=1.2, alpha=0.7)
    ax1.axvline(0.35, color=C_DYN, ls=':', lw=1.2, alpha=0.7)
    ax1.text(0.20, 0.74, 'p* ≈ 0.20', color=C_FIX, fontsize=8.5, ha='center')
    ax1.text(0.35, 0.74, 'p* ≈ 0.35', color=C_DYN, fontsize=8.5, ha='center')
    ax1.set_ylabel('Mean boundary density')
    ax1.set_ylim(-0.02, 0.80)
    ax1.set_xlim(0.05, 0.70)
    ax1.legend(loc='lower right')
    ax1.set_title('Phase diagram  (128×128, reinforce_min=4, 8 seeds)')

    # variance panel (log scale)
    ax2.semilogy(fx, fvar + 1e-9, 'o-', color=C_FIX,
                 label='Fixed lattice', lw=1.5, ms=4)
    ax2.semilogy(dx, dvar + 1e-9, 's-', color=C_DYN,
                 label='Adaptive network', lw=1.5, ms=4)
    ax2.axvline(0.20, color=C_FIX, ls=':', lw=1.2, alpha=0.7)
    ax2.axvline(0.35, color=C_DYN, ls=':', lw=1.2, alpha=0.7)
    ax2.set_ylabel('Within-run variance  (log scale)')
    ax2.set_xlabel('Mutation probability')
    ax2.set_xlim(0.05, 0.70)
    ax2.legend(loc='upper left')
    ax2.set_title('Variance peak locates the critical point')

    save_fig(fig, 'fig1_phase_diagram')

# ── Figure 2: bimodality ───────────────────────────────────────────────────────

def fig2_bimodality(run_sims):
    print('Figure 2: bimodality...')
    data = load_or_run_bimodality(run_sims)
    rng  = np.random.default_rng(42)

    probs  = [0.33, 0.35, 0.37]
    colors = {0.33: C_ORD, 0.35: '#f4a442', 0.37: C_DIS}
    labels = {
        0.33: 'p = 0.33  (ordered)',
        0.35: 'p = 0.35  (coexistence)',
        0.37: 'p = 0.37  (disordered)',
    }

    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)

    for i, p in enumerate(probs):
        vals = data.get(p, [])
        if not vals:
            continue
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, color=colors[p], s=60, zorder=3,
                   label=labels[p], edgecolors='white', linewidths=0.5)
        ax.hlines(np.mean(vals), i - 0.22, i + 0.22,
                  color=colors[p], lw=2.5, zorder=4)

    # empty gap zone
    ax.axhspan(0.19, 0.49, alpha=0.07, color='gray')
    ax.text(2.42, 0.34, 'empty\ngap', fontsize=8.5, color='gray',
            ha='left', va='center', style='italic')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f'p = {p}' for p in probs])
    ax.set_ylabel('Time-averaged boundary density')
    ax.set_ylim(-0.02, 0.70)
    ax.set_xlim(-0.55, 2.75)
    ax.legend(loc='upper left', framealpha=0.92)
    ax.set_title('Bimodal order parameter at the transition\n'
                 '(adaptive network, 128×128, rm=4, 30k frames, 8 seeds)')

    save_fig(fig, 'fig2_bimodality')

# ── Figure 3: hysteresis ───────────────────────────────────────────────────────

def fig3_hysteresis(run_sims):
    print('Figure 3: hysteresis...')
    data = load_or_run_hysteresis(run_sims)
    rng  = np.random.default_rng(42)

    probs = [0.35, 0.37, 0.40]
    fig, axes = plt.subplots(1, 3, figsize=(9, 4), tight_layout=True, sharey=True)

    for ax, p in zip(axes, probs):
        rand_vals = data.get((p, False), [])
        ord_vals  = data.get((p, True),  [])

        jx = rng.uniform(-0.09, 0.09, len(rand_vals))
        ax.scatter(jx, rand_vals, color=C_RAND, s=52, zorder=3,
                   label='Random init', edgecolors='white', linewidths=0.5)
        ax.hlines(np.mean(rand_vals), -0.28, 0.28, color=C_RAND, lw=2.5, zorder=4)

        jx = rng.uniform(-0.09, 0.09, len(ord_vals))
        ax.scatter(1 + jx, ord_vals, color=C_ORDI, s=52, zorder=3,
                   label='Ordered init', edgecolors='white', linewidths=0.5)
        ax.hlines(np.mean(ord_vals), 0.72, 1.28, color=C_ORDI, lw=2.5, zorder=4)

        ax.set_xlim(-0.55, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Random\ninit', 'Ordered\ninit'])
        ax.set_title(f'p = {p}', fontweight='bold', fontsize=12)
        ax.set_ylim(-0.02, 0.70)

    axes[0].set_ylabel('Time-averaged boundary density')
    axes[0].legend(loc='upper right', fontsize=8.5, framealpha=0.92)
    fig.suptitle(
        'Hysteresis: same parameters, different initial conditions\n'
        '(adaptive network, 128×128, rm=4, 30k frames, 8 seeds)',
        fontsize=10)

    save_fig(fig, 'fig3_hysteresis')

# ── Figure 4: tau(L) ──────────────────────────────────────────────────────────

def fig4_tau():
    print('Figure 4: tau(L)...')
    # Values from research-log/2026-03-10-b.md
    Ls   = np.array([64,    128,   256,   512  ])
    taus = np.array([1.711, 1.890, 1.999, 1.917])
    r2s  = np.array([0.965, 0.961, 0.925, 0.946])

    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

    ax.plot(1.0 / Ls, taus, 'o-', color=C_FIX, lw=1.8, ms=7, zorder=3,
            label='Measured τ(L)')
    for L, tau, r2 in zip(Ls, taus, r2s):
        ax.annotate(f'L={L}\nR²={r2:.3f}',
                    xy=(1/L, tau), xytext=(6, 5),
                    textcoords='offset points', fontsize=7.5)

    # Reference lines
    tau_perc = 187 / 91
    ax.axhline(2.0,      color='#888888', ls='--', lw=1.2,
               label='τ = 2.0')
    ax.axhline(tau_perc, color='#bbbbbb', ls=':', lw=1.0,
               label=f'τ = 187/91 ≈ {tau_perc:.3f}  (2D percolation, exact)')

    # Linear extrapolation through L=64, 128, 256
    inv_L  = 1.0 / Ls
    coeffs = np.polyfit(inv_L[:3], taus[:3], 1)
    x_fit  = np.linspace(0, inv_L[0] * 1.05, 200)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=C_FIX,
            lw=1, alpha=0.45,
            label=f'Linear fit  (L=64–256),  τ(∞) ≈ {coeffs[1]:.3f}')
    ax.plot(0, coeffs[1], '*', color=C_FIX, ms=9, zorder=4)

    ax.set_xlabel('1 / L')
    ax.set_ylabel('Domain-size exponent τ')
    ax.set_xlim(-0.0015, inv_L[0] * 1.25)
    ax.set_ylim(1.55, 2.15)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('Finite-size scaling of τ\n'
                 '(fixed lattice, mr=5, rm=4, 4 seeds per L)')

    save_fig(fig, 'fig4_tau')

# ── Figure 5: simulation frames ───────────────────────────────────────────────

def fig5_frames():
    print('Figure 5: simulation frames...')
    specs = [
        dict(binary=BIN_FIX,
             args=['--mutation-rate', '5'],
             warmup=1500,
             title='Fixed lattice\np = 0.20  (critical)',
             sub='Continuous transition · τ → 2.0\nVoter model universality class'),
        dict(binary=BIN_DYN,
             args=['--mutation-prob', '0.20', '--max-degree', '8'],
             warmup=3000,
             title='Adaptive network\np = 0.20  (ordered)',
             sub='Spontaneous homophily\nSelf-organised topology'),
        dict(binary=BIN_DYN,
             args=['--mutation-prob', '0.40', '--max-degree', '8'],
             warmup=800,
             title='Adaptive network\np = 0.40  (disordered)',
             sub='Discontinuous transition\nSeverance dominates'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), tight_layout=True)

    for ax, spec in zip(axes, specs):
        label = spec['title'].split('\n')[0]
        print(f'  capturing: {label}...')
        img = capture_frame(spec['binary'], spec['args'], warmup=spec['warmup'])
        if img is not None:
            big = img.repeat(4, axis=0).repeat(4, axis=1)
            ax.imshow(big, interpolation='nearest')
        else:
            ax.text(0.5, 0.5, '(capture failed)',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_title(spec['title'], fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel(spec['sub'], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle('Visual comparison: fixed lattice vs adaptive network  (128×128)',
                 fontsize=11)
    save_fig(fig, 'fig5_frames')

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-sims', action='store_true',
                        help='skip simulation runs; use cached/hardcoded data only')
    args = parser.parse_args()
    run_sims = not args.no_sims

    np.random.seed(42)
    print(f'Output: {FIGS}/')
    if run_sims:
        print('Simulation runs enabled. Figure 2 may take ~30 min on first run.')
        print('Use --no-sims to skip (uses cached data or hardcoded values).\n')

    fig1_phase_diagram()
    try:
        fig2_bimodality(run_sims)
    except RuntimeError as e:
        print(f'  SKIPPED: {e}')
    fig3_hysteresis(run_sims)
    fig4_tau()
    fig5_frames()

    print(f'\nAll figures written to {FIGS}/')

if __name__ == '__main__':
    main()
