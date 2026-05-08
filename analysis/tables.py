#!/usr/bin/env python3
"""
tables.py — produce summary tables from snapshot data for the paper.

Outputs LaTeX table fragments to paper/tables.tex and prints
human-readable versions to stdout.

Usage:
    venv/bin/python3 analysis/tables.py
"""

import re
from pathlib import Path
from collections import defaultdict

import numpy as np

# Reuse loading functions from figures.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from figures import (collect_experiment, group_by_mu, SNAP, ROOT,
                     compute_bd_fast, compute_deg_fast, seed_summary)

FIGDIR = ROOT / 'paper'


def fmt(val, decimals=4):
    """Format a float to fixed decimals."""
    return f'{val:.{decimals}f}'


def fmt_sci(val, decimals=1):
    """Format in scientific notation."""
    if val == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(val))))
    mantissa = val / 10**exp
    return f'{mantissa:.{decimals}f} \\times 10^{{{exp}}}'


# ══════════════════════════════════════════════════════════════════════════════
# Table 1: Fixed lattice — ρ_b vs μ
# ══════════════════════════════════════════════════════════════════════════════

def table1_fixed():
    print('Table 1: Fixed lattice phase diagram')
    print('=' * 65)
    data = collect_experiment(SNAP / '1_fixed_lattice_phase')
    groups = group_by_mu(data)

    header = f'{"μ":>8}  {"ρ_b (mean)":>12}  {"± std":>10}  {"var(ρ_b)":>12}  {"n seeds":>7}'
    print(header)
    print('-' * 65)

    rows = []
    for mu, seeds in groups:
        bds = [s['bd_mean'] for s in seeds]
        row = {
            'mu': mu,
            'bd_mean': np.mean(bds),
            'bd_std': np.std(bds),
            'bd_var': np.var(bds),
            'n': len(seeds),
        }
        rows.append(row)
        print(f'{mu:8.4f}  {row["bd_mean"]:12.4f}  {row["bd_std"]:10.4f}  '
              f'{row["bd_var"]:12.6f}  {row["n"]:7d}')

    print()
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Table 2: Adaptive network — ρ_b and ⟨k⟩ vs μ
# ══════════════════════════════════════════════════════════════════════════════

def table2_adaptive():
    print('Table 2: Adaptive network phase diagram')
    print('=' * 90)
    data_int = collect_experiment(SNAP / '3_dyn_graph_phase')
    data_frac = collect_experiment(SNAP / '4_first_order_probe')
    data = data_int + data_frac
    groups = group_by_mu(data)

    header = (f'{"μ":>8}  {"ρ_b (mean)":>12}  {"± std":>10}  '
              f'{"var(ρ_b)":>12}  {"⟨k⟩ (mean)":>12}  {"± std":>10}  {"n":>4}')
    print(header)
    print('-' * 90)

    rows = []
    for mu, seeds in groups:
        bds = [s['bd_mean'] for s in seeds]
        degs = [s['deg_mean'] for s in seeds]
        row = {
            'mu': mu,
            'bd_mean': np.mean(bds),
            'bd_std': np.std(bds),
            'bd_var': np.var(bds),
            'deg_mean': np.mean(degs),
            'deg_std': np.std(degs),
            'n': len(seeds),
        }
        rows.append(row)
        print(f'{mu:8.4f}  {row["bd_mean"]:12.4f}  {row["bd_std"]:10.4f}  '
              f'{row["bd_var"]:12.6f}  {row["deg_mean"]:12.2f}  '
              f'{row["deg_std"]:10.2f}  {row["n"]:4d}')

    print()
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Table 3: Hysteresis — random vs ordered init
# ══════════════════════════════════════════════════════════════════════════════

def table3_hysteresis():
    print('Table 3: Initial-condition dependence (hysteresis)')
    print('=' * 80)
    hyst_dir = SNAP / '5_hysteresis'

    header = (f'{"μ":>8}  {"Init":>8}  {"ρ_b (mean)":>12}  {"± std":>10}  '
              f'{"⟨k⟩ (mean)":>12}  {"n":>4}')
    print(header)
    print('-' * 80)

    rows = []
    for mu in [0.34, 0.345, 0.35, 0.355, 0.36]:
        mu_tag = f'{mu}'[2:].replace('.', '')  # 0.345 → '345', 0.34 → '34'
        for init_type in ['random', 'ordered']:
            data = collect_experiment(hyst_dir / f'0{mu_tag}_{init_type}')
            bds = [s['bd_mean'] for s in data]
            degs = [s['deg_mean'] for s in data]
            row = {
                'mu': mu,
                'init': init_type,
                'bd_mean': np.mean(bds),
                'bd_std': np.std(bds),
                'deg_mean': np.mean(degs),
                'deg_std': np.std(degs),
                'n': len(data),
            }
            rows.append(row)
            print(f'{mu:8.3f}  {init_type:>8}  {row["bd_mean"]:12.4f}  '
                  f'{row["bd_std"]:10.4f}  {row["deg_mean"]:12.2f}  {row["n"]:4d}')
        print()

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Table 4: FSS at μ=0.35
# ══════════════════════════════════════════════════════════════════════════════

def table4_fss():
    print('Table 4: Finite-size scaling at μ = 0.35')
    print('=' * 85)
    fss_dir = SNAP / '7_fss_035'

    header = (f'{"L":>6}  {"Init":>8}  {"ρ_b (mean)":>12}  {"± std":>10}  '
              f'{"⟨k⟩ (mean)":>12}  {"Combined var":>14}  {"n":>4}')
    print(header)
    print('-' * 85)

    rows = []
    for L in [64, 128, 256]:
        all_bds = []
        for init_type in ['random', 'ordered']:
            data = collect_experiment(fss_dir / f'L{L}_{init_type}')
            bds = [s['bd_mean'] for s in data]
            degs = [s['deg_mean'] for s in data]
            all_bds.extend(bds)
            row = {
                'L': L,
                'init': init_type,
                'bd_mean': np.mean(bds),
                'bd_std': np.std(bds),
                'deg_mean': np.mean(degs),
                'n': len(data),
            }
            rows.append(row)
            print(f'{L:6d}  {init_type:>8}  {row["bd_mean"]:12.4f}  '
                  f'{row["bd_std"]:10.4f}  {row["deg_mean"]:12.2f}  '
                  f'{"":>14}  {row["n"]:4d}')

        combined_var = np.var(all_bds)
        print(f'{"":>6}  {"":>8}  {"":>12}  {"":>10}  {"":>12}  '
              f'{combined_var:14.2e}')
        print()

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Table 5: Per-seed ρ_b at μ=0.35 (bimodality evidence)
# ══════════════════════════════════════════════════════════════════════════════

def table5_bimodality():
    print('Table 5: Per-seed boundary density at μ = 0.35 (bimodality)')
    print('=' * 55)
    data = collect_experiment(SNAP / '4_first_order_probe')
    seeds_035 = sorted([r for r in data if abs(r['mu'] - 0.35) < 0.005],
                       key=lambda r: r['seed'])

    header = f'{"Seed":>6}  {"ρ_b":>10}  {"⟨k⟩":>10}  {"Basin":>12}'
    print(header)
    print('-' * 55)

    for s in seeds_035:
        basin = 'ordered' if s['bd_mean'] < 0.28 else 'escaped'
        print(f'{s["seed"]:6d}  {s["bd_mean"]:10.4f}  {s["deg_mean"]:10.2f}  {basin:>12}')

    bds = [s['bd_mean'] for s in seeds_035]
    n_ord = sum(1 for b in bds if b < 0.3)
    n_dis = sum(1 for b in bds if b >= 0.3)
    print('-' * 55)
    print(f'  {n_ord} seeds in ordered basin,  {n_dis} in disordered basin')
    print(f'  Seed-to-seed var(ρ_b) = {np.var(bds):.4e}')
    print()

    return seeds_035


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX output
# ══════════════════════════════════════════════════════════════════════════════

def write_latex(t1, t2, t3, t4, t5):
    path = FIGDIR / 'tables.tex'
    with open(path, 'w') as f:
        # Table 1: Fixed lattice
        f.write('% Table 1: Fixed lattice phase diagram\n')
        f.write('\\begin{table}[t]\n\\centering\n')
        f.write('\\caption{Fixed lattice phase diagram ($128{\\times}128$, '
                '$r_{\\min}=4$, 64 seeds, 50k frames).}\n')
        f.write('\\label{tab:fixed}\n')
        f.write('\\begin{tabular}{rrrr}\n\\toprule\n')
        f.write('$\\mu$ & $\\bar{\\rho}_b$ & $\\sigma(\\rho_b)$ & '
                '$\\mathrm{var}(\\rho_b)$ \\\\\n\\midrule\n')
        for r in t1:
            f.write(f'{r["mu"]:.4f} & {r["bd_mean"]:.4f} & '
                    f'{r["bd_std"]:.4f} & {r["bd_var"]:.2e} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')

        # Table 2: Adaptive network
        f.write('% Table 2: Adaptive network phase diagram\n')
        f.write('\\begin{table}[t]\n\\centering\n')
        f.write('\\caption{Adaptive network phase diagram ($128{\\times}128$, '
                '$r_{\\min}=4$, $k_{\\max}=8$, 64 seeds, 30k frames).}\n')
        f.write('\\label{tab:adaptive}\n')
        f.write('\\begin{tabular}{rrrrr}\n\\toprule\n')
        f.write('$\\mu$ & $\\bar{\\rho}_b$ & $\\sigma(\\rho_b)$ & '
                '$\\mathrm{var}(\\rho_b)$ & $\\langle k \\rangle$ \\\\\n\\midrule\n')
        for r in t2:
            f.write(f'{r["mu"]:.4f} & {r["bd_mean"]:.4f} & '
                    f'{r["bd_std"]:.4f} & {r["bd_var"]:.2e} & '
                    f'{r["deg_mean"]:.2f} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')

        # Table 3: Hysteresis
        f.write('% Table 3: Initial-condition dependence\n')
        f.write('\\begin{table}[t]\n\\centering\n')
        f.write('\\caption{Initial-condition dependence across the bistable window '
                '($128{\\times}128$, 64 seeds, 30k frames).}\n')
        f.write('\\label{tab:hysteresis}\n')
        f.write('\\begin{tabular}{rlrrr}\n\\toprule\n')
        f.write('$\\mu$ & Init & $\\bar{\\rho}_b$ & $\\sigma(\\rho_b)$ & '
                '$\\langle k \\rangle$ \\\\\n\\midrule\n')
        prev_mu = None
        for r in t3:
            mu_str = f'{r["mu"]:.3f}' if r["mu"] != prev_mu else ''
            f.write(f'{mu_str} & {r["init"]} & {r["bd_mean"]:.4f} & '
                    f'{r["bd_std"]:.4f} & {r["deg_mean"]:.2f} \\\\\n')
            if r["init"] == "ordered" and r["mu"] != 0.36:
                f.write('\\addlinespace\n')
            prev_mu = r["mu"]
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')

        # Table 4: FSS
        f.write('% Table 4: Finite-size scaling at μ=0.35\n')
        f.write('\\begin{table}[t]\n\\centering\n')
        f.write('\\caption{Finite-size scaling at $\\mu = 0.35$ '
                '(32 seeds per condition).}\n')
        f.write('\\label{tab:fss}\n')
        f.write('\\begin{tabular}{rlrrr}\n\\toprule\n')
        f.write('$L$ & Init & $\\bar{\\rho}_b$ & $\\sigma(\\rho_b)$ & '
                '$\\langle k \\rangle$ \\\\\n\\midrule\n')
        prev_L = None
        for r in t4:
            L_str = str(r["L"]) if r["L"] != prev_L else ''
            f.write(f'{L_str} & {r["init"]} & {r["bd_mean"]:.4f} & '
                    f'{r["bd_std"]:.4f} & {r["deg_mean"]:.2f} \\\\\n')
            if r["init"] == "ordered":
                f.write('\\addlinespace\n')
            prev_L = r["L"]
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')

        # Table 5: Per-seed bimodality
        f.write('% Table 5: Per-seed bimodality at μ=0.35\n')
        f.write('\\begin{table}[t]\n\\centering\n')
        f.write('\\caption{Per-seed boundary density at $\\mu = 0.35$ '
                '($128{\\times}128$, 64 seeds, 30k frames). '
                'Most seeds cluster near $\\rho_b \\approx 0.16$; '
                'a few reach $\\rho_b > 0.3$ (classified as '
                '``escaped\'\').}\n')
        f.write('\\label{tab:bimodality}\n')
        f.write('\\begin{tabular}{rrrr}\n\\toprule\n')
        f.write('Seed & $\\rho_b$ & $\\langle k \\rangle$ & Basin \\\\\n\\midrule\n')
        for s in t5:
            basin = 'ordered' if s['bd_mean'] < 0.28 else 'escaped'
            f.write(f'{s["seed"]} & {s["bd_mean"]:.4f} & '
                    f'{s["deg_mean"]:.2f} & {basin} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    print(f'LaTeX tables written to {path}')


# ══════════════════════════════════════════════════════════════════════════════

def main():
    t1 = table1_fixed()
    t2 = table2_adaptive()
    t3 = table3_hysteresis()
    t4 = table4_fss()
    t5 = table5_bimodality()
    write_latex(t1, t2, t3, t4, t5)


if __name__ == '__main__':
    main()
