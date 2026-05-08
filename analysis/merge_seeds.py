"""Merge per-chunk seed CSVs from Tier 2.1 with original Binder seed data.

Finds all seed CSVs for a given L (original + tier2 chunks), concatenates
them, and writes a merged file. Then recomputes the Binder cumulant from
all available seeds.

Usage:
    python3 analysis/merge_seeds.py          # merge both L=256 and L=384
    python3 analysis/merge_seeds.py 256      # merge L=256 only
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def find_seed_csvs(L: int) -> list[Path]:
    """Find all seed CSVs for a given L, both original and tier2 chunks."""
    pattern = f"sweep_dyn_{L}x{L}_*_seeds.csv"
    files = sorted(RESULTS.glob(pattern))
    if not files:
        print(f"  No seed CSVs found for L={L} matching {pattern}")
    return files


def load_seeds(files: list[Path]) -> list[dict]:
    """Load and deduplicate seeds from multiple CSVs."""
    all_rows = []
    seen = set()
    for f in files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = (row["mutation_prob"], row["seed"])
                if key not in seen:
                    seen.add(key)
                    all_rows.append(row)
    return all_rows


def compute_binder(rows: list[dict]) -> dict[float, dict]:
    """Compute Binder cumulant U = 1 - <x^4>/(3<x^2>^2) per mu value."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        mu = float(r["mutation_prob"])
        bd = float(r["bd_mean"])
        groups[mu].append(bd)

    results = {}
    for mu in sorted(groups):
        vals = np.array(groups[mu])
        n = len(vals)
        m2 = np.mean(vals**2)
        m4 = np.mean(vals**4)
        U = 1.0 - m4 / (3.0 * m2**2) if m2 > 0 else 0.0
        results[mu] = {
            "mu": mu,
            "n_seeds": n,
            "bd_mean": np.mean(vals),
            "bd_std": np.std(vals),
            "U_binder": U,
            "m2": m2,
            "m4": m4,
        }
    return results


def main():
    sizes = [256, 384]
    if len(sys.argv) > 1:
        sizes = [int(sys.argv[1])]

    for L in sizes:
        print(f"\n=== L={L} ===")
        files = find_seed_csvs(L)
        if not files:
            continue

        print(f"  Found {len(files)} seed CSV(s):")
        for f in files:
            with open(f) as fh:
                n = sum(1 for _ in fh) - 1
            print(f"    {f.name} ({n} seeds)")

        rows = load_seeds(files)
        print(f"  Total unique (mu, seed) pairs: {len(rows)}")

        merged_path = RESULTS / f"binder_L{L}_merged_seeds.csv"
        fields = list(rows[0].keys())
        with open(merged_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (float(r["mutation_prob"]), int(r["seed"]))))
        print(f"  Merged seeds written to: {merged_path}")

        binder = compute_binder(rows)
        binder_path = RESULTS / f"binder_L{L}_merged.csv"
        with open(binder_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mu", "n_seeds", "bd_mean", "bd_std", "U_binder"])
            w.writeheader()
            for mu in sorted(binder):
                b = binder[mu]
                w.writerow({k: b[k] for k in ["mu", "n_seeds", "bd_mean", "bd_std", "U_binder"]})
        print(f"  Binder cumulant written to: {binder_path}")

        print(f"\n  Binder cumulant summary:")
        print(f"  {'mu':>6s}  {'n':>4s}  {'U_binder':>9s}  {'bd_mean':>8s}")
        for mu in sorted(binder):
            b = binder[mu]
            print(f"  {mu:6.3f}  {b['n_seeds']:4d}  {b['U_binder']:9.4f}  {b['bd_mean']:8.4f}")

        U_vals = [binder[mu]["U_binder"] for mu in sorted(binder)]
        mu_vals = sorted(binder.keys())
        i_min = np.argmin(U_vals)
        print(f"\n  U_min = {U_vals[i_min]:.4f} at mu = {mu_vals[i_min]:.3f} ({binder[mu_vals[i_min]]['n_seeds']} seeds)")


if __name__ == "__main__":
    main()
