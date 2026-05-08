"""Recover per-seed bd_mean values from snapshot data.

The March L=256 sweep saved 5000-frame snapshots but not a per-seed CSV.
This script re-computes bd_mean (second-half average over snapshot frames)
for every (mu, seed) in the snapshot tree and writes a seeds CSV in the
same format the sweep script produces, so merge_seeds.py picks it up.

Usage:
    python3 analysis/recover_seeds_from_snapshots.py \\
        [snapshots/C_binder_L256_fine]
        [results/sweep_dyn_256x256_f100000_s32_RECOVERED_seeds.csv]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent.parent


def compute_bd(snap_dir: Path, frame: int) -> tuple[float, float, int] | None:
    """Return (bd, deg_mean, deg_max) at a single snapshot frame, or None if missing."""
    nodes_file = snap_dir / f"snapshot_{frame:06d}_nodes.csv"
    edges_file = snap_dir / f"snapshot_{frame:06d}_edges.csv"
    if not nodes_file.exists() or not edges_file.exists():
        return None

    dominant = {}
    degrees = []
    with open(nodes_file) as f:
        for row in csv.DictReader(f):
            dominant[int(row["node"])] = row["dominant"]
            degrees.append(int(row["degree"]))

    total = boundary = 0
    with open(edges_file) as f:
        for row in csv.DictReader(f):
            total += 1
            if dominant.get(int(row["src"])) != dominant.get(int(row["dst"])):
                boundary += 1
    bd = boundary / total if total else 0.0
    return bd, float(np.mean(degrees)), int(max(degrees))


def process_seed(snap_dir: Path) -> dict | None:
    """Return a dict with bd_mean, bd_std, bd_var, deg_mean, deg_max over the
    second half of snapshots, or None if no snapshots."""
    frames = sorted(
        int(f.stem.split("_")[1])
        for f in snap_dir.glob("snapshot_*_nodes.csv")
    )
    if not frames:
        return None

    bds = []
    degs = []
    deg_maxes = []
    for fr in frames:
        r = compute_bd(snap_dir, fr)
        if r is None:
            continue
        bd, deg_mean, deg_max = r
        bds.append(bd)
        degs.append(deg_mean)
        deg_maxes.append(deg_max)

    if not bds:
        return None

    half = len(bds) // 2
    post_bd = np.array(bds[half:])
    post_deg = np.array(degs[half:])

    return {
        "bd_mean": post_bd.mean(),
        "bd_std": post_bd.std(),
        "bd_var": post_bd.var(),
        "deg_mean": post_deg.mean(),
        "deg_max": max(deg_maxes),
    }


def parse_dir_name(name: str) -> tuple[float, int, int] | None:
    """Parse mr{mu}_rm{rm}_s{seed} -> (mu, rm, seed)."""
    try:
        parts = name.split("_")
        mu = float(parts[0][2:])
        rm = int(parts[1][2:])
        seed = int(parts[2][1:])
        return mu, rm, seed
    except (ValueError, IndexError):
        return None


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "snapshots" / "C_binder_L256_fine"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "sweep_dyn_256x256_f100000_s32_RECOVERED_seeds.csv"

    if not base.is_dir():
        print(f"Error: snapshot dir not found: {base}", file=sys.stderr)
        sys.exit(1)

    seed_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    print(f"Found {len(seed_dirs)} seed directories in {base}")

    rows = []
    by_mu = defaultdict(int)
    for i, sd in enumerate(seed_dirs):
        parsed = parse_dir_name(sd.name)
        if parsed is None:
            print(f"  skip: cannot parse {sd.name}")
            continue
        mu, rm, seed = parsed

        stats = process_seed(sd)
        if stats is None:
            print(f"  skip: {sd.name} has no snapshots")
            continue

        row = {
            "mutation_rate": mu,
            "mutation_prob": mu,
            "reinforce_min": rm,
            "seed": seed,
            **stats,
        }
        rows.append(row)
        by_mu[mu] += 1

        if (i + 1) % 32 == 0:
            print(f"  processed {i+1}/{len(seed_dirs)} seeds")

    if not rows:
        print("No rows generated", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: (r["mutation_prob"], r["seed"]))

    fields = ["mutation_rate", "mutation_prob", "reinforce_min", "seed",
              "bd_mean", "bd_std", "bd_var", "deg_mean", "deg_max"]

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out}")
    print(f"μ coverage:")
    for mu in sorted(by_mu):
        print(f"  μ={mu}: {by_mu[mu]} seeds")


if __name__ == "__main__":
    main()
