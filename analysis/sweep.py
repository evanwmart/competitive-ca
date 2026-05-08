#!/usr/bin/env python3
"""
sweep.py — parameter sweep over mutation_rate and/or reinforce_min.

Works with both ./torus (fixed lattice) and ./torus_dyn (dynamic graph).

Usage:
    python3 analysis/sweep.py
    python3 analysis/sweep.py --frames 10000 --seeds 3 -o sweep.png

    # dynamic graph
    python3 analysis/sweep.py --binary torus_dyn --topo-rate 0 --mutation-rates 3 4 5 6 7 8 9
"""

import argparse
import subprocess
import sys
import itertools
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT = Path(__file__).parent.parent

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MUTATION_RATES = [
    10000, 3000, 1000, 500, 250, 150, 75, 40, 20, 10,
]

DEFAULT_REINFORCE_MINS = [2, 3, 4]

# column indices in the stats CSV (0-based after stripping frame/step)
COL_BD  = 2
COL_DC  = 3
COL_MDS = 4
# histogram: cols 8..8+HIST_BINS-1
HIST_BINS   = 24
COL_HIST    = 8
# degree stats follow the histogram (torus_dyn only)
COL_DEG_MEAN = COL_HIST + HIST_BINS       # 32
COL_DEG_VAR  = COL_HIST + HIST_BINS + 1  # 33
COL_DEG_MAX  = COL_HIST + HIST_BINS + 2  # 34
MIN_COLS_DEG = COL_DEG_MEAN + 1

# ── simulation ────────────────────────────────────────────────────────────────

def run_one(binary: Path, mutation_rate: int | float, reinforce_min: int, seed: int,
            frames: int, width: int, height: int,
            topo_rate: int | None = None,
            max_degree: int | None = None,
            ordered_init: bool = False,
            local_formation: bool = False,
            stats_interval: int = 50,
            snapshot_dir: str | None = None) -> dict | None:
    """
    Run one simulation instance and return summary statistics over the second
    half of the run (to discard the transient).
    mutation_rate: integer N (1-in-N) or float probability (0.0–1.0, uses --mutation-prob).
    """
    is_prob = isinstance(mutation_rate, float)
    cmd = [
        str(binary),
        "--headless",
        "--stats-interval", str(stats_interval),
        "--frames",         str(frames),
    ]
    if is_prob:
        cmd += ["--mutation-prob", f"{mutation_rate:.8f}"]
    else:
        cmd += ["--mutation-rate", str(mutation_rate)]
    if topo_rate is not None:
        cmd += ["--topo-rate", str(topo_rate)]
    if max_degree is not None:
        cmd += ["--max-degree", str(max_degree)]
    if ordered_init:
        cmd += ["--ordered-init"]
    if local_formation:
        cmd += ["--local-formation"]
    if snapshot_dir:
        # Per-seed subdirectory: snapshot_dir/mr{rate}_rm{rm}_s{seed}[_ordered]
        tag = f"{mutation_rate}" if is_prob else str(mutation_rate)
        init_tag = "_ordered" if ordered_init else ""
        seed_dir = f"{snapshot_dir}/mr{tag}_rm{reinforce_min}_s{seed}{init_tag}"
        Path(seed_dir).mkdir(parents=True, exist_ok=True)
        cmd += ["--snapshot-dir", seed_dir]
    cmd += [str(width), str(height), "0", str(seed), str(reinforce_min)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=43200
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT  mr={mutation_rate} rm={reinforce_min} seed={seed}",
              file=sys.stderr)
        return None

    rows = []
    for line in result.stderr.splitlines():
        if line.startswith("seed=") or line.startswith("frame,"):
            continue
        parts = line.split(",")
        if len(parts) >= 8:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue

    if len(rows) < 4:
        return None

    arr  = np.array(rows)
    half = len(arr) // 2
    post = arr[half:]

    bd  = post[:, COL_BD]
    dc  = post[:, COL_DC]
    mds = post[:, COL_MDS]

    # Normalise to probability for consistent aggregation/plotting.
    eff_prob = mutation_rate if isinstance(mutation_rate, float) else 1.0 / mutation_rate
    out = {
        "mutation_rate": mutation_rate,
        "mutation_prob": eff_prob,
        "reinforce_min": reinforce_min,
        "seed":          seed,
        "bd_mean":       bd.mean(),
        "bd_std":        bd.std(),
        "bd_var":        bd.var(),
        "dc_mean":       dc.mean(),
        "mds_mean":      mds.mean(),
        "n_frames":      len(arr),
    }

    # degree stats (torus_dyn only — present when enough columns)
    if arr.shape[1] >= MIN_COLS_DEG:
        out["deg_mean"]  = post[:, COL_DEG_MEAN].mean()
        out["deg_var"]   = post[:, COL_DEG_VAR].mean()
        out["deg_max"]   = post[:, COL_DEG_MAX].max()

    return out


def run_point(args):
    binary, mr, rm, seed, frames, width, height, topo_rate, max_degree, ordered_init, local_formation, snapshot_dir, stats_interval = args
    return run_one(binary, mr, rm, seed, frames, width, height, topo_rate, max_degree,
                   ordered_init, local_formation, stats_interval=stats_interval, snapshot_dir=snapshot_dir)


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict[tuple, dict]:
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        # Key on (mutation_prob, reinforce_min) so int and float rates merge cleanly.
        groups[(round(r["mutation_prob"], 8), r["reinforce_min"])].append(r)

    agg = {}
    for key, rlist in groups.items():
        bd_means = np.array([r["bd_mean"] for r in rlist])
        bd_vars  = np.array([r["bd_var"]  for r in rlist])
        entry = {
            "mutation_rate": rlist[0]["mutation_rate"],
            "mutation_prob": key[0],
            "reinforce_min": key[1],
            "bd_mean":       bd_means.mean(),
            "bd_mean_std":   bd_means.std(),
            "bd_var_mean":   bd_vars.mean(),
            "n_seeds":       len(rlist),
        }
        if "deg_mean" in rlist[0]:
            entry["deg_mean"] = np.mean([r["deg_mean"] for r in rlist])
            entry["deg_max"]  = np.max([r["deg_max"]  for r in rlist])
        agg[key] = entry
    return agg


# ── plotting ──────────────────────────────────────────────────────────────────

def save_csv(agg: dict, path: str) -> None:
    import csv as csv_mod
    rows = sorted(agg.values(), key=lambda r: (r["reinforce_min"], r["mutation_rate"]))
    with open(path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {path}", file=sys.stderr)


def plot(agg: dict, output_path: str | None, params_label: str = "") -> None:
    reinforce_mins = sorted(set(k[1] for k in agg))
    colors = cm.tab10(np.linspace(0, 0.4, len(reinforce_mins)))
    has_degree = "deg_mean" in next(iter(agg.values()))

    n_panels = 3 if has_degree else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels),
                              tight_layout=True)

    title = "Phase diagram: boundary density vs mutation rate"
    if params_label:
        title += f"\n{params_label}"
    fig.suptitle(title, fontsize=12)

    for rm, color in zip(reinforce_mins, colors):
        pts = sorted(
            (v for v in agg.values() if v["reinforce_min"] == rm),
            key=lambda v: v["mutation_prob"]
        )
        if not pts:
            continue

        x    = np.array([p["mutation_prob"] for p in pts])
        bd   = np.array([p["bd_mean"]     for p in pts])
        bd_e = np.array([p["bd_mean_std"] for p in pts])
        bv   = np.array([p["bd_var_mean"] for p in pts])
        label = f"reinforce_min={rm}"

        axes[0].errorbar(x, bd, yerr=bd_e, fmt="o-", color=color,
                         label=label, linewidth=1.5, markersize=4, capsize=3)
        axes[1].plot(x, bv, "o-", color=color, label=label,
                     linewidth=1.5, markersize=4)

        if has_degree:
            dg = np.array([p["deg_mean"] for p in pts])
            dm = np.array([p["deg_max"]  for p in pts])
            axes[2].plot(x, dg, "o-", color=color, label=f"mean {label}",
                         linewidth=1.5, markersize=4)
            axes[2].plot(x, dm, "s--", color=color, alpha=0.5,
                         label=f"max {label}", linewidth=1, markersize=3)

    panel_meta = [
        ("mean boundary density",   "Order parameter (lower = more ordered)"),
        ("mean within-run variance","Variance peak locates the critical point"),
    ]
    if has_degree:
        panel_meta.append(("degree", "Degree evolution (mean + max)"))

    for ax, (ylabel, title_) in zip(axes, panel_meta):
        ax.set_xlabel("mutation probability  →  more disordered")
        ax.set_ylabel(ylabel)
        ax.set_title(title_)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"saved: {output_path}", file=sys.stderr)
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sweep mutation_rate and reinforce_min, plot phase diagram."
    )
    parser.add_argument("--binary", default="torus",
                        help="simulation binary (default: torus; use torus_dyn for dynamic graph)")
    parser.add_argument("--topo-rate", type=int, default=None,
                        help="topology change rate passed to torus_dyn (default: n_nodes)")
    parser.add_argument("--max-degree", type=int, default=None,
                        help="max degree cap for torus_dyn edge formation (default: unlimited)")
    parser.add_argument("--frames", type=int, default=5000,
                        help="frames per run (default 5000)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="seeds per parameter point (default 3)")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="start seed numbering from this value (default 0)")
    parser.add_argument("--width",  type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel workers (default 4)")
    parser.add_argument("--reinforce-mins", type=int, nargs="+",
                        default=DEFAULT_REINFORCE_MINS)
    parser.add_argument("--mutation-rates", type=int, nargs="+",
                        default=DEFAULT_MUTATION_RATES)
    parser.add_argument("--mutation-probs", type=float, nargs="+", default=None,
                        help="float mutation probabilities (overrides --mutation-rates)")
    parser.add_argument("--ordered-init", action="store_true", default=False,
                        help="start all nodes ordered (torus_dyn only)")
    parser.add_argument("--local-formation", action="store_true", default=False,
                        help="restrict edge formation to neighbours-of-neighbours (distance 2)")
    parser.add_argument("--stats-interval", type=int, default=50,
                        help="emit stats (and snapshots) every N frames (default 50)")
    parser.add_argument("--snapshot-dir", default=None,
                        help="dump node/edge CSV snapshots at each stats interval")
    parser.add_argument("--save-seeds", action="store_true", default=False,
                        help="save per-seed results CSV (for Binder cumulant etc.)")
    parser.add_argument("-o", "--output",
                        help="save plot to file instead of displaying")
    args = parser.parse_args()

    binary = ROOT / args.binary

    # Use float probs if provided, otherwise integer rates.
    mutation_values = args.mutation_probs if args.mutation_probs else args.mutation_rates

    jobs = [
        (binary, mr, rm, seed, args.frames, args.width, args.height,
         args.topo_rate, args.max_degree, args.ordered_init, args.local_formation,
         args.snapshot_dir, args.stats_interval)
        for mr, rm, seed in itertools.product(
            mutation_values,
            args.reinforce_mins,
            range(args.seed_offset, args.seed_offset + args.seeds),
        )
    ]
    total = len(jobs)
    dyn_label = f"  topo_rate={args.topo_rate}" if args.topo_rate is not None else ""
    print(f"binary={args.binary}{dyn_label}  "
          f"running {total} jobs  "
          f"({len(mutation_values)} rates × "
          f"{len(args.reinforce_mins)} reinforce_mins × "
          f"{args.seeds} seeds)  "
          f"workers={args.workers}",
          file=sys.stderr)

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_point, j): j for j in jobs}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r:
                results.append(r)
                j = futures[fut]
                deg_str = f"  deg={r['deg_mean']:.2f}" if "deg_mean" in r else ""
                print(f"  [{done}/{total}]  mr={j[1]:6}  rm={j[2]}  "
                      f"seed={j[3]}  bd={r['bd_mean']:.4f}  "
                      f"var={r['bd_var']:.6f}{deg_str}",
                      file=sys.stderr)
            else:
                print(f"  [{done}/{total}]  FAILED {futures[fut][1:4]}", file=sys.stderr)

    if not results:
        print("no results collected", file=sys.stderr)
        return

    agg = aggregate(results)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mv_lo = min(mutation_values)
    mv_hi = max(mutation_values)
    params_label = (
        f"{args.binary}  {args.width}×{args.height}  "
        f"frames={args.frames}  seeds={args.seeds}  "
        f"rm={args.reinforce_mins}  "
        f"mr=[{mv_lo}..{mv_hi}]  "
        f"{ts}"
    )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    bin_tag = "dyn" if "dyn" in args.binary else "fix"
    stem = f"sweep_{bin_tag}_{args.width}x{args.height}_f{args.frames}_s{args.seeds}_{ts}"

    csv_path  = results_dir / (stem + ".csv")
    plot_path = args.output or str(results_dir / (stem + ".png"))

    save_csv(agg, str(csv_path))

    # Save per-seed CSV BEFORE plotting so plotting failures do not
    # destroy hours of compute. (Learned the hard way 2026-04-12.)
    if args.save_seeds:
        import csv as csv_mod
        seeds_path = results_dir / (stem + "_seeds.csv")
        sorted_results = sorted(results, key=lambda r: (r["mutation_prob"], r["seed"]))
        fields = ["mutation_rate", "mutation_prob", "reinforce_min", "seed",
                  "bd_mean", "bd_std", "bd_var"]
        if "deg_mean" in sorted_results[0]:
            fields += ["deg_mean", "deg_max"]
        with open(seeds_path, "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted_results)
        print(f"saved: {seeds_path} ({len(sorted_results)} seeds)", file=sys.stderr)

    try:
        plot(agg, plot_path, params_label)
    except Exception as e:
        print(f"warning: plot failed ({type(e).__name__}: {e}) — CSVs were saved", file=sys.stderr)


if __name__ == "__main__":
    main()
