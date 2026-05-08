"""Convergence diagnostics: rho_b vs frame at three mu values near mu_c.

Produces a 3-panel figure showing time-series at mu well below, near, and
above the pseudocritical point, plus a first-half vs second-half scatter.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_bd(snap_dir, frame):
    nodes_file = snap_dir / f"snapshot_{frame:06d}_nodes.csv"
    edges_file = snap_dir / f"snapshot_{frame:06d}_edges.csv"
    if not nodes_file.exists() or not edges_file.exists():
        return None

    dominant = {}
    with open(nodes_file) as f:
        for row in csv.DictReader(f):
            dominant[int(row["node"])] = row["dominant"]

    total = boundary = 0
    with open(edges_file) as f:
        for row in csv.DictReader(f):
            total += 1
            if dominant.get(int(row["src"])) != dominant.get(int(row["dst"])):
                boundary += 1
    return boundary / total if total else 0.0


def process_seed(snap_dir):
    snap_dir = Path(snap_dir)
    frames = sorted(
        int(f.stem.split("_")[1])
        for f in snap_dir.glob("snapshot_*_nodes.csv")
    )
    bd = [compute_bd(snap_dir, fr) for fr in frames]
    return np.array(frames), np.array(bd, dtype=float)


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("snapshots/C_binder_L256_fine")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("paper")
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    mu_values = ["0.341", "0.349", "0.355"]
    mu_labels = [
        r"$\mu=0.341$ (ordered)",
        r"$\mu=0.349$ (near $\mu_c$)",
        r"$\mu=0.355$ (disordered)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    summary = {}
    for ax, mu, label in zip(axes, mu_values, mu_labels):
        seed_dirs = sorted(base.glob(f"mr{mu}_rm4_s*"))[:n_seeds]
        if not seed_dirs:
            ax.set_title(f"No data for μ={mu}")
            continue

        all_bd = []
        for sd in seed_dirs:
            sn = sd.name.split("_s")[-1]
            frames, bd = process_seed(sd)
            ax.plot(frames / 1000, bd, alpha=0.5, linewidth=0.7)
            all_bd.append(bd)

        all_bd = np.array(all_bd)
        mean_trace = np.nanmean(all_bd, axis=0)
        ax.plot(frames / 1000, mean_trace, "k-", linewidth=2, zorder=10)

        half = all_bd.shape[1] // 2
        if half > 0:
            ax.axvline(x=frames[half] / 1000, color="gray", ls="--", lw=1)

        ax.set_xlabel("Frame (×10³)")
        ax.set_title(label, fontsize=10)

        fh = np.nanmean(all_bd[:, :half], axis=1)
        sh = np.nanmean(all_bd[:, half:], axis=1)
        summary[mu] = {
            "n_seeds": len(seed_dirs),
            "n_frames": all_bd.shape[1],
            "first_half_mean": float(np.mean(fh)),
            "second_half_mean": float(np.mean(sh)),
            "first_half_std": float(np.std(fh)),
            "second_half_std": float(np.std(sh)),
            "max_drift": float(np.max(np.abs(fh - sh))),
        }

    axes[0].set_ylabel(r"$\rho_b$")

    plt.tight_layout()
    fig.savefig(outdir / "fig_convergence.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved to {outdir}/fig_convergence.pdf and .png")
    for mu, s in summary.items():
        print(f"  μ={mu}: {s['n_seeds']} seeds, {s['n_frames']} frames, "
              f"1st-half mean={s['first_half_mean']:.4f}±{s['first_half_std']:.4f}, "
              f"2nd-half mean={s['second_half_mean']:.4f}±{s['second_half_std']:.4f}, "
              f"max_drift={s['max_drift']:.4f}")


if __name__ == "__main__":
    main()
