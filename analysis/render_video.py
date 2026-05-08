"""
render_video.py — Generate an N-phase video from snapshot directories.

Renders one column per snapshot directory: top row is the spatial
grid view, bottom row is the network view (nodes + edges).
Useful for showing a μ-sweep across the transition.

Layout (N phases)::

    +---------+---------+---------+---------+
    | grid 0  | grid 1  | grid 2  | grid 3  |
    +---------+---------+---------+---------+
    | net  0  | net  1  | net  2  | net  3  |
    +---------+---------+---------+---------+

Each panel title shows the current frame number and ρ_b
(boundary density, computed from edges).

Outputs default to the gitignored ``videos/`` directory: a bare
filename like ``--output sim_demo.mp4`` resolves to
``videos/sim_demo.mp4``. Pass an absolute or directory-qualified path
to override.

Usage::

    venv/bin/python3 analysis/render_video.py \\
        --snapshots \\
            snapshots/4_first_order_probe/mr0.33_rm4_s0 \\
            snapshots/4_first_order_probe/mr0.345_rm4_s0 \\
            snapshots/4_first_order_probe/mr0.355_rm4_s0 \\
            snapshots/4_first_order_probe/mr0.37_rm4_s0 \\
        --labels "far before" "just before" "just after" "far after" \\
        --output supplementary_video.mp4 \\
        --fps 10

Requires: matplotlib, numpy, ffmpeg on PATH (with libopenh264 encoder).
"""

VIDEO_DIR = "videos"

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection


REL_COLOR = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4", "N": "0.7"}
DOM_RGB = {"R": (1.0, 0.2, 0.2), "G": (0.2, 0.8, 0.2), "B": (0.2, 0.4, 1.0)}


def list_frames(directory):
    """Return sorted list of frame numbers that have both nodes+edges CSVs."""
    d = Path(directory)
    pattern = re.compile(r"snapshot_(\d+)_nodes\.csv$")
    frames = []
    for p in d.glob("snapshot_*_nodes.csv"):
        m = pattern.search(p.name)
        if not m:
            continue
        n = int(m.group(1))
        if (d / f"snapshot_{n:06d}_edges.csv").exists():
            frames.append(n)
    return sorted(frames)


def load_nodes(path):
    """Return dict {node_idx: (x, y, r, g, b, dominant)}."""
    nodes = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["node"])
            nodes[idx] = (
                int(row["x"]),
                int(row["y"]),
                int(row["r"]),
                int(row["g"]),
                int(row["b"]),
                row["dominant"],
            )
    return nodes


def load_edges(path):
    """Return list of (src, dst, rel) tuples."""
    edges = []
    with open(path) as f:
        for row in csv.DictReader(f):
            edges.append((int(row["src"]), int(row["dst"]), row["rel"]))
    return edges


def grid_image(nodes, W, H):
    """Build (H, W, 3) RGB image from node colours."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for x, y, r, g, b, _dom in nodes.values():
        img[y, x] = (r, g, b)
    return img


def boundary_density(edges):
    """Fraction of edges with relation 'N' (boundary)."""
    if not edges:
        return 0.0
    n_null = sum(1 for _, _, rel in edges if rel == "N")
    return n_null / len(edges)


def edge_segments(nodes, edges, W, H):
    """Return (segments, colors) for LineCollection.

    Skips edges that wrap the torus (|dx| > W/2 or |dy| > H/2)
    so they don't draw long lines across the panel.
    """
    segs = []
    cols = []
    for src, dst, rel in edges:
        if src not in nodes or dst not in nodes:
            continue
        x1, y1 = nodes[src][0], nodes[src][1]
        x2, y2 = nodes[dst][0], nodes[dst][1]
        if abs(x1 - x2) > W // 2 or abs(y1 - y2) > H // 2:
            continue  # skip wrapped edge
        segs.append([(x1, y1), (x2, y2)])
        cols.append(REL_COLOR.get(rel, "0.5"))
    return segs, cols


def node_scatter(nodes):
    """Return (xs, ys, colors) for scatter plot."""
    xs, ys, cs = [], [], []
    for x, y, _r, _g, _b, dom in nodes.values():
        xs.append(x)
        ys.append(y)
        cs.append(DOM_RGB.get(dom, (0.5, 0.5, 0.5)))
    return np.array(xs), np.array(ys), cs


def extract_mu(dirname):
    """Try to extract μ value from snapshot dir name like 'mr0.345_rm4_s0'."""
    m = re.search(r"mr([\d.]+)", Path(dirname).name)
    return m.group(1) if m else None


def render_frame(axes_grid, axes_net, frame_num, dirs, labels, W, H):
    """Populate panels for one frame.

    axes_grid: list of N grid-view axes (always present).
    axes_net:  list of N network-view axes, or None to skip.
    """
    all_axes = list(axes_grid) + (list(axes_net) if axes_net is not None else [])
    for ax in all_axes:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])

    for col, (d, label) in enumerate(zip(dirs, labels)):
        nodes = load_nodes(d / f"snapshot_{frame_num:06d}_nodes.csv")
        edges = load_edges(d / f"snapshot_{frame_num:06d}_edges.csv")
        bd = boundary_density(edges)

        ax_g = axes_grid[col]
        ax_g.imshow(grid_image(nodes, W, H), origin="lower",
                    interpolation="nearest")
        mu_str = extract_mu(d)
        title = f"{label}\n"
        if mu_str:
            title += f"$\\mu = {mu_str}$,  "
        title += f"$\\rho_b = {bd:.3f}$"
        ax_g.set_title(title, fontsize=10)

        if axes_net is not None:
            ax_n = axes_net[col]
            segs, cols = edge_segments(nodes, edges, W, H)
            if segs:
                lc = LineCollection(segs, colors=cols, linewidths=0.3,
                                    alpha=0.4)
                ax_n.add_collection(lc)
            xs, ys, ccs = node_scatter(nodes)
            ax_n.scatter(xs, ys, c=ccs, s=3, marker="s",
                         edgecolors="none", alpha=0.9)
            ax_n.set_xlim(-0.5, W - 0.5)
            ax_n.set_ylim(-0.5, H - 0.5)
            ax_n.set_aspect("equal")

    axes_grid[0].set_ylabel("grid view", fontsize=9)
    if axes_net is not None:
        axes_net[0].set_ylabel("network view", fontsize=9)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshots", nargs="+", required=True,
                   help="N snapshot directories (one per phase column)")
    p.add_argument("--labels", nargs="*", default=None,
                   help="optional label per column (defaults to μ value from dir name)")
    p.add_argument("--output", required=True,
                   help="output MP4 filename (bare names resolve under "
                        f"{VIDEO_DIR}/; pass a path with a directory to override)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--width", type=int, default=128, help="grid width L")
    p.add_argument("--height", type=int, default=128, help="grid height L")
    p.add_argument("--max-frames", type=int, default=0,
                   help="limit to first N common frames (0 = all)")
    p.add_argument("--stride", type=int, default=1,
                   help="render every Nth frame")
    p.add_argument("--no-network", action="store_true",
                   help="omit the network-view row (use for fixed-topology runs)")
    p.add_argument("--dpi", type=int, default=180,
                   help="output dpi (default 180; 120 = ~720p, 180 = ~1080p)")
    args = p.parse_args()

    dirs = [Path(s) for s in args.snapshots]
    n = len(dirs)

    if args.labels is None:
        labels = [f"$\\mu = {extract_mu(d) or '?'}$" for d in dirs]
    elif len(args.labels) != n:
        raise SystemExit(f"--labels count ({len(args.labels)}) "
                         f"must match --snapshots count ({n})")
    else:
        labels = args.labels

    out_path = Path(args.output)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = Path(VIDEO_DIR) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Find frames common to ALL directories
    all_frames = [set(list_frames(d)) for d in dirs]
    common = sorted(set.intersection(*all_frames)) if all_frames else []
    common = common[::args.stride]
    if args.max_frames > 0:
        common = common[:args.max_frames]

    if not common:
        raise SystemExit(f"No common frames across {len(dirs)} directories")

    print(f"Rendering {len(common)} frames @ {args.fps} fps "
          f"({len(common) / args.fps:.1f} s of video), "
          f"{n} phase columns")
    for d, label in zip(dirs, labels):
        print(f"  {label}:  {d}")
    print(f"  output: {out_path}")

    # Figure layout: 1 or 2 rows × N cols
    if args.no_network:
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 5.0),
                                 constrained_layout=True)
        if n == 1:
            axes = np.array([axes])
        axes_grid = axes
        axes_net = None
        suptitle = "Competitive CA — μ progression"
    else:
        fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 8.5),
                                 constrained_layout=True)
        if n == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        axes_grid = axes[0]
        axes_net = axes[1]
        suptitle = "Competitive CA on adaptive network — μ progression"
    fig.suptitle(suptitle, fontsize=14)

    def update(i):
        frame = common[i]
        render_frame(axes_grid, axes_net, frame, dirs, labels,
                     args.width, args.height)
        if i % max(1, len(common) // 20) == 0 or i == len(common) - 1:
            print(f"  frame {i+1}/{len(common)} (snapshot {frame})")
        return list(axes_grid) + (list(axes_net) if axes_net is not None else [])

    anim = FuncAnimation(fig, update, frames=len(common),
                         interval=1000 // args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, codec="libopenh264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out_path, writer=writer, dpi=args.dpi)

    plt.close(fig)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
