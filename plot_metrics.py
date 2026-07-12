"""
plot_trajectorytools_metrics.py -- figures for stage-3 trajectorytools metrics.

Reads the *_tt_summary.csv, *_tt_perfly.csv, and *_tt_perframe.csv files created
by analyze_trajectorytools.py and writes publication/QC-style PNG plots.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


COLORS = {
    "blue": "#2C7FB8",
    "green": "#41AB5D",
    "orange": "#F28E2B",
    "red": "#D95F0E",
    "purple": "#756BB1",
    "gray": "#6B7280",
    "teal": "#1B9E77",
}


def smooth(y, window=300):
    y = pd.Series(y, dtype="float64")
    return y.rolling(window, center=True, min_periods=max(5, window // 20)).mean().to_numpy()


def setup_ax(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, out_dir, name, pdf=None):
    fig.tight_layout()
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return png


def bar_values(ax, labels, values, title, ylabel=None, colors=None):
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors or COLORS["blue"], width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    setup_ax(ax, title, ylabel)
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--perfly", required=True)
    ap.add_argument("--perframe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    summary_path = Path(args.summary)
    stem = summary_path.name.replace("_tt_summary.csv", "")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path).iloc[0]
    perfly = pd.read_csv(args.perfly)
    perframe = pd.read_csv(args.perframe)
    time_min = perframe["frame"] / args.fps / 60.0

    made = []
    pdf_path = out_dir / f"{stem}_metric_figures.pdf"
    with PdfPages(pdf_path) as pdf:
        # 1. Compact summary dashboard.
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        bar_values(
            axes[0, 0],
            ["mean speed", "median speed", "COM speed"],
            [summary["mean_speed"], summary["median_speed"], summary["mean_center_of_mass_speed"]],
            "Kinematics",
            "px/s",
            [COLORS["blue"], COLORS["teal"], COLORS["purple"]],
        )
        bar_values(
            axes[0, 1],
            ["NND", "IID"],
            [summary["mean_nnd"], summary["mean_iid"]],
            "Social Spacing",
            "px",
            [COLORS["green"], COLORS["orange"]],
        )
        bar_values(
            axes[1, 0],
            ["velocity pol.", "heading pol.", "nematic", "pair align"],
            [
                summary["mean_velocity_polarization"],
                summary["mean_group_heading_polarization"],
                summary["mean_group_heading_nematic_order"],
                summary["mean_pairwise_heading_alignment"],
            ],
            "Collective / Heading Order",
            "0-1 or cosine",
            [COLORS["blue"], COLORS["purple"], COLORS["teal"], COLORS["gray"]],
        )
        bar_values(
            axes[1, 1],
            ["forward", "sideways", "backward", "toward NN", "away NN"],
            [
                summary["forward_motion_frac"],
                summary["sideways_motion_frac"],
                summary["backward_motion_frac"],
                summary["facing_neighbor_frac"],
                summary["facing_away_neighbor_frac"],
            ],
            "Orientation Fractions",
            "fraction",
            [COLORS["green"], COLORS["gray"], COLORS["red"], COLORS["teal"], COLORS["orange"]],
        )
        fig.suptitle(f"{stem} trajectorytools metric dashboard", fontsize=14, fontweight="bold")
        made.append(save(fig, out_dir, f"{stem}_dashboard", pdf))

        # 2. Per-fly bars.
        metrics = [
            ("mean_speed", "Mean speed", "px/s", COLORS["blue"]),
            ("total_distance", "Total distance", "px", COLORS["blue"]),
            ("active_frac", "Active fraction", "fraction", COLORS["green"]),
            ("mean_nnd", "Nearest-neighbor distance", "px", COLORS["orange"]),
            ("heading_coverage", "Heading coverage", "fraction", COLORS["purple"]),
            ("mean_heading_motion_agree", "Heading-motion agreement", "cosine", COLORS["teal"]),
            ("forward_motion_frac", "Forward moving fraction", "fraction", COLORS["green"]),
            ("mean_facing_nearest_neighbor", "Facing nearest neighbor", "cosine", COLORS["orange"]),
        ]
        fig, axes = plt.subplots(4, 2, figsize=(13, 13))
        tracks = perfly["track"].astype(str).tolist()
        for ax, (col, title, ylabel, color) in zip(axes.flat, metrics):
            ax.bar(tracks, perfly[col], color=color)
            setup_ax(ax, title, ylabel, "track")
        fig.suptitle(f"{stem} per-fly metrics", fontsize=14, fontweight="bold")
        made.append(save(fig, out_dir, f"{stem}_perfly_bars", pdf))

        # 3. Per-frame time series.
        series = [
            ("mean_speed", "Mean Speed", "px/s", COLORS["blue"]),
            ("mean_nnd", "Nearest-Neighbor Distance", "px", COLORS["orange"]),
            ("velocity_polarization", "Velocity Polarization", "0-1", COLORS["green"]),
            ("group_heading_polarization", "Group Heading Polarization", "0-1", COLORS["purple"]),
            ("mean_heading_motion_agree_moving", "Heading-Motion Agreement", "cosine", COLORS["teal"]),
            ("mean_facing_nearest_neighbor", "Facing Nearest Neighbor", "cosine", COLORS["red"]),
        ]
        fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
        for ax, (col, title, ylabel, color) in zip(axes.flat, series):
            ax.plot(time_min, perframe[col], color=color, alpha=0.18, linewidth=0.6)
            ax.plot(time_min, smooth(perframe[col]), color=color, linewidth=1.8)
            setup_ax(ax, title, ylabel, "time (min)")
        fig.suptitle(f"{stem} per-frame time series (thin=raw, thick=rolling mean)", fontsize=14, fontweight="bold")
        made.append(save(fig, out_dir, f"{stem}_perframe_timeseries", pdf))

        # 4. Distributions.
        dist_metrics = [
            ("mean_speed", "Per-frame mean speed", "px/s", COLORS["blue"]),
            ("mean_nnd", "Per-frame nearest-neighbor distance", "px", COLORS["orange"]),
            ("velocity_polarization", "Velocity polarization", "0-1", COLORS["green"]),
            ("group_heading_polarization", "Group heading polarization", "0-1", COLORS["purple"]),
            ("mean_heading_motion_agree_moving", "Heading-motion agreement", "cosine", COLORS["teal"]),
            ("mean_facing_nearest_neighbor", "Facing nearest neighbor", "cosine", COLORS["red"]),
        ]
        fig, axes = plt.subplots(3, 2, figsize=(13, 11))
        for ax, (col, title, xlabel, color) in zip(axes.flat, dist_metrics):
            vals = perframe[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=50, color=color, alpha=0.85)
            setup_ax(ax, title, "frames", xlabel)
        fig.suptitle(f"{stem} metric distributions", fontsize=14, fontweight="bold")
        made.append(save(fig, out_dir, f"{stem}_distributions", pdf))

    made.append(pdf_path)
    print("wrote figures:")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
