#!/usr/bin/env python3
"""Plot MHD and pure-Euler GPU speedup against 76-thread OpenMP."""

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MHD_INPUT = (
    ROOT
    / "timing"
    / "final_comparison"
    / "paper_figures"
    / "mhd_timing_summary.csv"
)
EULER_INPUT = ROOT / "timing" / "final_comparison" / "real_euler_timing_summary.csv"
OUTPUT = (
    ROOT
    / "paper"
    / "thesis-src"
    / "Figs"
    / "203090"
    / "performance_gpu_speedup"
)

STYLES = {
    "hll": ("HLL", "#0072B2", "o"),
    "hllc": ("HLLC", "#E69F00", "s"),
    "hlld": ("HLLD", "#D55E00", "D"),
    "force": ("FORCE", "#009E73", "^"),
}
PANELS = (
    ("mhd", "orszag_tang", "MHD: Orszag--Tang vortex"),
    ("mhd", "rotor", "MHD: Rotor problem"),
    ("pure_euler", "shock_bubble", "PURE EULER: Shock--bubble interaction"),
    ("pure_euler", "blast_wave", "PURE EULER: Circular blast wave"),
)


def format_cells(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return str(value)


def load_mhd():
    with MHD_INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {}
    for row in rows:
        key = (row["case"], row["solver"], row["backend"], int(row["n"]))
        lookup[key] = row
    points = []
    for case in ("orszag_tang", "rotor"):
        for solver in STYLES:
            for n in (1, 2, 4, 8):
                omp = lookup[(case, solver, "cpu_openmp", n)]
                gpu = lookup[(case, solver, "gpu", n)]
                points.append(
                    {
                        "architecture": "mhd",
                        "case": case,
                        "solver": solver,
                        "n": n,
                        "cells": int(omp["total_cells"]),
                        "speedup": (
                            float(omp["median_wall_seconds"])
                            / float(gpu["median_wall_seconds"])
                        ),
                    }
                )
    return points


def load_euler():
    points = []
    with EULER_INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["openmp_samples"] == "0":
                continue
            points.append(
                {
                    "architecture": "pure_euler",
                    "case": row["case"],
                    "solver": row["solver"],
                    "n": int(row["n"]),
                    "cells": int(row["nx"]) * int(row["ny"]),
                    "speedup": float(row["gpu_speedup_vs_openmp"]),
                }
            )
    return points


def main():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    points = load_mhd() + load_euler()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.45), squeeze=False)
    for axis, (architecture, case, title) in zip(axes.flat, PANELS):
        panel = [
            point
            for point in points
            if point["architecture"] == architecture and point["case"] == case
        ]
        solvers = ("hll", "hllc", "hlld", "force") if architecture == "mhd" else (
            "hll",
            "hllc",
            "force",
        )
        for solver in solvers:
            selected = sorted(
                (point for point in panel if point["solver"] == solver),
                key=lambda point: point["n"],
            )
            label, color, marker = STYLES[solver]
            axis.plot(
                [point["cells"] for point in selected],
                [point["speedup"] for point in selected],
                color=color,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=0.9,
                linewidth=1.35,
                label=label,
            )
        cells = sorted({point["cells"] for point in panel})
        axis.set_xscale("log", base=4)
        axis.set_xticks(cells)
        axis.set_xticklabels(
            [format_cells(value) for value in cells], rotation=24, ha="right"
        )
        axis.axhline(1, color="#666666", linestyle="--", linewidth=0.7)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title(title)
        axis.set_xlabel("Grid cells")
        axis.set_ylabel("GPU speedup vs OpenMP")

    handles = []
    labels = []
    for solver in ("hll", "hllc", "hlld", "force"):
        label, color, marker = STYLES[solver]
        handle, = axes[0, 0].plot(
            [], [], color=color, marker=marker, markerfacecolor="white", label=label
        )
        handles.append(handle)
        labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.suptitle(
        "GPU acceleration across MHD and pure-Euler architectures",
        fontsize=9.2,
        y=0.955,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.0, h_pad=1.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
