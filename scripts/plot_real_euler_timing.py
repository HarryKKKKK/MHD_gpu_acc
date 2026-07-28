#!/usr/bin/env python3
"""Plot the pure-Euler OpenMP, MPI, and A100 runtime matrix."""

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "timing" / "final_comparison" / "real_euler_timing_summary.csv"
OUTPUT = (
    ROOT
    / "paper"
    / "thesis-src"
    / "Figs"
    / "203090"
    / "performance_euler_backend_times"
)

CASES = ("shock_bubble", "blast_wave")
SOLVERS = ("hll", "hllc", "force")
CASE_LABELS = {
    "shock_bubble": "Shock--bubble interaction",
    "blast_wave": "Circular blast wave",
}
SOLVER_LABELS = {"hll": "HLL", "hllc": "HLLC", "force": "FORCE"}
BACKENDS = (
    (
        "openmp",
        "OpenMP (76 threads)",
        "#0072B2",
        "o",
    ),
    ("mpi", "MPI (76 ranks)", "#E69F00", "s"),
    ("gpu", "GPU (A100)", "#009E73", "^"),
)


def format_cells(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return str(value)


def load_rows():
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["n"] = int(row["n"])
        row["cells"] = int(row["nx"]) * int(row["ny"])
    return rows


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
    rows = load_rows()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.25), squeeze=False)
    for row_index, case in enumerate(CASES):
        for col_index, solver in enumerate(SOLVERS):
            axis = axes[row_index, col_index]
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["case"] == case and row["solver"] == solver
                ),
                key=lambda row: row["n"],
            )
            for prefix, label, color, marker in BACKENDS:
                points = [
                    row
                    for row in selected
                    if row[f"{prefix}_samples"] != "0"
                ]
                median = [float(row[f"{prefix}_median_wall_s"]) for row in points]
                minimum = [float(row[f"{prefix}_min_wall_s"]) for row in points]
                maximum = [float(row[f"{prefix}_max_wall_s"]) for row in points]
                axis.errorbar(
                    [row["cells"] for row in points],
                    median,
                    yerr=[
                        [m - lo for m, lo in zip(median, minimum)],
                        [hi - m for m, hi in zip(median, maximum)],
                    ],
                    color=color,
                    marker=marker,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    capsize=2,
                    elinewidth=0.7,
                    linewidth=1.35,
                    label=label,
                )
            axis.set_xscale("log", base=4)
            axis.set_yscale("log")
            cells = [row["cells"] for row in selected]
            axis.set_xticks(cells)
            axis.set_xticklabels(
                [format_cells(value) for value in cells], rotation=24, ha="right"
            )
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_title(SOLVER_LABELS[solver])
            if col_index == 0:
                axis.set_ylabel(f"{CASE_LABELS[case]}\nRuntime (s)")
            if row_index == 1:
                axis.set_xlabel("Grid cells")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.suptitle(
        "Pure-Euler runtime scaling (median wall time; range bars)",
        fontsize=9.2,
        y=0.955,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=0.9, h_pad=1.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
