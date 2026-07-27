#!/usr/bin/env python3
"""Create publication-quality runtime and GPU-speedup figures.

The script consumes the clean consolidated CSVs produced by
``scripts/consolidate_final_comparison.py``.  By default it plots only rows
marked ``is_canonical=1``: all 31-prefix MHD jobs, and the newest coherent
backend/solver jobs from the 32-prefix Euler reruns.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT / "timing" / "final_comparison" / "mhd_timing_31_all.csv",
    ROOT / "timing" / "final_comparison" / "euler_timing_32_all.csv",
)
DEFAULT_OUTPUT = ROOT / "timing" / "final_comparison" / "paper_figures"

EXPERIMENT_CASES = {
    "mhd": ("orszag_tang", "rotor"),
    "euler": ("shock_bubble", "blast_wave"),
}
SOLVERS = ("hll", "hllc", "hlld", "force")
BACKEND_ORDER = ("cpu_openmp", "mpi", "gpu")

CASE_LABELS = {
    "orszag_tang": "Orszag–Tang vortex",
    "rotor": "Rotor problem",
    "shock_bubble": "Shock–bubble interaction",
    "blast_wave": "Circular blast wave",
}
SOLVER_LABELS = {
    "hll": "HLL",
    "hllc": "HLLC",
    "hlld": "HLLD",
    "force": "FORCE",
}
BACKEND_LABELS = {
    "cpu_openmp": "OpenMP (76 threads)",
    "mpi": "MPI (76 ranks)",
    "gpu": "GPU (A100)",
}

# Okabe--Ito colour-blind-safe palette.  Architecture colour is kept fixed
# across every panel; solver colour is kept fixed in the speedup figure.
BACKEND_STYLES = {
    "cpu_openmp": {"color": "#0072B2", "marker": "o"},
    "mpi": {"color": "#E69F00", "marker": "s"},
    "gpu": {"color": "#009E73", "marker": "^"},
}
SOLVER_STYLES = {
    "hll": {"color": "#0072B2", "marker": "o"},
    "hllc": {"color": "#E69F00", "marker": "s"},
    "hlld": {"color": "#D55E00", "marker": "D"},
    "force": {"color": "#009E73", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime-vs-resolution and GPU speedup for MHD/Euler."
    )
    parser.add_argument(
        "--mhd-csv", type=Path, default=DEFAULT_INPUTS[0], help="31-prefix MHD CSV"
    )
    parser.add_argument(
        "--euler-csv",
        type=Path,
        default=DEFAULT_INPUTS[1],
        help="32-prefix Euler CSV",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metric",
        choices=("wall_seconds", "app_elapsed_s"),
        default="wall_seconds",
        help="Runtime metric; wall_seconds is the paper default.",
    )
    parser.add_argument(
        "--include-noncanonical",
        action="store_true",
        help="Pool superseded Euler reruns instead of using the newest jobs only.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def load_rows(paths: Tuple[Path, Path], include_noncanonical: bool) -> List[Dict]:
    rows = []  # type: List[Dict]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run scripts/consolidate_final_comparison.py first."
            )
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if raw["exit_status"] != "0":
                    continue
                if not include_noncanonical and raw["is_canonical"] != "1":
                    continue
                row = dict(raw)
                for field in (
                    "n",
                    "repeat",
                    "nx",
                    "ny",
                    "total_cells",
                    "steps",
                ):
                    row[field] = int(raw[field])
                for field in (
                    "wall_seconds",
                    "app_elapsed_s",
                    "Mcell_updates_s",
                ):
                    row[field] = float(raw[field])
                    if not math.isfinite(row[field]) or row[field] <= 0:
                        raise RuntimeError(f"Invalid {field} in {path}: {raw}")
                rows.append(row)
    return rows


def summarize(rows: List[Dict], metric: str) -> List[Dict]:
    grouped = defaultdict(list)  # type: Dict[Tuple, List[Dict]]
    for row in rows:
        key = (
            row["experiment"],
            row["case"],
            row["solver"],
            row["backend"],
            row["n"],
        )
        grouped[key].append(row)

    summary = []  # type: List[Dict]
    for key, observations in sorted(grouped.items()):
        values = [row[metric] for row in observations]
        nx_values = {row["nx"] for row in observations}
        ny_values = {row["ny"] for row in observations}
        cell_values = {row["total_cells"] for row in observations}
        step_values = {row["steps"] for row in observations}
        if any(len(values_) != 1 for values_ in (nx_values, ny_values, cell_values)):
            raise RuntimeError(f"Inconsistent grid metadata for {key}")
        summary.append(
            {
                "experiment": key[0],
                "case": key[1],
                "solver": key[2],
                "backend": key[3],
                "n": key[4],
                "nx": next(iter(nx_values)),
                "ny": next(iter(ny_values)),
                "total_cells": next(iter(cell_values)),
                "steps": tuple(sorted(step_values)),
                "samples": len(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        )
    validate_coverage(summary)
    return summary


def validate_coverage(summary: List[Dict]) -> None:
    keys = {
        (
            row["experiment"],
            row["case"],
            row["solver"],
            row["backend"],
            row["n"],
        )
        for row in summary
    }
    missing = []
    for experiment, cases in EXPERIMENT_CASES.items():
        for case in cases:
            for solver in SOLVERS:
                if experiment == "mhd":
                    requirements = {
                        backend: (1, 2, 4, 8) for backend in BACKEND_ORDER
                    }
                else:
                    requirements = {
                        "cpu_openmp": (1, 2, 4),
                        "gpu": (1, 2, 4, 8),
                    }
                for backend, scales in requirements.items():
                    for n in scales:
                        key = (experiment, case, solver, backend, n)
                        if key not in keys:
                            missing.append(key)
    if missing:
        raise RuntimeError(f"Incomplete canonical timing coverage: {missing}")


def clean_axis(axis) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def format_cells(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return str(value)


def select(
    summary: List[Dict],
    *,
    experiment: str,
    case: str,
    solver: str,
    backend: str,
) -> List[Dict]:
    return sorted(
        (
            row
            for row in summary
            if row["experiment"] == experiment
            and row["case"] == case
            and row["solver"] == solver
            and row["backend"] == backend
        ),
        key=lambda row: row["n"],
    )


def plot_runtime_matrix(
    summary: List[Dict], experiment: str, metric: str, output_dir: Path
) -> None:
    cases = EXPERIMENT_CASES[experiment]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.25), squeeze=False)
    for row_index, case in enumerate(cases):
        for col_index, solver in enumerate(SOLVERS):
            axis = axes[row_index, col_index]
            present_backends = [
                backend
                for backend in BACKEND_ORDER
                if select(
                    summary,
                    experiment=experiment,
                    case=case,
                    solver=solver,
                    backend=backend,
                )
            ]
            all_cells = set()
            for backend in present_backends:
                points = select(
                    summary,
                    experiment=experiment,
                    case=case,
                    solver=solver,
                    backend=backend,
                )
                x = [point["total_cells"] for point in points]
                y = [point["median"] for point in points]
                lower = [point["median"] - point["minimum"] for point in points]
                upper = [point["maximum"] - point["median"] for point in points]
                style = BACKEND_STYLES[backend]
                axis.errorbar(
                    x,
                    y,
                    yerr=[lower, upper],
                    color=style["color"],
                    marker=style["marker"],
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    capsize=2.0,
                    elinewidth=0.7,
                    label=BACKEND_LABELS[backend],
                    zorder=3,
                )
                all_cells.update(x)
            axis.set_xscale("log", base=4)
            axis.set_yscale("log")
            ticks = sorted(all_cells)
            axis.set_xticks(ticks)
            axis.set_xticklabels(
                [format_cells(value) for value in ticks], rotation=28, ha="right"
            )
            axis.set_title(SOLVER_LABELS[solver])
            clean_axis(axis)
            if col_index == 0:
                axis.set_ylabel(f"{CASE_LABELS[case]}\nRuntime (s)")
            if row_index == 1:
                axis.set_xlabel("Grid cells")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    metric_label = "end-to-end wall time" if metric == "wall_seconds" else "solver time"
    fig.suptitle(
        f"{experiment.upper()} runtime scaling ({metric_label}; median, range bars)",
        fontsize=9.2,
        y=0.955,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=0.8, h_pad=1.0)
    stem = output_dir / f"{experiment}_runtime_vs_resolution"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def matched_gpu_speedups(summary: List[Dict]) -> List[Dict]:
    lookup = {
        (
            row["experiment"],
            row["case"],
            row["solver"],
            row["backend"],
            row["n"],
        ): row
        for row in summary
    }
    speedups = []
    for experiment, cases in EXPERIMENT_CASES.items():
        for case in cases:
            for solver in SOLVERS:
                for n in (1, 2, 4, 8):
                    cpu_key = (experiment, case, solver, "cpu_openmp", n)
                    gpu_key = (experiment, case, solver, "gpu", n)
                    if cpu_key not in lookup or gpu_key not in lookup:
                        continue
                    cpu = lookup[cpu_key]
                    gpu = lookup[gpu_key]
                    if cpu["total_cells"] != gpu["total_cells"]:
                        raise RuntimeError(
                            f"CPU/GPU grid mismatch for "
                            f"{(experiment, case, solver, n)}"
                        )
                    speedups.append(
                        {
                            "experiment": experiment,
                            "case": case,
                            "solver": solver,
                            "n": n,
                            "total_cells": cpu["total_cells"],
                            "speedup": cpu["median"] / gpu["median"],
                        }
                    )
    return speedups


def plot_gpu_speedup(summary: List[Dict], output_dir: Path) -> None:
    speedups = matched_gpu_speedups(summary)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.45), squeeze=False)
    for row_index, experiment in enumerate(("mhd", "euler")):
        for col_index, case in enumerate(EXPERIMENT_CASES[experiment]):
            axis = axes[row_index, col_index]
            for solver in SOLVERS:
                points = sorted(
                    (
                        row
                        for row in speedups
                        if row["experiment"] == experiment
                        and row["case"] == case
                        and row["solver"] == solver
                    ),
                    key=lambda row: row["n"],
                )
                style = SOLVER_STYLES[solver]
                axis.plot(
                    [point["total_cells"] for point in points],
                    [point["speedup"] for point in points],
                    color=style["color"],
                    marker=style["marker"],
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    label=SOLVER_LABELS[solver],
                    zorder=3,
                )
            axis.axhline(1.0, color="#666666", linestyle="--", linewidth=0.7)
            axis.set_title(f"{experiment.upper()}: {CASE_LABELS[case]}")
            cells = sorted(
                {
                    point["total_cells"]
                    for point in speedups
                    if point["experiment"] == experiment and point["case"] == case
                }
            )
            axis.set_xscale("log", base=4)
            axis.set_xticks(cells)
            axis.set_xticklabels(
                [format_cells(value) for value in cells], rotation=24, ha="right"
            )
            axis.set_xlabel("Grid cells")
            clean_axis(axis)
            if col_index == 0:
                axis.set_ylabel("GPU speedup vs OpenMP")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "GPU acceleration across MHD and Euler workloads",
        fontsize=9.2,
        y=0.96,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.3, h_pad=1.2)
    stem = output_dir / "gpu_speedup_vs_resolution"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    rows = load_rows(
        (args.mhd_csv, args.euler_csv),
        include_noncanonical=args.include_noncanonical,
    )
    summary = summarize(rows, args.metric)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_runtime_matrix(summary, "mhd", args.metric, args.output_dir)
    plot_runtime_matrix(summary, "euler", args.metric, args.output_dir)
    plot_gpu_speedup(summary, args.output_dir)
    print(
        f"Plotted {len(rows)} successful observations into {args.output_dir} "
        f"using {args.metric}."
    )


if __name__ == "__main__":
    main()
