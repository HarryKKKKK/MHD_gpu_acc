#!/usr/bin/env python3
"""Validate and plot the final 2D CPU/OpenMP, MPI, and GPU timing matrix."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "timing" / "final_comparison"
OUTPUT_DIR = ROOT / "paper" / "thesis-src" / "figs" / "203090"
SUMMARY_CSV = OUTPUT_DIR / "performance_timing_summary.csv"

CASES = ("orszag_tang", "rotor")
SOLVERS = ("hll", "force", "hllc", "hlld")
BACKENDS = ("cpu_openmp", "mpi", "gpu")
SCALES = (1, 2, 4, 8)

CASE_LABELS = {
    "orszag_tang": "Orszag--Tang",
    "rotor": "First rotor",
}
SOLVER_LABELS = {
    "hll": "HLL",
    "force": "FORCE",
    "hllc": "HLLC",
    "hlld": "HLLD",
}
BACKEND_LABELS = {
    "cpu_openmp": "OpenMP (76 threads)",
    "mpi": "MPI (76 ranks)",
    "gpu": "GPU (A100)",
}
BACKEND_STYLES = {
    "cpu_openmp": ("#0072B2", "o"),
    "mpi": ("#E69F00", "s"),
    "gpu": ("#009E73", "^"),
}


def read_rows():
    grouped = defaultdict(list)
    rows = []
    for path in sorted(INPUT_ROOT.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                # The archived timestamps use a comma as the fractional-second
                # separator without CSV quoting.  This shifts the trailing
                # provenance fields by two columns; the actual exit status is
                # consequently read under ``git_branch``.  Timing fields occur
                # before the malformed timestamps and are unaffected.
                if row["git_branch"] != "0":
                    raise RuntimeError(f"Unsuccessful timing row in {path}")
                key = (
                    row["case"],
                    row["solver"],
                    row["backend"],
                    int(row["n"]),
                )
                if key[0] not in CASES:
                    raise RuntimeError(f"Unexpected case {key[0]} in {path}")
                if key[1] not in SOLVERS:
                    raise RuntimeError(f"Unexpected solver {key[1]} in {path}")
                if key[2] not in BACKENDS:
                    raise RuntimeError(f"Unexpected backend {key[2]} in {path}")
                if key[3] not in SCALES:
                    raise RuntimeError(f"Unexpected resolution multiplier {key[3]}")
                grouped[key].append(row)
                rows.append(row)

    expected = {
        (case, solver, backend, n)
        for case in CASES
        for solver in SOLVERS
        for backend in BACKENDS
        for n in SCALES
    }
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))
        extra = sorted(set(grouped) - expected)
        raise RuntimeError(f"Incomplete timing matrix; missing={missing}, extra={extra}")
    if len(rows) != 192:
        raise RuntimeError(f"Expected 192 successful rows, found {len(rows)}")

    for case in CASES:
        for solver in SOLVERS:
            for n in SCALES:
                step_sets = {
                    tuple(
                        sorted(
                            {
                                int(row["steps"])
                                for row in grouped[(case, solver, backend, n)]
                            }
                        )
                    )
                    for backend in BACKENDS
                }
                if len(step_sets) != 1:
                    raise RuntimeError(
                        f"Timestep mismatch for {(case, solver, n)}: {step_sets}"
                    )
    return grouped


def mean(rows, field):
    values = [float(row[field]) for row in rows]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError(f"Invalid {field} values: {values}")
    return statistics.fmean(values)


def build_summary(grouped):
    summary = []
    for case in CASES:
        for solver in SOLVERS:
            for n in SCALES:
                backend_values = {}
                for backend in BACKENDS:
                    rows = grouped[(case, solver, backend, n)]
                    backend_values[backend] = {
                        "samples": len(rows),
                        "nx": int(rows[0]["nx"]),
                        "ny": int(rows[0]["ny"]),
                        "steps": int(rows[0]["steps"]),
                        "wall_s": mean(rows, "wall_seconds"),
                        "app_s": mean(rows, "app_elapsed_s"),
                        "mcell_s": mean(rows, "Mcell_updates_s"),
                    }
                cpu = backend_values["cpu_openmp"]
                mpi = backend_values["mpi"]
                gpu = backend_values["gpu"]
                summary.append(
                    {
                        "case": case,
                        "solver": solver,
                        "n": n,
                        "nx": cpu["nx"],
                        "ny": cpu["ny"],
                        "steps": cpu["steps"],
                        "samples_per_backend": cpu["samples"],
                        "openmp_wall_s": cpu["wall_s"],
                        "mpi_wall_s": mpi["wall_s"],
                        "gpu_wall_s": gpu["wall_s"],
                        "mpi_speedup_vs_openmp": cpu["wall_s"] / mpi["wall_s"],
                        "gpu_speedup_vs_openmp": cpu["wall_s"] / gpu["wall_s"],
                        "gpu_mcell_s": gpu["mcell_s"],
                    }
                )
    return summary


def write_summary(summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary[0])
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            formatted = dict(row)
            for field in (
                "openmp_wall_s",
                "mpi_wall_s",
                "gpu_wall_s",
                "mpi_speedup_vs_openmp",
                "gpu_speedup_vs_openmp",
                "gpu_mcell_s",
            ):
                formatted[field] = f"{row[field]:.6f}"
            writer.writerow(formatted)


def plot_backend_times(summary, case):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.5), sharex=True, sharey=True)
    for axis, solver in zip(axes.flat, SOLVERS):
        rows = sorted(
            (
                row
                for row in summary
                if row["case"] == case and row["solver"] == solver
            ),
            key=lambda row: row["n"],
        )
        for backend, field in (
            ("cpu_openmp", "openmp_wall_s"),
            ("mpi", "mpi_wall_s"),
            ("gpu", "gpu_wall_s"),
        ):
            color, marker = BACKEND_STYLES[backend]
            axis.plot(
                [row["n"] for row in rows],
                [row[field] for row in rows],
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=4.2,
                label=BACKEND_LABELS[backend],
            )
        axis.set_title(SOLVER_LABELS[solver], fontsize=10)
        axis.set_yscale("log")
        axis.set_xticks(SCALES)
        axis.grid(which="major", color="#D9D9D9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Resolution multiplier, n")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean wall time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(CASE_LABELS[case], y=0.94, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(
        OUTPUT_DIR / f"performance_{case}_backend_times.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_gpu_speedup(summary):
    colors = {
        "hll": "#0072B2",
        "force": "#009E73",
        "hllc": "#E69F00",
        "hlld": "#D55E00",
    }
    markers = {"hll": "o", "force": "s", "hllc": "^", "hlld": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), sharey=True)
    for axis, case in zip(axes, CASES):
        for solver in SOLVERS:
            rows = sorted(
                (
                    row
                    for row in summary
                    if row["case"] == case and row["solver"] == solver
                ),
                key=lambda row: row["n"],
            )
            axis.plot(
                [row["n"] for row in rows],
                [row["gpu_speedup_vs_openmp"] for row in rows],
                color=colors[solver],
                marker=markers[solver],
                linewidth=1.6,
                markersize=4.5,
                label=SOLVER_LABELS[solver],
            )
        axis.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--")
        axis.set_title(CASE_LABELS[case], fontsize=10)
        axis.set_xlabel("Resolution multiplier, n")
        axis.set_xticks(SCALES)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("End-to-end speedup, OpenMP wall / GPU wall")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUTPUT_DIR / "performance_gpu_speedup.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    grouped = read_rows()
    summary = build_summary(grouped)
    write_summary(summary)
    for case in CASES:
        plot_backend_times(summary, case)
    plot_gpu_speedup(summary)
    print(f"Validated 192 runs and wrote {len(summary)} summary rows.")
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
