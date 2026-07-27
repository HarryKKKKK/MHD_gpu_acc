#!/usr/bin/env python3
"""Summarize OpenMP, MPI, and GPU Euler runtime job arrays."""

import argparse
import csv
import glob
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


CASES = ("shock_bubble", "blast_wave")
SOLVERS = ("hll", "hllc", "force")
CPU_SCALES = (1, 2, 4)
GPU_SCALES = (1, 2, 4, 8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("timing/final_comparison"))
    parser.add_argument("--omp-job", required=True)
    parser.add_argument("--mpi-job", required=True)
    parser.add_argument("--gpu-job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_backend(root, job_id, backend):
    pattern = str(root / f"{job_id}_*" / backend / f"{backend}_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(f"No {backend} CSV files matched: {pattern}")

    grouped = defaultdict(list)
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("exit_status") != "0":
                    continue
                try:
                    key = (row["case"], row["solver"], int(row["n"]))
                    elapsed = float(row["app_elapsed_s"])
                    throughput = float(row["Mcell_updates_s"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"Invalid timing row in {path}: {row}") from exc
                if not math.isfinite(elapsed) or elapsed <= 0.0:
                    raise RuntimeError(f"Invalid elapsed time in {path}: {row}")
                if not math.isfinite(throughput) or throughput <= 0.0:
                    raise RuntimeError(f"Invalid throughput in {path}: {row}")
                grouped[key].append(row)
    return grouped


def expected_keys(scales):
    return {
        (case, solver, n)
        for case in CASES
        for solver in SOLVERS
        for n in scales
    }


def require_complete(label, rows, scales):
    missing = sorted(expected_keys(scales) - rows.keys())
    if missing:
        raise RuntimeError(f"Missing successful {label} configurations: {missing}")


def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def write_runtime_summary(path, backends):
    fields = (
        "backend",
        "case",
        "solver",
        "n",
        "samples",
        "median_elapsed_s",
        "median_Mcell_updates_s",
        "min_elapsed_s",
        "max_elapsed_s",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for backend, rows, scales in backends:
            for case in CASES:
                for solver in SOLVERS:
                    for n in scales:
                        samples = rows[(case, solver, n)]
                        elapsed = [float(row["app_elapsed_s"]) for row in samples]
                        writer.writerow(
                            {
                                "backend": backend,
                                "case": case,
                                "solver": solver,
                                "n": n,
                                "samples": len(samples),
                                "median_elapsed_s": f"{statistics.median(elapsed):.9f}",
                                "median_Mcell_updates_s": (
                                    f"{median(samples, 'Mcell_updates_s'):.9f}"
                                ),
                                "min_elapsed_s": f"{min(elapsed):.9f}",
                                "max_elapsed_s": f"{max(elapsed):.9f}",
                            }
                        )


def write_speedup_summary(path, omp, mpi, gpu):
    fields = (
        "case",
        "solver",
        "n",
        "omp_median_elapsed_s",
        "mpi_median_elapsed_s",
        "gpu_median_elapsed_s",
        "gpu_speedup_vs_omp",
        "gpu_speedup_vs_mpi",
        "mpi_speedup_vs_omp",
        "omp_median_Mcell_updates_s",
        "mpi_median_Mcell_updates_s",
        "gpu_median_Mcell_updates_s",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in CASES:
            for solver in SOLVERS:
                for n in CPU_SCALES:
                    key = (case, solver, n)
                    omp_elapsed = median(omp[key], "app_elapsed_s")
                    mpi_elapsed = median(mpi[key], "app_elapsed_s")
                    gpu_elapsed = median(gpu[key], "app_elapsed_s")
                    writer.writerow(
                        {
                            "case": case,
                            "solver": solver,
                            "n": n,
                            "omp_median_elapsed_s": f"{omp_elapsed:.9f}",
                            "mpi_median_elapsed_s": f"{mpi_elapsed:.9f}",
                            "gpu_median_elapsed_s": f"{gpu_elapsed:.9f}",
                            "gpu_speedup_vs_omp": f"{omp_elapsed / gpu_elapsed:.9f}",
                            "gpu_speedup_vs_mpi": f"{mpi_elapsed / gpu_elapsed:.9f}",
                            "mpi_speedup_vs_omp": f"{omp_elapsed / mpi_elapsed:.9f}",
                            "omp_median_Mcell_updates_s": (
                                f"{median(omp[key], 'Mcell_updates_s'):.9f}"
                            ),
                            "mpi_median_Mcell_updates_s": (
                                f"{median(mpi[key], 'Mcell_updates_s'):.9f}"
                            ),
                            "gpu_median_Mcell_updates_s": (
                                f"{median(gpu[key], 'Mcell_updates_s'):.9f}"
                            ),
                        }
                    )


def main():
    args = parse_args()
    try:
        omp = read_backend(args.root, args.omp_job, "cpu_openmp")
        mpi = read_backend(args.root, args.mpi_job, "mpi")
        gpu = read_backend(args.root, args.gpu_job, "gpu")
        require_complete("OpenMP", omp, CPU_SCALES)
        require_complete("MPI", mpi, CPU_SCALES)
        require_complete("GPU", gpu, GPU_SCALES)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = args.output_dir / "runtime_summary.csv"
    speedup_path = args.output_dir / "speedup_summary.csv"
    write_runtime_summary(
        runtime_path,
        (
            ("openmp", omp, CPU_SCALES),
            ("mpi", mpi, CPU_SCALES),
            ("gpu", gpu, GPU_SCALES),
        ),
    )
    write_speedup_summary(speedup_path, omp, mpi, gpu)
    print(f"Wrote runtime summary: {runtime_path}")
    print(f"Wrote speedup summary: {speedup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
