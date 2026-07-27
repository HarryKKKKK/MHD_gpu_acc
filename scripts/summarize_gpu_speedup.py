#!/usr/bin/env python3
"""Summarize paired OpenMP CPU and GPU final_comparison Slurm arrays."""

import argparse
import csv
import glob
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_CASES = ("shock_bubble", "blast_wave")
EXPECTED_SOLVERS = ("hll", "hllc", "force")
# A same-size GPU speedup requires a CPU timing at the same n.  OpenMP omits
# n=8, so the paired speedup summary covers n={1,2,4}; the raw GPU CSV still
# retains its n=8 timing row.
EXPECTED_SCALES = (1, 2, 4)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine timing/final_comparison CPU and GPU job-array CSVs and "
            "calculate GPU speedup = median(CPU elapsed) / median(GPU elapsed)."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("timing/final_comparison"))
    parser.add_argument("--cpu-job", required=True, help="OpenMP CPU Slurm array job ID")
    parser.add_argument("--gpu-job", required=True, help="GPU Slurm array job ID")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_backend_rows(root, job_id, backend):
    pattern = str(root / f"{job_id}_*" / backend / f"{backend}_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(f"No {backend} CSV files matched: {pattern}")

    grouped = defaultdict(list)
    for csv_path in paths:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("exit_status") != "0":
                    continue
                try:
                    key = (row["case"], row["solver"], int(row["n"]))
                    elapsed = float(row["app_elapsed_s"])
                    throughput = float(row["Mcell_updates_s"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"Invalid timing row in {csv_path}: {row}") from exc
                if not math.isfinite(elapsed) or elapsed <= 0.0:
                    raise RuntimeError(f"Non-positive elapsed time in {csv_path}: {row}")
                if not math.isfinite(throughput) or throughput <= 0.0:
                    raise RuntimeError(f"Non-positive throughput in {csv_path}: {row}")
                grouped[key].append(row)
    return grouped


def median_field(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def main():
    args = parse_args()
    cpu = read_backend_rows(args.root, args.cpu_job, "cpu_openmp")
    gpu = read_backend_rows(args.root, args.gpu_job, "gpu")

    expected = {
        (case, solver, n)
        for case in EXPECTED_CASES
        for solver in EXPECTED_SOLVERS
        for n in EXPECTED_SCALES
    }
    missing_cpu = sorted(expected - cpu.keys())
    missing_gpu = sorted(expected - gpu.keys())
    if missing_cpu or missing_gpu:
        if missing_cpu:
            print(f"[ERROR] Missing successful CPU configurations: {missing_cpu}", file=sys.stderr)
        if missing_gpu:
            print(f"[ERROR] Missing successful GPU configurations: {missing_gpu}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "solver",
        "n",
        "cpu_job_id",
        "gpu_job_id",
        "cpu_samples",
        "gpu_samples",
        "cpu_median_elapsed_s",
        "gpu_median_elapsed_s",
        "gpu_speedup_vs_cpu",
        "cpu_median_Mcell_updates_s",
        "gpu_median_Mcell_updates_s",
        "throughput_ratio_gpu_vs_cpu",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in EXPECTED_CASES:
            for solver in EXPECTED_SOLVERS:
                for n in EXPECTED_SCALES:
                    key = (case, solver, n)
                    cpu_elapsed = median_field(cpu[key], "app_elapsed_s")
                    gpu_elapsed = median_field(gpu[key], "app_elapsed_s")
                    cpu_rate = median_field(cpu[key], "Mcell_updates_s")
                    gpu_rate = median_field(gpu[key], "Mcell_updates_s")
                    writer.writerow(
                        {
                            "case": case,
                            "solver": solver,
                            "n": n,
                            "cpu_job_id": args.cpu_job,
                            "gpu_job_id": args.gpu_job,
                            "cpu_samples": len(cpu[key]),
                            "gpu_samples": len(gpu[key]),
                            "cpu_median_elapsed_s": f"{cpu_elapsed:.9f}",
                            "gpu_median_elapsed_s": f"{gpu_elapsed:.9f}",
                            "gpu_speedup_vs_cpu": f"{cpu_elapsed / gpu_elapsed:.9f}",
                            "cpu_median_Mcell_updates_s": f"{cpu_rate:.9f}",
                            "gpu_median_Mcell_updates_s": f"{gpu_rate:.9f}",
                            "throughput_ratio_gpu_vs_cpu": f"{gpu_rate / cpu_rate:.9f}",
                        }
                    )

    print(f"Wrote {len(expected)} speedup rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
