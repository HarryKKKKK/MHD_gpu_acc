#!/usr/bin/env python3
"""Compare all field CSVs in two independently produced MHD run directories.

Examples:

    python3 scripts/compare_two_runs.py outputs/job/run1 outputs/job/run2
    python3 scripts/compare_two_runs.py outputs/old/run1 outputs/new/run1 \
        --atol 0 --rtol 0 --verbose

    python3 scripts/compare_two_runs.py \
    outputs/31461372_orszag_tang_hlld_n8/run1 \
    outputs/31496485_orszag_tang_hlld_n8/run1 \
    --atol 0 \
    --rtol 0 \
    --verbose

The two directory trees must contain exactly the same relative CSV paths.
Missing files, malformed CSVs, shape differences, non-finite values, and
values outside ``atol + rtol * abs(run_a)`` fail the comparison.  Only the
Python standard library is required.
"""

import argparse
import csv
import math
import sys
from pathlib import Path


def load_field(path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if row:
                rows.append([float(value) for value in row])
    return rows


def compare_field(
    reference,
    candidate,
    atol,
    rtol,
):
    if len(reference) != len(candidate):
        return {"passed": False, "error": "row-count mismatch"}

    values = 0
    failed_values = 0
    max_abs = 0.0
    max_rel = 0.0
    sum_abs = 0.0
    sum_sq = 0.0

    for row_index, (ref_row, cand_row) in enumerate(zip(reference, candidate)):
        if len(ref_row) != len(cand_row):
            return {
                "passed": False,
                "error": f"column-count mismatch at row {row_index}",
            }

        for ref, cand in zip(ref_row, cand_row):
            values += 1
            if not math.isfinite(ref) or not math.isfinite(cand):
                failed_values += 1
                max_abs = math.inf
                max_rel = math.inf
                continue

            difference = abs(cand - ref)
            relative = difference / abs(ref) if ref != 0.0 else (
                0.0 if difference == 0.0 else math.inf
            )
            if difference > atol + rtol * abs(ref):
                failed_values += 1

            max_abs = max(max_abs, difference)
            max_rel = max(max_rel, relative)
            sum_abs += difference
            sum_sq += difference * difference

    if values == 0:
        return {"passed": False, "error": "empty CSV"}

    return {
        "passed": failed_values == 0,
        "error": "",
        "values": values,
        "failed_values": failed_values,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_abs": sum_abs / values,
        "l2_abs": math.sqrt(sum_sq / values),
    }


def collect_csvs(root):
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*.csv"))
    }


def default_report_path(run_a, run_b):
    if run_a.parent == run_b.parent:
        parent = run_a.parent
    else:
        parent = Path.cwd()
    return parent / f"comparison_{run_a.name}_vs_{run_b.name}.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path, help="reference/baseline run directory")
    parser.add_argument("run_b", type=Path, help="candidate/new run directory")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--fields", default="", help="comma-separated field names")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.atol < 0.0 or args.rtol < 0.0:
        parser.error("--atol and --rtol must be non-negative")
    for run in (args.run_a, args.run_b):
        if not run.is_dir():
            print(f"[ERROR] run directory not found: {run}", file=sys.stderr)
            return 2

    fields = {item.strip() for item in args.fields.split(",") if item.strip()}
    files_a = collect_csvs(args.run_a)
    files_b = collect_csvs(args.run_b)
    if fields:
        files_a = {
            name: path for name, path in files_a.items()
            if Path(name).stem.rsplit("_", 1)[-1] in fields
        }
        files_b = {
            name: path for name, path in files_b.items()
            if Path(name).stem.rsplit("_", 1)[-1] in fields
        }

    names_a = set(files_a)
    names_b = set(files_b)
    common = sorted(names_a & names_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    report_path = args.report or default_report_path(args.run_a, args.run_b)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    overall_pass = bool(common) and not only_a and not only_b
    report_rows = []

    print("=" * 116)
    print("MHD two-run field comparison")
    print(f"run A     : {args.run_a}")
    print(f"run B     : {args.run_b}")
    print(f"CSV pairs : {len(common)}")
    print(f"A only    : {len(only_a)}")
    print(f"B only    : {len(only_b)}")
    print(f"tolerance : abs <= {args.atol:g} + {args.rtol:g} * abs(run A)")
    print("=" * 116)

    for name in only_a:
        print(f"[FAIL] missing from run B: {name}")
        report_rows.append([name, "FAIL", "missing from run B", "", "", "", "", "", ""])
    for name in only_b:
        print(f"[FAIL] missing from run A: {name}")
        report_rows.append([name, "FAIL", "missing from run A", "", "", "", "", "", ""])
    if not common:
        print("[FAIL] no matching CSV files")

    print(
        f"{'file':<55} {'max_abs':>13} {'max_rel':>13} {'L2_abs':>13} "
        f"{'bad values':>12} {'result':>8}"
    )
    print("-" * 116)

    for name in common:
        try:
            stats = compare_field(
                load_field(files_a[name]),
                load_field(files_b[name]),
                args.atol,
                args.rtol,
            )
        except Exception as exc:
            stats = {"passed": False, "error": str(exc)}

        passed = bool(stats["passed"])
        if not passed:
            overall_pass = False
        tag = "PASS" if passed else "FAIL"
        error = str(stats.get("error", ""))
        max_abs = stats.get("max_abs", "")
        max_rel = stats.get("max_rel", "")
        mean_abs = stats.get("mean_abs", "")
        l2_abs = stats.get("l2_abs", "")
        values = stats.get("values", "")
        failed_values = stats.get("failed_values", "")
        report_rows.append(
            [
                name, tag, error, values, failed_values,
                max_abs, max_rel, mean_abs, l2_abs,
            ]
        )

        if args.verbose or not passed:
            if max_abs == "":
                numbers = f"{'--':>13} {'--':>13} {'--':>13} {'--':>12}"
            else:
                numbers = (
                    f"{float(max_abs):>13.5e} {float(max_rel):>13.5e} "
                    f"{float(l2_abs):>13.5e} {int(failed_values):>12}"
                )
            suffix = f"  {error}" if error else ""
            print(f"{name:<55} {numbers} {tag:>8}{suffix}")

    with report_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file", "result", "error", "values", "failed_values",
                "max_abs", "max_rel", "mean_abs", "l2_abs",
            ]
        )
        writer.writerows(report_rows)

    print("-" * 116)
    print(f"report : {report_path}")
    print(f"result : {'PASS - runs match' if overall_pass else 'FAIL - runs differ'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
