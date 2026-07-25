#!/usr/bin/env python3
"""Compute discrete magnetic-divergence norms from exported Bx/By CSV files.

The solver writes cell-centred fields with x increasing across columns and
y decreasing across rows.  This script uses a second-order centred difference
on interior cells, avoiding assumptions about ghost cells or physical boundary
conditions.  It scans one or more directories recursively and pairs every
``*_Bx.csv`` file with the corresponding ``*_By.csv`` file.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path


CASE_DOMAINS = {
    "kelvin_helmholtz": (0.0, 1.0, -1.0, 1.0),
    "shock_bubble": (0.0, 0.225, 0.0, 0.089),
    "brio_wu": (0.0, 1.0, 0.0, 4.0 / 800.0),
    "orszag_tang": (0.0, 2.0 * math.pi, 0.0, 2.0 * math.pi),
    "rotor": (0.0, 1.0, 0.0, 1.0),
}


def load_csv(path):
    rows = []
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if row:
                rows.append([float(value) for value in row])
    if not rows or not rows[0]:
        raise ValueError("empty field")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("ragged CSV rows")
    if len(rows) < 3 or width < 3:
        raise ValueError("at least 3 x 3 cells are required")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("non-finite field value")
    return rows


def identify_case(filename):
    for case_name in sorted(CASE_DOMAINS, key=len, reverse=True):
        if filename.startswith(case_name + "_"):
            return case_name
    return None


def parse_run_metadata(path):
    directory = path.parent.name
    match = re.match(
        r"(?P<arch>cpu|gpu|mpi)_(?P<case>.+)_(?P<solver>hllc|hlld|hll|force)(?:_n\d+)?$",
        directory,
    )
    if match:
        return match.group("arch"), match.group("solver")
    match = re.search(r"_(cpu|gpu|mpi)_", path.name)
    return (match.group(1) if match else "unknown"), "unknown"


def snapshot_name(path, case_name, arch):
    stem = path.name[:-len("_Bx.csv")]
    prefix = case_name + "_" + arch + "_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def divergence_stats(bx, by, domain):
    if len(bx) != len(by) or len(bx[0]) != len(by[0]):
        raise ValueError("Bx/By shape mismatch")

    ny, nx = len(bx), len(bx[0])
    x_min, x_max, y_min, y_max = domain
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    max_div = 0.0
    sum_div_sq = 0.0
    max_b = 0.0
    sum_b_sq = 0.0
    count = 0

    # CSV row zero is the largest-y row, hence the minus sign in dBy/dy.
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            dbx_dx = (bx[j][i + 1] - bx[j][i - 1]) / (2.0 * dx)
            dby_dy = (by[j - 1][i] - by[j + 1][i]) / (2.0 * dy)
            div_b = dbx_dx + dby_dy
            b_mag = math.hypot(bx[j][i], by[j][i])
            max_div = max(max_div, abs(div_b))
            sum_div_sq += div_b * div_b
            max_b = max(max_b, b_mag)
            sum_b_sq += b_mag * b_mag
            count += 1

    l2_div = math.sqrt(sum_div_sq / count)
    l2_b = math.sqrt(sum_b_sq / count)
    h = min(dx, dy)
    return {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "n_interior": count,
        "div_linf": max_div,
        "div_l2": l2_div,
        "h_div_linf_over_b_linf": h * max_div / max_b if max_b else 0.0,
        "h_div_l2_over_b_l2": h * l2_div / l2_b if l2_b else 0.0,
    }


def find_pairs(roots):
    seen = set()
    for root in roots:
        for bx_path in sorted(root.rglob("*_Bx.csv")):
            resolved = bx_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            by_path = bx_path.with_name(bx_path.name[:-len("_Bx.csv")] + "_By.csv")
            yield bx_path, by_path


def self_test():
    nx, ny = 7, 6
    dx, dy = 1.0 / nx, 1.0 / ny
    xs = [(i + 0.5) * dx for i in range(nx)]
    ys_top_down = [1.0 - (j + 0.5) * dy for j in range(ny)]

    # B=(y,-x) is divergence-free; B=(x,y) has divergence exactly two.
    bx_zero = [[y for _ in xs] for y in ys_top_down]
    by_zero = [[-x for x in xs] for _ in ys_top_down]
    zero = divergence_stats(bx_zero, by_zero, (0.0, 1.0, 0.0, 1.0))
    if zero["div_linf"] > 1.0e-13:
        raise AssertionError("divergence-free self-test failed")

    bx_two = [[x for x in xs] for _ in ys_top_down]
    by_two = [[y for _ in xs] for y in ys_top_down]
    two = divergence_stats(bx_two, by_two, (0.0, 1.0, 0.0, 1.0))
    if abs(two["div_linf"] - 2.0) > 1.0e-13:
        raise AssertionError("nonzero-divergence self-test failed")
    print("magnetic-divergence self-test: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, help="directories to scan recursively")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="output CSV (default: <first-root>/magnetic_divergence_report.csv)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        if not args.roots:
            return 0
    if not args.roots:
        parser.error("provide at least one output directory or use --self-test")
    for root in args.roots:
        if not root.is_dir():
            parser.error("directory not found: {}".format(root))

    report = args.report or args.roots[0] / "magnetic_divergence_report.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case", "solver", "arch", "snapshot", "nx", "ny", "dx", "dy",
        "n_interior", "div_linf", "div_l2",
        "h_div_linf_over_b_linf", "h_div_l2_over_b_l2",
    ]
    rows = []
    errors = []

    for bx_path, by_path in find_pairs(args.roots):
        case_name = identify_case(bx_path.name)
        if case_name is None:
            errors.append("{}: cannot infer case".format(bx_path))
            continue
        if not by_path.is_file():
            errors.append("{}: matching By file is missing".format(bx_path))
            continue
        try:
            bx = load_csv(bx_path)
            by = load_csv(by_path)
            stats = divergence_stats(bx, by, CASE_DOMAINS[case_name])
            arch, solver = parse_run_metadata(bx_path)
            row = {
                "case": case_name,
                "solver": solver,
                "arch": arch,
                "snapshot": snapshot_name(bx_path, case_name, arch),
                **stats,
            }
            rows.append(row)
            print(
                "{case:16s} {solver:7s} {arch:7s} {snapshot:12s} "
                "Linf={div_linf:.6e} L2={div_l2:.6e} "
                "hLinf/B={h_div_linf_over_b_linf:.6e}".format(**row)
            )
        except (OSError, ValueError) as exc:
            errors.append("{}: {}".format(bx_path, exc))

    with report.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("magnetic-divergence report: {}".format(report))
    for error in errors:
        print("[ERROR] " + error, file=sys.stderr)
    if not rows:
        print("[ERROR] no complete Bx/By pairs found", file=sys.stderr)
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
