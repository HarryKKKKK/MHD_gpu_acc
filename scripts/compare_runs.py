#!/usr/bin/env python3
"""
compare_runs.py — compare MHD field CSV outputs between two run directories.

Usage:
    python scripts/compare_runs.py <dir_A> <dir_B> [options]

Arguments:
    dir_A, dir_B   Root output directories to compare (e.g. outputs/12345 outputs/67890)

Options:
    --tol FLOAT    Absolute tolerance for pass/fail  (default: 1e-10)
    --rtol FLOAT   Relative tolerance for pass/fail  (default: 1e-6)
    --verbose      Print per-file differences, not just the summary table
    --fields LIST  Comma-separated fields to compare (default: all found)

Exit code:
    0  all differences within tolerance
    1  at least one field exceeds tolerance
    2  usage / IO error
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> np.ndarray:
    """Load a field CSV (rows = j, cols = i) as a float64 array."""
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


def find_csv_files(root: Path) -> dict[str, Path]:
    """Return {relative_path_str: absolute_path} for every .csv under root."""
    result = {}
    for p in sorted(root.rglob("*.csv")):
        rel = str(p.relative_to(root))
        result[rel] = p
    return result


def field_name_from_path(rel: str) -> str:
    """Extract the MHD field name from the filename, e.g. 'brio_wu_cpu_t100_rho.csv' → 'rho'."""
    stem = Path(rel).stem          # drop .csv
    return stem.rsplit("_", 1)[-1] # last underscore-separated token


def diff_stats(a: np.ndarray, b: np.ndarray):
    """Return (max_abs, mean_abs, max_rel, l2_abs) between two arrays."""
    if a.shape != b.shape:
        return None  # shape mismatch — can't compare
    d   = np.abs(a - b)
    ref = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(ref > 0, d / ref, 0.0)
    return {
        "max_abs":  float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel":  float(rel.max()),
        "l2_abs":   float(np.sqrt((d**2).mean())),
        "shape_a":  a.shape,
        "shape_b":  b.shape,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare MHD field CSV outputs from two run directories."
    )
    parser.add_argument("dir_a", help="First run directory  (e.g. outputs/12345)")
    parser.add_argument("dir_b", help="Second run directory (e.g. outputs/67890)")
    parser.add_argument("--tol",     type=float, default=1e-10,
                        help="Absolute tolerance for PASS/FAIL (default 1e-10)")
    parser.add_argument("--rtol",    type=float, default=1e-6,
                        help="Relative tolerance for PASS/FAIL (default 1e-6)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print a row for every file, not just failures")
    parser.add_argument("--fields",  default="",
                        help="Comma-separated fields to include (default: all)")
    args = parser.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)

    for d in (dir_a, dir_b):
        if not d.is_dir():
            print(f"[ERROR] directory not found: {d}", file=sys.stderr)
            sys.exit(2)

    field_filter = set(f.strip() for f in args.fields.split(",") if f.strip())

    files_a = find_csv_files(dir_a)
    files_b = find_csv_files(dir_b)

    only_a  = sorted(set(files_a) - set(files_b))
    only_b  = sorted(set(files_b) - set(files_a))
    common  = sorted(set(files_a) & set(files_b))

    print(f"\n{'='*70}")
    print(f"  MHD run comparison")
    print(f"  A : {dir_a}")
    print(f"  B : {dir_b}")
    print(f"{'='*70}")
    print(f"  CSV files in A only : {len(only_a)}")
    print(f"  CSV files in B only : {len(only_b)}")
    print(f"  Matched pairs       : {len(common)}")

    if only_a:
        print("\n  [ONLY IN A]")
        for f in only_a:
            print(f"    {f}")
    if only_b:
        print("\n  [ONLY IN B]")
        for f in only_b:
            print(f"    {f}")

    if not common:
        print("\n  No matching files to compare.")
        sys.exit(0)

    # Filter by field name if requested
    if field_filter:
        common = [f for f in common if field_name_from_path(f) in field_filter]
        print(f"\n  After field filter ({', '.join(sorted(field_filter))}): {len(common)} files")

    # ── per-file comparison ──────────────────────────────────────────────────
    col_w = [42, 10, 10, 10, 10, 8]   # column widths
    header = (
        f"\n  {'File':<{col_w[0]}}  "
        f"{'max|Δ|':>{col_w[1]}}  "
        f"{'mean|Δ|':>{col_w[2]}}  "
        f"{'max|Δ|/ref':>{col_w[3]}}  "
        f"{'L2(Δ)':>{col_w[4]}}  "
        f"{'result':>{col_w[5]}}"
    )
    separator = "  " + "-" * (sum(col_w) + 2 * (len(col_w) - 1))

    print(header)
    print(separator)

    n_pass = n_fail = n_shape = 0
    any_fail = False

    for rel in common:
        field = field_name_from_path(rel)

        try:
            a = load_csv(files_a[rel])
            b = load_csv(files_b[rel])
        except Exception as e:
            print(f"  [ERROR] could not load '{rel}': {e}")
            n_fail += 1
            continue

        stats = diff_stats(a, b)

        if stats is None:
            tag = "SHAPE!"
            n_shape += 1
            line = (
                f"  {rel:<{col_w[0]}}  "
                f"  {'shape A=' + str(a.shape):>{col_w[1]+col_w[2]+col_w[3]+col_w[4]+6}}"
                f"  {'vs B=' + str(b.shape)}"
                f"  {tag:>{col_w[5]}}"
            )
            print(line)
            any_fail = True
            continue

        passed = (stats["max_abs"] <= args.tol) or (stats["max_rel"] <= args.rtol)
        tag    = "PASS" if passed else "FAIL"
        if not passed:
            any_fail = True
            n_fail += 1
        else:
            n_pass += 1

        show = args.verbose or not passed
        if show:
            label = Path(rel).name           # just filename, no subdir clutter
            line = (
                f"  {label:<{col_w[0]}}  "
                f"{stats['max_abs']:>{col_w[1]}.3e}  "
                f"{stats['mean_abs']:>{col_w[2]}.3e}  "
                f"{stats['max_rel']:>{col_w[3]}.3e}  "
                f"{stats['l2_abs']:>{col_w[4]}.3e}  "
                f"{tag:>{col_w[5]}}"
            )
            print(line)

    print(separator)

    # ── summary ──────────────────────────────────────────────────────────────
    total = n_pass + n_fail + n_shape
    print(f"\n  Compared : {total} file pairs")
    print(f"  PASS     : {n_pass}")
    print(f"  FAIL     : {n_fail}")
    if n_shape:
        print(f"  SHAPE    : {n_shape}  (grids differ — expected if n differs between runs)")
    print(f"\n  Absolute tolerance : {args.tol:.2e}")
    print(f"  Relative tolerance : {args.rtol:.2e}")
    print(f"\n  Overall result     : {'PASS ✓' if not any_fail else 'FAIL ✗'}")
    print()

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
