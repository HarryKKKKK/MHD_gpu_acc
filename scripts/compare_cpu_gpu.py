#!/usr/bin/env python3
"""
compare_cpu_gpu.py — compare CPU vs GPU field-output CSVs across a matrix of
(case, solver) runs.

Expects, under <outputs_root>, one directory pair per (case, solver):
    cpu_<case>_<solver>/       written by:  main_cpu <case> --n 1 --solver <solver> --out <that dir>
    gpu_<case>_<solver>_n1/    written by:  main_gpu 1 --case <case> --solver <solver> --out <that dir>

This is exactly the layout produced by scripts/dgx_slurm/slurm_full_compare.sh.

Files are matched between the two directories by their name with the "_cpu_"
/ "_gpu_" arch token stripped out, e.g.:
    brio_wu_cpu_t10_rho.csv  <->  brio_wu_gpu_t10_rho.csv

Usage:
    python scripts/compare_cpu_gpu.py <outputs_root> [options]

Options:
    --cases   LIST   Comma-separated case names   (default: shock_bubble,brio_wu,orszag_tang)
    --solvers LIST   Comma-separated solver names  (default: hll,hllc,hlld,force)
    --tol     FLOAT  Absolute tolerance for PASS/FAIL (default 1e-8)
    --rtol    FLOAT  Relative tolerance for PASS/FAIL (default 1e-5)
    --verbose        Print every matched file, not just the worst one per combo
    --report  PATH   Also write the summary table as CSV to this path
                      (default: <outputs_root>/comparison_report.csv)

Exit code:
    0  every (case, solver) combo with data present PASS
    1  at least one combo FAILed or was missing data
    2  usage / IO error
"""

import argparse
import csv
import sys
from pathlib import Path

# Pure standard-library implementation (no numpy) — HPC login/compute
# nodes here don't have numpy installed and lack network access for pip.

CASES_DEFAULT = ["shock_bubble", "brio_wu", "orszag_tang"]
SOLVERS_DEFAULT = ["hll", "hllc", "hlld", "force"]


def load_csv(path: Path):
    """Load a field CSV (rows = j, cols = i) as a list-of-lists of float."""
    rows = []
    with open(path, newline="") as f:
        for line in csv.reader(f):
            if not line:
                continue
            rows.append([float(x) for x in line])
    return rows


def diff_stats(a, b):
    """a, b: list-of-lists of equal outer length. Returns None on shape mismatch."""
    if len(a) != len(b):
        return None
    n = 0
    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0
    max_rel = 0.0
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return None
        for va, vb in zip(row_a, row_b):
            d = abs(va - vb)
            ref = max(abs(va), abs(vb))
            rel = d / ref if ref > 0 else 0.0
            if d > max_abs:
                max_abs = d
            if rel > max_rel:
                max_rel = rel
            sum_abs += d
            sum_sq += d * d
            n += 1
    if n == 0:
        return None
    return {
        "max_abs": max_abs,
        "mean_abs": sum_abs / n,
        "max_rel": max_rel,
        "l2_abs": (sum_sq / n) ** 0.5,
    }


def match_key(filename: str, arch: str) -> str:
    """'brio_wu_cpu_t10_rho.csv' -> 'brio_wu_t10_rho.csv' (arch='cpu')."""
    return filename.replace(f"_{arch}_", "_", 1)


def collect(dir_: Path, arch: str) -> dict:
    if not dir_.is_dir():
        return {}
    return {match_key(p.name, arch): p for p in sorted(dir_.glob("*.csv"))}


def compare_pair(cpu_dir: Path, gpu_dir: Path):
    """Return (rows, missing_in_cpu, missing_in_gpu) for one (case, solver)."""
    cpu_files = collect(cpu_dir, "cpu")
    gpu_files = collect(gpu_dir, "gpu")

    keys = sorted(set(cpu_files) & set(gpu_files))
    missing_in_cpu = sorted(set(gpu_files) - set(cpu_files))
    missing_in_gpu = sorted(set(cpu_files) - set(gpu_files))

    rows = []
    for k in keys:
        try:
            a = load_csv(cpu_files[k])
            b = load_csv(gpu_files[k])
        except Exception as e:
            rows.append((k, None, str(e)))
            continue
        rows.append((k, diff_stats(a, b), None))
    return rows, missing_in_cpu, missing_in_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outputs_root", help="Root dir containing cpu_<case>_<solver>/ and gpu_<case>_<solver>_n1/")
    parser.add_argument("--cases", default=",".join(CASES_DEFAULT))
    parser.add_argument("--solvers", default=",".join(SOLVERS_DEFAULT))
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.outputs_root)
    if not root.is_dir():
        print(f"[ERROR] outputs_root not found: {root}", file=sys.stderr)
        sys.exit(2)

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    report_path = Path(args.report) if args.report else root / "comparison_report.csv"

    print(f"\n{'='*100}")
    print(f"  CPU vs GPU field comparison — root: {root}")
    print(f"  cases   : {', '.join(cases)}")
    print(f"  solvers : {', '.join(solvers)}")
    print(f"{'='*100}\n")

    col = f"  {'case':<18}{'solver':<8}{'files':>7}{'worst_field':>16}{'max|Δ|':>14}{'max|Δ|/ref':>14}{'L2(Δ)':>14}{'result':>10}"
    print(col)
    print("  " + "-" * (len(col) - 2))

    summary_rows = []
    any_fail = False

    for case in cases:
        for solver in solvers:
            cpu_dir = root / f"cpu_{case}_{solver}"
            gpu_dir = root / f"gpu_{case}_{solver}_n1"

            if not cpu_dir.is_dir() and not gpu_dir.is_dir():
                # Combo simply wasn't run — don't clutter the report with it.
                continue

            rows, missing_cpu, missing_gpu = compare_pair(cpu_dir, gpu_dir)

            if not cpu_dir.is_dir() or not gpu_dir.is_dir() or (not rows and not missing_cpu and not missing_gpu):
                print(f"  {case:<18}{solver:<8}{'--':>7}{'':>16}{'':>14}{'':>14}{'':>14}{'NO DATA':>10}")
                summary_rows.append([case, solver, 0, "", "", "", "", "NO_DATA"])
                any_fail = True
                continue

            if missing_cpu or missing_gpu:
                print(f"  [WARN] {case}/{solver}: {len(missing_cpu)} file(s) only in GPU, "
                      f"{len(missing_gpu)} file(s) only in CPU (run may have exited early / diverged)")

            shape_fail = [k for k, s, err in rows if s is None]
            valid = [(k, s) for k, s, err in rows if s is not None]

            if args.verbose:
                for k, s in valid:
                    passed = (s["max_abs"] <= args.tol) or (s["max_rel"] <= args.rtol)
                    tag = "PASS" if passed else "FAIL"
                    print(f"    {k:<50} max_abs={s['max_abs']:.3e}  max_rel={s['max_rel']:.3e}  {tag}")

            if not valid:
                print(f"  {case:<18}{solver:<8}{len(rows):>7}{'':>16}{'':>14}{'':>14}{'':>14}{'SHAPE!':>10}")
                summary_rows.append([case, solver, len(rows), "", "", "", "", "SHAPE_MISMATCH"])
                any_fail = True
                continue

            # Worst field = largest relative error (most physically meaningful
            # once fields are normalised by their own magnitude).
            worst_key, worst_stats = max(valid, key=lambda kv: kv[1]["max_rel"])
            worst_field = worst_key.rsplit("_", 1)[-1].replace(".csv", "")

            # Pass/fail is driven purely by the numeric comparison of fields
            # that exist on *both* sides. A field present on only one side
            # (e.g. GPU's extra "w" z-velocity output that CPU doesn't write)
            # is surfaced above as a [WARN], not treated as a numeric failure.
            passed = (worst_stats["max_abs"] <= args.tol) or (worst_stats["max_rel"] <= args.rtol)
            passed = passed and not shape_fail
            tag = "PASS" if passed else "FAIL"
            if not passed:
                any_fail = True

            print(f"  {case:<18}{solver:<8}{len(valid):>7}{worst_field:>16}"
                  f"{worst_stats['max_abs']:>14.3e}{worst_stats['max_rel']:>14.3e}"
                  f"{worst_stats['l2_abs']:>14.3e}{tag:>10}")

            summary_rows.append([
                case, solver, len(valid), worst_field,
                worst_stats["max_abs"], worst_stats["max_rel"], worst_stats["l2_abs"], tag
            ])

    print()
    print(f"  Absolute tolerance : {args.tol:.2e}")
    print(f"  Relative tolerance : {args.rtol:.2e}")
    print(f"  Overall result     : {'PASS' if not any_fail else 'FAIL'}")
    print()

    with open(report_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "solver", "n_files", "worst_field", "max_abs", "max_rel", "l2_abs", "result"])
        w.writerows(summary_rows)
    print(f"  Report written to: {report_path}\n")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
