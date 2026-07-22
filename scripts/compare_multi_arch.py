#!/usr/bin/env python3
"""
compare_multi_arch.py — compare field-output CSVs across CPU/GPU/MPI runs for
a matrix of (case, solver) combos, doing every pairwise comparison among the
archs that are actually present.

Expects, under <outputs_root>, one directory per (arch, case, solver):
    cpu_<case>_<solver>/       written by:  main_cpu <case> --n 1 --solver <solver> --out <that dir>
    gpu_<case>_<solver>_n1/    written by:  main_gpu 1 --case <case> --solver <solver> --out <that dir>
    mpi_<case>_<solver>_n1/    written by:  main_mpi <case> --n 1 --solver <solver> --out <that dir> (root rank only)

This is exactly the layout produced by scripts/dgx_slurm/slurm_all_compare.sh.
(cpu_<case>_<solver>/ has no "_n1" suffix and gpu/mpi do — matches the
existing convention from compare_cpu_gpu.py / slurm_full_compare.sh.)

Files are matched between two arch directories by stripping the "_<arch>_"
token from each side's filename, e.g.:
    brio_wu_cpu_t10_rho.csv  <->  brio_wu_mpi_t10_rho.csv   (both -> brio_wu_t10_rho.csv)

Usage:
    python scripts/compare_multi_arch.py <outputs_root> [options]

Options:
    --cases   LIST   Comma-separated case names   (default: shock_bubble,brio_wu,orszag_tang,rotor)
    --solvers LIST   Comma-separated solver names  (default: hll,hllc,hlld,force)
    --archs   LIST   Comma-separated archs to compare, pairwise (default: cpu,gpu,mpi)
    --tol     FLOAT  Absolute tolerance for PASS/FAIL (default 1e-8)
    --rtol    FLOAT  Relative tolerance for PASS/FAIL (default 1e-5)
    --verbose        Print every matched file, not just the worst one per combo
    --report  PATH   Also write the summary table as CSV to this path
                      (default: <outputs_root>/comparison_report.csv)

Exit code:
    0  every (case, solver, arch-pair) combo with data present on both sides PASS
    1  at least one combo FAILed or was missing data
    2  usage / IO error
"""

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

# Pure standard-library implementation (no numpy) — HPC login/compute
# nodes here don't have numpy installed and lack network access for pip.

CASES_DEFAULT   = ["shock_bubble", "brio_wu", "orszag_tang", "rotor"]
SOLVERS_DEFAULT = ["hll", "hllc", "hlld", "force"]
ARCHS_DEFAULT   = ["cpu", "gpu", "mpi"]

# Directory-name pattern per arch. cpu has no "_n1" suffix (pre-existing
# convention from compare_cpu_gpu.py); gpu and mpi both explicitly carry the
# --n scale factor used for the run.
DIR_PATTERN = {
    "cpu": "cpu_{case}_{solver}",
    "gpu": "gpu_{case}_{solver}_n1",
    "mpi": "mpi_{case}_{solver}_n1",
}


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
    nonfinite_values = 0
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return None
        for va, vb in zip(row_a, row_b):
            if not math.isfinite(va) or not math.isfinite(vb):
                nonfinite_values += 1
                n += 1
                continue
            d = abs(va - vb)
            ref = max(abs(va), abs(vb))
            rel = d / ref if ref > 0 else 0.0
            if not math.isfinite(d) or not math.isfinite(rel):
                nonfinite_values += 1
                n += 1
                continue
            if d > max_abs:
                max_abs = d
            if rel > max_rel:
                max_rel = rel
            sum_abs += d
            sum_sq += d * d
            n += 1
    if n == 0:
        return None
    if nonfinite_values:
        max_abs = math.inf
        max_rel = math.inf
        mean_abs = math.inf
        l2_abs = math.inf
    else:
        mean_abs = sum_abs / n
        l2_abs = (sum_sq / n) ** 0.5
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "l2_abs": l2_abs,
        "nonfinite_values": nonfinite_values,
    }


def match_key(filename: str, arch: str) -> str:
    """'brio_wu_cpu_t10_rho.csv' -> 'brio_wu_t10_rho.csv' (arch='cpu')."""
    return filename.replace(f"_{arch}_", "_", 1)


def collect(dir_: Path, arch: str) -> dict:
    if not dir_.is_dir():
        return {}
    return {match_key(p.name, arch): p for p in sorted(dir_.glob("*.csv"))}


def compare_pair(dir_a: Path, arch_a: str, dir_b: Path, arch_b: str):
    """Return (rows, missing_in_a, missing_in_b) for one (case, solver, arch pair)."""
    files_a = collect(dir_a, arch_a)
    files_b = collect(dir_b, arch_b)

    keys = sorted(set(files_a) & set(files_b))
    missing_in_a = sorted(set(files_b) - set(files_a))
    missing_in_b = sorted(set(files_a) - set(files_b))

    rows = []
    for k in keys:
        try:
            a = load_csv(files_a[k])
            b = load_csv(files_b[k])
        except Exception as e:
            rows.append((k, None, str(e)))
            continue
        rows.append((k, diff_stats(a, b), None))
    return rows, missing_in_a, missing_in_b


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outputs_root", help="Root dir containing <arch>_<case>_<solver>[_n1]/ subdirs")
    parser.add_argument("--cases", default=",".join(CASES_DEFAULT))
    parser.add_argument("--solvers", default=",".join(SOLVERS_DEFAULT))
    parser.add_argument("--archs", default=",".join(ARCHS_DEFAULT))
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--report-full", default=None,
                         help="Per-field (not just worst-field) report CSV path "
                              "(default: <outputs_root>/comparison_report_full.csv)")
    args = parser.parse_args()

    root = Path(args.outputs_root)
    if not root.is_dir():
        print(f"[ERROR] outputs_root not found: {root}", file=sys.stderr)
        sys.exit(2)

    cases   = [c.strip() for c in args.cases.split(",") if c.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    archs   = [a.strip() for a in args.archs.split(",") if a.strip()]
    for a in archs:
        if a not in DIR_PATTERN:
            print(f"[ERROR] unknown arch '{a}' (known: {sorted(DIR_PATTERN)})", file=sys.stderr)
            sys.exit(2)
    arch_pairs = list(itertools.combinations(archs, 2))
    report_path = Path(args.report) if args.report else root / "comparison_report.csv"
    report_full_path = Path(args.report_full) if args.report_full else root / "comparison_report_full.csv"

    print(f"\n{'='*100}")
    print(f"  Multi-arch field comparison — root: {root}")
    print(f"  cases   : {', '.join(cases)}")
    print(f"  solvers : {', '.join(solvers)}")
    print(f"  archs   : {', '.join(archs)}  (pairs: {', '.join(f'{a}-{b}' for a, b in arch_pairs)})")
    print(f"{'='*100}\n")

    col = (f"  {'case':<16}{'solver':<8}{'pair':<10}{'files':>7}{'worst_field':>16}"
           f"{'max|Δ|':>14}{'max|Δ|/ref':>14}{'L2(Δ)':>14}{'result':>10}")
    print(col)
    print("  " + "-" * (len(col) - 2))

    summary_rows = []
    full_rows = []
    any_fail = False

    for case in cases:
        for solver in solvers:
            dirs = {a: root / DIR_PATTERN[a].format(case=case, solver=solver) for a in archs}
            present = {a: d.is_dir() for a, d in dirs.items()}

            for arch_a, arch_b in arch_pairs:
                pair_label = f"{arch_a}-{arch_b}"
                dir_a, dir_b = dirs[arch_a], dirs[arch_b]

                if not present[arch_a] and not present[arch_b]:
                    # Combo simply wasn't run — don't clutter the report with it.
                    continue

                rows, missing_a, missing_b = compare_pair(dir_a, arch_a, dir_b, arch_b)

                if not present[arch_a] or not present[arch_b] or (not rows and not missing_a and not missing_b):
                    print(f"  {case:<16}{solver:<8}{pair_label:<10}{'--':>7}{'':>16}{'':>14}{'':>14}{'':>14}{'NO DATA':>10}")
                    summary_rows.append([case, solver, pair_label, 0, "", "", "", "", "NO_DATA"])
                    any_fail = True
                    continue

                if missing_a or missing_b:
                    print(f"  [WARN] {case}/{solver}/{pair_label}: {len(missing_a)} file(s) only in {arch_b}, "
                          f"{len(missing_b)} file(s) only in {arch_a} (run may have exited early / diverged)")
                    any_fail = True

                shape_fail = [k for k, s, err in rows if s is None]
                valid = [(k, s) for k, s, err in rows if s is not None]

                # Every matched field (not just the worst one) goes into the
                # full per-field report, regardless of --verbose.
                for k, s in valid:
                    field_passed = (s["max_abs"] <= args.tol) or (s["max_rel"] <= args.rtol)
                    field_name = k.rsplit("_", 1)[-1].replace(".csv", "")
                    full_rows.append([
                        case, solver, pair_label, field_name,
                        s["max_abs"], s["mean_abs"], s["max_rel"], s["l2_abs"],
                        "PASS" if field_passed else "FAIL"
                    ])

                if args.verbose:
                    for k, s in valid:
                        passed = (s["max_abs"] <= args.tol) or (s["max_rel"] <= args.rtol)
                        tag = "PASS" if passed else "FAIL"
                        print(f"    {k:<50} max_abs={s['max_abs']:.3e}  max_rel={s['max_rel']:.3e}  {tag}")

                if not valid:
                    print(f"  {case:<16}{solver:<8}{pair_label:<10}{len(rows):>7}{'':>16}{'':>14}{'':>14}{'':>14}{'SHAPE!':>10}")
                    summary_rows.append([case, solver, pair_label, len(rows), "", "", "", "", "SHAPE_MISMATCH"])
                    any_fail = True
                    continue

                # Worst field = largest relative error (most physically meaningful
                # once fields are normalised by their own magnitude).
                worst_key, worst_stats = max(valid, key=lambda kv: kv[1]["max_rel"])
                worst_field = worst_key.rsplit("_", 1)[-1].replace(".csv", "")

                passed = (worst_stats["max_abs"] <= args.tol) or (worst_stats["max_rel"] <= args.rtol)
                passed = passed and not shape_fail and not missing_a and not missing_b
                tag = "PASS" if passed else "FAIL"
                if not passed:
                    any_fail = True

                print(f"  {case:<16}{solver:<8}{pair_label:<10}{len(valid):>7}{worst_field:>16}"
                      f"{worst_stats['max_abs']:>14.3e}{worst_stats['max_rel']:>14.3e}"
                      f"{worst_stats['l2_abs']:>14.3e}{tag:>10}")

                summary_rows.append([
                    case, solver, pair_label, len(valid), worst_field,
                    worst_stats["max_abs"], worst_stats["max_rel"], worst_stats["l2_abs"], tag
                ])

    print()
    print(f"  Absolute tolerance : {args.tol:.2e}")
    print(f"  Relative tolerance : {args.rtol:.2e}")
    print(f"  Overall result     : {'PASS' if not any_fail else 'FAIL'}")
    print()

    with open(report_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "solver", "arch_pair", "n_files", "worst_field", "max_abs", "max_rel", "l2_abs", "result"])
        w.writerows(summary_rows)
    print(f"  Summary report (worst field per combo) written to: {report_path}")

    with open(report_full_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "solver", "arch_pair", "field", "max_abs", "mean_abs", "max_rel", "l2_abs", "result"])
        w.writerows(full_rows)
    print(f"  Full per-field report written to: {report_full_path}\n")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
