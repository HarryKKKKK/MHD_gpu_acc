"""
Batch-plot every case x solver combination under a single output root by
calling the existing per-case scripts in this directory (plot_brio_wu.py,
plot_orszag_tang.py, plot_shock_bubble.py, plot_rotor.py) as subprocesses.

Directory layouts:
  CPU: cpu_<case>_<solver>
  GPU: gpu_<case>_<solver>_n1
  MPI: mpi_<case>_<solver>_n1

Usage:
  python visualization/plot_all_cpu.py \
      --arch gpu \
      --root outputs/csd3_gpu_n1_plots/JOB_ID \
      --out-dir figs/csd3_gpu_n1_plots/JOB_ID \
      --cases brio_wu,orszag_tang,rotor \
      --solvers force,hll,hllc,hlld
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_CASES   = ["brio_wu", "orszag_tang", "rotor", "shock_bubble"]
ALL_SOLVERS = ["force", "hll", "hllc", "hlld"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=("cpu", "gpu", "mpi"), default="cpu",
                    help="Architecture prefix used by the output directories.")
    p.add_argument("--root",    default="outputs/203090",
                    help="Root directory containing per-case/per-solver folders.")
    p.add_argument("--out-dir", default="figs/203090",
                    help="Directory to write the generated figures into.")
    p.add_argument("--cases",   default=",".join(ALL_CASES))
    p.add_argument("--solvers", default=",".join(ALL_SOLVERS))
    return p.parse_args()


def run(cmd):
    print("  $ " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}")
    return result.returncode == 0


def plot_brio_wu(output_dir, prefix, label, out_dir, arch):
    return run([
        sys.executable, os.path.join(SCRIPT_DIR, "plot_brio_wu.py"),
        "--output-dir", output_dir,
        "--prefix", prefix,
        "--label", label,
        "--out", os.path.join(out_dir, f"brio_wu_{arch}_{label.lower()}.png"),
    ])


def plot_orszag_tang(output_dir, prefix, label, out_dir, arch):
    return run([
        sys.executable, os.path.join(SCRIPT_DIR, "plot_orszag_tang.py"),
        "--output-dir", output_dir,
        "--prefix", prefix,
        "--label", label,
        "--out-panels", os.path.join(out_dir, f"orszag_tang_{arch}_{label.lower()}_panels.png"),
        "--out-cut", os.path.join(out_dir, f"orszag_tang_{arch}_{label.lower()}_cut.png"),
    ])


def plot_rotor(output_dir, prefix, label, out_dir, arch):
    return run([
        sys.executable, os.path.join(SCRIPT_DIR, "plot_rotor.py"),
        "--output-dir", output_dir,
        "--prefix", prefix,
        "--label", label,
        "--out", os.path.join(out_dir, f"rotor_{arch}_{label.lower()}.png"),
    ])


def output_subdir(root, arch, case, solver):
    suffix = "_n1" if arch in ("gpu", "mpi") else ""
    return os.path.join(root, f"{arch}_{case}_{solver}{suffix}")


def plot_rotor_compare(root, solvers, out_dir, arch):
    """Fig. 20 style: Mach number compared across all available rotor solvers."""
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "plot_rotor.py")]
    found = 0
    for solver in solvers:
        output_dir = output_subdir(root, arch, "rotor", solver)
        if not os.path.isdir(output_dir):
            continue
        cmd += ["--series", f"{solver.upper()}={output_dir}=rotor_{arch}"]
        found += 1
    if found < 2:
        print("[skip] rotor Mach-number comparison needs >=2 solver directories")
        return False
    cmd += ["--out-compare", os.path.join(out_dir, f"rotor_{arch}_mach_compare.png")]
    return run(cmd)


def plot_shock_bubble(output_dir, prefix, label, out_dir, arch):
    return run([
        sys.executable, os.path.join(SCRIPT_DIR, "plot_shock_bubble.py"),
        "--series", f"{label}={output_dir}={prefix}",
        "--out", os.path.join(out_dir, f"shock_bubble_{arch}_{label.lower()}.png"),
    ])


DISPATCH = {
    "brio_wu":     plot_brio_wu,
    "orszag_tang": plot_orszag_tang,
    "rotor":       plot_rotor,
    "shock_bubble": plot_shock_bubble,
}


def main():
    args = parse_args()
    cases   = [c.strip() for c in args.cases.split(",") if c.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    ok, failed, missing = 0, 0, 0
    for case in cases:
        plot_fn = DISPATCH.get(case)
        if plot_fn is None:
            print(f"[skip] no plotting script registered for case '{case}'")
            continue

        for solver in solvers:
            output_dir = output_subdir(args.root, args.arch, case, solver)
            if not os.path.isdir(output_dir):
                print(f"[skip] {output_dir} not found")
                missing += 1
                continue

            prefix = f"{case}_{args.arch}"
            label  = solver.upper()
            print(f"\n=== {case} / {solver} ===")
            if plot_fn(output_dir, prefix, label, args.out_dir, args.arch):
                ok += 1
            else:
                failed += 1

        if case == "rotor":
            print("\n=== rotor / solver comparison (Fig. 20 style) ===")
            if plot_rotor_compare(args.root, solvers, args.out_dir, args.arch):
                ok += 1
            else:
                failed += 1

    print(f"\nDone: {ok} plotted, {failed} failed, {missing} missing.")


if __name__ == "__main__":
    main()
