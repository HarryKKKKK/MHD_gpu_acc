"""Render every figure from a GPU n=1 full-output sweep.

Example:
  python3 visualization/plot_gpu_n1_all_outputs.py \
      outputs/dgx_gpu_n1_plots/295330 \
      --fig-dir figs/dgx_gpu_n1_295330
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SOLVERS = ("hll", "hllc", "hlld", "force")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all three MHD cases for every solver in a GPU n=1 sweep."
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Sweep directory containing gpu_<case>_<solver>_n1 directories.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Destination directory (default: figs/<output-root-name>).",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if not output_root.is_dir():
        raise SystemExit(f"[ERROR] Output root does not exist: {output_root}")

    fig_dir = (
        args.fig_dir.expanduser().resolve()
        if args.fig_dir is not None
        else (Path.cwd() / "figs" / output_root.name).resolve()
    )
    fig_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    python = sys.executable

    missing = []
    for case_name in ("brio_wu", "orszag_tang", "rotor"):
        for solver in SOLVERS:
            run_dir = output_root / f"gpu_{case_name}_{solver}_n1"
            if not run_dir.is_dir():
                missing.append(run_dir)
    if missing:
        joined = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(f"[ERROR] Missing run directories:\n{joined}")

    for solver in SOLVERS:
        label = solver.upper()

        brio_dir = output_root / f"gpu_brio_wu_{solver}_n1"
        run(
            [
                python,
                str(script_dir / "plot_brio_wu.py"),
                "--output-dir",
                str(brio_dir),
                "--prefix",
                "brio_wu_gpu",
                "--label",
                label,
                "--out",
                str(fig_dir / f"brio_wu_{solver}.png"),
            ]
        )

        orszag_dir = output_root / f"gpu_orszag_tang_{solver}_n1"
        run(
            [
                python,
                str(script_dir / "plot_orszag_tang.py"),
                "--output-dir",
                str(orszag_dir),
                "--prefix",
                "orszag_tang_gpu",
                "--label",
                label,
                "--out-panels",
                str(fig_dir / f"orszag_tang_{solver}_panels.png"),
                "--out-cut",
                str(fig_dir / f"orszag_tang_{solver}_pressure_cut.png"),
            ]
        )

        rotor_dir = output_root / f"gpu_rotor_{solver}_n1"
        run(
            [
                python,
                str(script_dir / "plot_rotor.py"),
                "--output-dir",
                str(rotor_dir),
                "--prefix",
                "rotor_gpu",
                "--label",
                label,
                "--out",
                str(fig_dir / f"rotor_{solver}_panels.png"),
            ]
        )

    rotor_compare = [
        python,
        str(script_dir / "plot_rotor.py"),
        "--out-compare",
        str(fig_dir / "rotor_all_solvers_mach.png"),
    ]
    for solver in SOLVERS:
        rotor_dir = output_root / f"gpu_rotor_{solver}_n1"
        rotor_compare.extend(
            [
                "--series",
                f"{solver.upper()}={rotor_dir}=rotor_gpu",
            ]
        )
    run(rotor_compare)

    figures = sorted(fig_dir.glob("*.png"))
    print(f"\n[DONE] Generated {len(figures)} figures in {fig_dir}")
    for figure in figures:
        print(f"  {figure.name}")


if __name__ == "__main__":
    main()
