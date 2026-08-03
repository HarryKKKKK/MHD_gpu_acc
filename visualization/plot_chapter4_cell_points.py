"""Regenerate the Chapter 4 point plots from archived n=1 solver outputs.

The script addresses two presentation requirements:

* Brio--Wu: retain density only and show every one of the 800 cell-centre
  values as points, without a connecting line.
* Orszag--Tang: show every one of the 192 cell-centre values in the pressure
  cut at the stored row nearest y=0.3125, again without a connecting line.

Typical CSD3 use
----------------

python visualization/plot_chapter4_cell_points.py \
    outputs/csd3_gpu_n1_plots/31930327 \
    --out-dir figs/chapter4_cell_points

To replace the files already referenced by the report directly:

python visualization/plot_chapter4_cell_points.py \
    outputs/csd3_gpu_n1_plots/31930327 \
    --report-root paper/report2-final-20260728-edit

The output root is expected to contain directories such as
``gpu_brio_wu_hlld_n1`` and ``gpu_orszag_tang_hlld_n1``.  A directory name
without the ``_n1`` suffix is also accepted.  ``--backend cpu`` is provided
for parity-data checks; GPU is the default.
"""

import argparse
from pathlib import Path


SOLVERS = ("force", "hll", "hllc", "hlld")
SOLVER_LABELS = {
    "force": "FORCE",
    "hll": "HLL",
    "hllc": "HLLC-L",
    "hlld": "HLLD",
}

BRIO_NX = 800
BRIO_X_MIN = 0.0
BRIO_X_MAX = 1.0

OT_NX = 192
OT_NY = 192
OT_Y_CUT = 0.3125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Chapter 4 all-cell point plots."
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Directory containing the per-case, per-solver n=1 output folders.",
    )
    parser.add_argument(
        "--backend",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Output prefix and directory family to read (default: gpu).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figs/chapter4_cell_points"),
        help="Standalone destination when --report-root is not used.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help=(
            "Optional report source directory. When set, files are written "
            "directly to the paths already referenced by Chapter 4."
        ),
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help=(
            "Figure subdirectory for the Orszag--Tang files when using "
            "--report-root (default: output-root directory name)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=350)
    return parser.parse_args()


def find_run_dir(root: Path, backend: str, case: str, solver: str) -> Path:
    candidates = (
        root / f"{backend}_{case}_{solver}_n1",
        root / f"{backend}_{case}_{solver}",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    tried = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        f"No output directory found for {case}/{solver}. Tried:\n{tried}"
    )


def load_field(
    np,
    run_dir: Path,
    case: str,
    backend: str,
    tag: str,
    field: str,
):
    path = run_dir / f"{case}_{backend}_{tag}_{field}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing field file: {path}")
    return np.loadtxt(path, delimiter=",")


def output_paths(args, solver):
    if args.report_root is None:
        destination = args.out_dir.expanduser().resolve()
        return (
            destination / f"brio_wu_density_{solver}.png",
            destination / f"orszag_tang_gpu_{solver}_cut.png",
        )

    report_root = args.report_root.expanduser().resolve()
    job_id = args.job_id or args.output_root.name
    return (
        report_root / "Figs" / "chapter4_revised" / f"brio_wu_density_{solver}.png",
        report_root
        / "Figs"
        / "csd3_gpu_n1_plots"
        / job_id
        / f"orszag_tang_gpu_{solver}_cut.png",
    )


def configure_axes(ax) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_brio(np, plt, args: argparse.Namespace, solver: str, output: Path) -> None:
    run_dir = find_run_dir(args.output_root, args.backend, "brio_wu", solver)
    raw = load_field(
        np, run_dir, "brio_wu", args.backend, "t10", "rho"
    )
    if raw.ndim != 2 or raw.shape[1] != BRIO_NX:
        raise ValueError(
            f"Expected Brio--Wu array (*,{BRIO_NX}), found {raw.shape} in {run_dir}"
        )

    # Output rows are written high-y first.  The four rows are identical for
    # this embedded one-dimensional problem; use the low-y row after flipping.
    density = raw[::-1][0]
    dx = (BRIO_X_MAX - BRIO_X_MIN) / BRIO_NX
    x = np.linspace(BRIO_X_MIN + 0.5 * dx, BRIO_X_MAX - 0.5 * dx, BRIO_NX)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(
        x,
        density,
        linestyle="none",
        marker=".",
        color="black",
        markersize=1.55,
        rasterized=True,
    )
    ax.set_xlim(BRIO_X_MIN, BRIO_X_MAX)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(f"{SOLVER_LABELS[solver]}: 800 cell-centre values", fontsize=9)
    configure_axes(ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_orszag_tang(np, plt, args, solver, output):
    run_dir = find_run_dir(args.output_root, args.backend, "orszag_tang", solver)
    raw = load_field(
        np, run_dir, "orszag_tang", args.backend, "tpi", "p"
    )
    if raw.shape != (OT_NY, OT_NX):
        raise ValueError(
            f"Expected Orszag--Tang array {(OT_NY, OT_NX)}, found {raw.shape} "
            f"in {run_dir}"
        )

    pressure = raw[::-1]
    x = (np.arange(OT_NX) + 0.5) / OT_NX
    y = (np.arange(OT_NY) + 0.5) / OT_NY
    j_cut = int(np.argmin(np.abs(y - OT_Y_CUT)))
    stored_y = float(y[j_cut])

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(
        x,
        pressure[j_cut],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="#D62728",
        markeredgewidth=0.55,
        markersize=2.15,
        rasterized=True,
        label=f"{SOLVER_LABELS[solver]}: 192 cell-centre values",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("position")
    ax.set_ylabel("gas pressure")
    ax.set_title(r"Orszag--Tang at $t=\pi$; nominal cut $y=0.3125$", fontsize=9)
    ax.legend(loc="best", fontsize=7.5, frameon=False)
    configure_axes(ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return j_cut, stored_y


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    if not args.output_root.is_dir():
        raise SystemExit(f"Output root does not exist: {args.output_root}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    for solver in SOLVERS:
        brio_output, ot_output = output_paths(args, solver)
        plot_brio(np, plt, args, solver, brio_output)
        j_cut, stored_y = plot_orszag_tang(np, plt, args, solver, ot_output)
        print(f"[{SOLVER_LABELS[solver]}] {brio_output}")
        print(
            f"[{SOLVER_LABELS[solver]}] {ot_output} "
            f"(row j={j_cut}, stored y={stored_y:.10f})"
        )


if __name__ == "__main__":
    main()
