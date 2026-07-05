"""
Shock-bubble interaction: density contours at the 9 dimensionless times
used in Haas & Sturtevant (1987), Ms = 1.22.

Output files expected (one per snapshot per series):
  <output_dir>/<prefix>_t006_rho.csv  (t=0.6)
  ...
  <output_dir>/<prefix>_t190_rho.csv  (t=19.0)

Tag format: "t" + int(t_paper * 10) zero-padded to 3 digits.

Usage:
  # default: CPU vs GPU HLLC comparison (two columns)
  python plot_shock_bubble.py

  # one column per --series "label=output_dir=prefix" (repeatable)
  python plot_shock_bubble.py \
      --series "HLLC=outputs/201502/cpu_shock_bubble_hllc=shock_bubble_cpu" \
      --out figs/shock_bubble_hllc.png
"""

import argparse
import os

NX, NY   = 500, 197
X_MIN, X_MAX = 0.0, 0.225
Y_MIN, Y_MAX = 0.0, 0.089

DEFAULT_SERIES = [
    {"label": "CPU (HLLC)", "dir": "outputs/cpu_shock_bubble_hllc",    "prefix": "shock_bubble_cpu"},
    {"label": "GPU (HLLC)", "dir": "outputs/gpu_shock_bubble_hllc_n1", "prefix": "shock_bubble_gpu"},
]

DENSITY_LEVELS_DEFAULT = (0.1, 2.8, 45)

BUBBLE_CX, BUBBLE_CY, R_BUBBLE = 0.035, 0.0445, 0.025

SNAPSHOTS = [
    (0.6,  "t006"),
    (1.2,  "t012"),
    (1.8,  "t018"),
    (3.0,  "t030"),
    (4.6,  "t046"),
    (6.2,  "t062"),
    (7.8,  "t078"),
    (12.6, "t126"),
    (19.0, "t190"),
]


def parse_series(spec):
    label, out_dir, prefix = spec.split("=", 2)
    return {"label": label, "dir": out_dir, "prefix": prefix}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--series", action="append", type=parse_series,
                    help='Repeatable: "label=output_dir=prefix". '
                         "Defaults to the CPU-vs-GPU HLLC comparison if omitted.")
    p.add_argument("--out", default="figs/shock_bubble_panels.png")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    series_list = args.series if args.series else DEFAULT_SERIES

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(X_MIN + (X_MAX - X_MIN) / (2 * NX),
                    X_MAX - (X_MAX - X_MIN) / (2 * NX), NX)
    y = np.linspace(Y_MIN + (Y_MAX - Y_MIN) / (2 * NY),
                    Y_MAX - (Y_MAX - Y_MIN) / (2 * NY), NY)

    density_levels = np.linspace(*DENSITY_LEVELS_DEFAULT)

    theta_circ = np.linspace(0.0, 2.0 * np.pi, 400)
    bx = BUBBLE_CX + R_BUBBLE * np.cos(theta_circ)
    by = BUBBLE_CY + R_BUBBLE * np.sin(theta_circ)

    series_snaps = []
    for s in series_list:
        snaps = []
        for t_paper, tag in SNAPSHOTS:
            path = f"{s['dir']}/{s['prefix']}_{tag}_rho.csv"
            if os.path.exists(path):
                snaps.append((t_paper, tag, path))
            else:
                print(f"  [skip] {path} not found")
        series_snaps.append(snaps)

    n_rows = max((len(s) for s in series_snaps), default=0)
    n_cols = len(series_list)

    if n_rows == 0:
        raise FileNotFoundError("No snapshot files found. Run the simulations first.")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 1.6 * n_rows), squeeze=False)

    for col, s in enumerate(series_list):
        axes[0][col].set_title(s["label"], fontsize=11, pad=6, fontweight="bold")

    for col, (s, snaps) in enumerate(zip(series_list, series_snaps)):
        for row, (t_paper, tag, path) in enumerate(snaps):
            ax = axes[row][col]
            rho = np.loadtxt(path, delimiter=",")[::-1]

            ax.contour(x, y, rho, levels=density_levels, colors="k", linewidths=0.5)
            ax.plot(bx, by, "k--", linewidth=0.8)

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("black")

            if col == 0:
                ax.set_ylabel(f"$t = {t_paper}$", fontsize=9, rotation=0,
                              labelpad=28, va="center")

        for row in range(len(snaps), n_rows):
            axes[row][col].set_visible(False)

    fig.suptitle(r"Shock–bubble interaction, $M_s = 1.22$", fontsize=12, y=1.002)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}  ({n_rows} rows x {n_cols} cols)")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
