"""
MHD Rotor problem -- first rotor problem (Toth 2000, JCP 161, Sec 6.6; originally
Balsara & Spicer), gamma = 1.4, v0 = 2, p = 1, Bx = 5/sqrt(4*pi), on [0,1]^2 with
400 x 400 cells (see rotor_state() in src/test_cases.cpp -- matches the paper's
setup term for term). No analytic snapshot schedule is configured for this case,
so the solver only ever writes an initial (t0) and a final-time field; final tag
is "t" + round(t_end * 100), i.e. "t15" for t_end = 0.15.

Two plot modes:

1. Single run, 4-panel contour plot (density, pressure, Mach number |v|/c_s,
   magnetic pressure B^2/2) at the final time -- mirrors Figs. 18/19 of the paper.

     python plot_rotor.py --output-dir outputs/201502/cpu_rotor_hlld \
         --prefix rotor_cpu --label HLLD --out figs/rotor_hlld.png

2. Multi-solver comparison of the Mach number only -- mirrors Fig. 20 of the
   paper (there it compares divergence-cleaning schemes; here there is only one
   scheme, so it compares Riemann solvers instead).

     python plot_rotor.py \
         --series "FORCE=outputs/201502/cpu_rotor_force=rotor_cpu" \
         --series "HLL=outputs/201502/cpu_rotor_hll=rotor_cpu" \
         --series "HLLC=outputs/201502/cpu_rotor_hllc=rotor_cpu" \
         --series "HLLD=outputs/201502/cpu_rotor_hlld=rotor_cpu" \
         --out-compare figs/rotor_mach_compare.png
"""

import argparse
import os

NX = NY = 400
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
GAMMA = 1.4

N_LEVELS = 30


def parse_series(spec):
    label, out_dir, prefix = spec.split("=", 2)
    return {"label": label, "dir": out_dir, "prefix": prefix}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/cpu_rotor_hlld")
    p.add_argument("--prefix",     default="rotor_cpu")
    p.add_argument("--tag",        default="t15")
    p.add_argument("--label",      default="HLLD")
    p.add_argument("--out",        default="figs/rotor_panels.png")
    p.add_argument("--series", action="append", type=parse_series,
                    help='Repeatable: "label=output_dir=prefix". When given, '
                         "produces the multi-solver Mach-number comparison "
                         "(Fig. 20 style) instead of the single-run 4-panel plot.")
    p.add_argument("--out-compare", default="figs/rotor_mach_compare.png")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_fields(output_dir, prefix, tag, np):
    def load(field):
        path = f"{output_dir}/{prefix}_{tag}_{field}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        return np.loadtxt(path, delimiter=",")[::-1]   # shape (NY, NX)

    rho = load("rho")
    p_  = load("p")
    u   = load("u")
    v   = load("v")
    w   = load("w")
    Bx  = load("Bx")
    By  = load("By")
    Bz  = load("Bz")

    speed = np.sqrt(u * u + v * v + w * w)
    cs    = np.sqrt(GAMMA * p_ / rho)
    mach  = speed / cs
    pmag  = 0.5 * (Bx * Bx + By * By + Bz * Bz)
    return rho, p_, mach, pmag


def contour_panel(ax, x, y, data, title, np):
    levels = np.linspace(data.min(), data.max(), N_LEVELS)
    ax.contour(x, y, data, levels=levels, colors="k", linewidths=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")


def plot_single(args, np, plt):
    x = np.linspace(X_MIN + 0.5 * (X_MAX - X_MIN) / NX, X_MAX - 0.5 * (X_MAX - X_MIN) / NX, NX)
    y = np.linspace(Y_MIN + 0.5 * (Y_MAX - Y_MIN) / NY, Y_MAX - 0.5 * (Y_MAX - Y_MIN) / NY, NY)

    rho, p, mach, pmag = load_fields(args.output_dir, args.prefix, args.tag, np)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    panels = [
        (rho,  r"Density $\rho$"),
        (p,    r"Pressure $p$"),
        (mach, r"Mach number $|v|/c_s$"),
        (pmag, r"Magnetic pressure $B^2/2$"),
    ]
    for ax, (data, title) in zip(axes, panels):
        contour_panel(ax, x, y, data, title, np)

    fig.suptitle(rf"MHD rotor problem, $t = 0.15$, $400^2$  ({args.label})", fontsize=11)
    fig.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")
    return fig


def plot_compare(args, np, plt):
    x = np.linspace(X_MIN + 0.5 * (X_MAX - X_MIN) / NX, X_MAX - 0.5 * (X_MAX - X_MIN) / NX, NX)
    y = np.linspace(Y_MIN + 0.5 * (Y_MAX - Y_MIN) / NY, Y_MAX - 0.5 * (Y_MAX - Y_MIN) / NY, NY)

    series_list = args.series
    n = len(series_list)
    n_cols = 2 if n > 1 else 1
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, s in zip(axes_flat, series_list):
        try:
            _, _, mach, _ = load_fields(s["dir"], s["prefix"], args.tag, np)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            ax.set_visible(False)
            continue
        contour_panel(ax, x, y, mach, s["label"], np)

    for ax in axes_flat[len(series_list):]:
        ax.set_visible(False)

    fig.suptitle(r"MHD rotor problem, Mach number $|v|/c_s$ by solver, $t = 0.15$", fontsize=12)
    fig.tight_layout()

    out_dir = os.path.dirname(args.out_compare)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out_compare, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out_compare}")
    return fig


def main():
    args = parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt

    if args.series:
        fig = plot_compare(args, np, plt)
    else:
        fig = plot_single(args, np, plt)

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
