"""
Orszag-Tang 2D MHD vortex, γ = 5/3.

Figure 1 — contour panels (like Mignone Fig A.13):
  Two rows: t=π (≡ Mignone t=0.5) and t=2π (≡ Mignone t=1).
  Left column: density.  Right column: pressure.

Figure 2 — horizontal pressure cut (like Mignone Fig A.15):
  Cut at y = 0.3125 (paper coords), i.e. y = 0.3125×2π in simulation coords.
  t=π only.  x-axis normalised to [0, 1] for direct comparison with the paper.
  Optional reference: place a file at OUTPUT_DIR/ref_pressure_cut.csv
  (two columns: x_norm, pressure) to overlay a solid reference line.

Reference: Mignone et al. (2010) §4.5, Figs A.13 & A.15.

Usage:
  python plot_orszag_tang.py [--output-dir DIR] [--prefix PREFIX]
                             [--label LABEL] [--out-panels PATH]
                             [--out-cut PATH] [--show]
"""

import argparse
import os

# ── simulation parameters ────────────────────────────────────────────────────
NX = NY = 192
X_MIN, X_MAX = 0.0, 2 * 3.141592653589793
Y_MIN, Y_MAX = 0.0, 2 * 3.141592653589793

SNAPSHOTS = [
    ("tpi",  r"$t = \pi$  (Mignone $t = 0.5$)"),
    ("t2pi", r"$t = 2\pi$  (Mignone $t = 1$)"),
]

N_LEVELS = 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/gpu_orszag_tang_hlld_n1")
    p.add_argument("--prefix",     default="orszag_tang_gpu")
    p.add_argument("--label",      default="HLLD")
    p.add_argument("--out-panels", default="figs/orszag_tang_panels.png")
    p.add_argument("--out-cut",    default="figs/orszag_tang_pressure_cut.png")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Cell-centre coordinate arrays
    dx = (X_MAX - X_MIN) / NX
    dy = (Y_MAX - Y_MIN) / NY
    x  = np.linspace(X_MIN + 0.5 * dx, X_MAX - 0.5 * dx, NX)
    y  = np.linspace(Y_MIN + 0.5 * dy, Y_MAX - 0.5 * dy, NY)

    def load(tag, field):
        """Load a 2-D field CSV and flip rows so row-0 <-> low-y."""
        path = f"{args.output_dir}/{args.prefix}_{tag}_{field}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        return np.loadtxt(path, delimiter=",")[::-1]   # shape (NY, NX)

    available = []
    for tag, label in SNAPSHOTS:
        try:
            rho = load(tag, "rho")
            p   = load(tag, "p")
            available.append((tag, label, rho, p))
        except FileNotFoundError as e:
            print(f"  [skip] {e}")

    if not available:
        raise FileNotFoundError(
            f"No snapshot files found in '{args.output_dir}'. "
            "Run the simulation with --out to generate them."
        )

    out_dir = os.path.dirname(args.out_panels)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════
    # Figure 1 — density & pressure contour maps  (Mignone Fig A.13 style)
    # ════════════════════════════════════════════════════════════════════

    n_rows = len(available)
    fig1, axes = plt.subplots(n_rows, 2, figsize=(9, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (tag, label, rho, p) in enumerate(available):
        for col, (data, title) in enumerate([(rho, r"Density $\rho$"),
                                              (p,   r"Pressure $p$")]):
            ax = axes[row, col]
            levels = np.linspace(data.min(), data.max(), N_LEVELS)
            ax.contour(x, y, data, levels=levels, colors="k", linewidths=0.5)
            ax.set_title(f"{title},  {label}", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_color("black")

    fig1.suptitle(rf"Orszag–Tang vortex, $\gamma = 5/3$, $192^2$  ({args.label})", fontsize=11)
    fig1.tight_layout()
    fig1.savefig(args.out_panels, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out_panels}")

    # ════════════════════════════════════════════════════════════════════
    # Figure 2 — horizontal pressure cut at y=0.3125  (Mignone Fig A.15 style)
    # ════════════════════════════════════════════════════════════════════

    # Paper y=0.3125 on [0,1]^2 -> simulation y = 0.3125 x 2*pi
    Y_CUT_PAPER = 0.3125
    y_cut_sim   = Y_CUT_PAPER * 2.0 * np.pi
    j_cut       = int(np.argmin(np.abs(y - y_cut_sim)))
    print(f"Pressure cut: paper y={Y_CUT_PAPER}, sim y={y_cut_sim:.4f}, "
          f"closest cell j={j_cut} (y={y[j_cut]:.4f})")

    # x normalised to [0, 1] for direct comparison with the paper
    x_norm = x / (2.0 * np.pi)

    # Try to find the t=pi snapshot
    p_cut = None
    for tag, label, rho, p_data in available:
        if tag == "tpi":
            p_cut = p_data[j_cut, :]
            break

    if p_cut is None:
        # Fall back to the first available snapshot
        _, label, _, p_data = available[0]
        p_cut = p_data[j_cut, :]
        print("  [warn] t=pi snapshot not found; using first available snapshot.")

    # Optional reference solution (e.g. high-resolution CT or Stone et al.)
    ref_path = f"{args.output_dir}/ref_pressure_cut.csv"
    ref_data = None
    if os.path.exists(ref_path):
        ref_data = np.loadtxt(ref_path, delimiter=",")   # columns: x_norm, pressure
        print(f"  [ref] loaded reference from {ref_path}")

    fig2, ax = plt.subplots(figsize=(7, 4))

    # Plot all 192 cell-centre samples without connecting them.
    ax.plot(x_norm, p_cut, linestyle="none", marker="o", color="red",
            markersize=2.0, markerfacecolor="none", markeredgewidth=0.55,
            label=rf"{args.label} PLM  $192^2$  (192 cells)", zorder=3)

    if ref_data is not None:
        ax.plot(ref_data[:, 0], ref_data[:, 1],
                "k-", linewidth=1.0, label="Ref", zorder=4)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("position", fontsize=11)
    ax.set_ylabel("gas pressure", fontsize=11)
    ax.set_title(
        rf"Horizontal cut at $y = 0.3125$,  Orszag–Tang,  $t = \pi$  ({args.label})",
        fontsize=9
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(fontsize=9, framealpha=0.9)

    fig2.tight_layout()
    fig2.savefig(args.out_cut, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out_cut}")

    if args.show:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
