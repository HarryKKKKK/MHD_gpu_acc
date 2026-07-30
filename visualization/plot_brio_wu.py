"""
Brio-Wu 1D MHD shock tube at t = 0.1, γ = 2.
Four-panel comparison: ρ, p, v_y, B_y vs x.
Reference: Brio & Wu (1988) Fig. 4; Dedner et al. (2002) Table IV.

Usage:
  python plot_brio_wu.py [--output-dir DIR] [--prefix PREFIX] [--tag TAG]
                          [--label LABEL] [--out PATH] [--show]
"""

import argparse
import os

NX, NY   = 800, 4
X_MIN, X_MAX = 0.0, 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/gpu_brio_wu_hlld_n1")
    p.add_argument("--prefix",     default="brio_wu_gpu")
    p.add_argument("--tag",        default="t10")
    p.add_argument("--label",      default="HLLD")
    p.add_argument("--out",        default="figs/brio_wu_profiles.png")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(X_MIN + (X_MAX - X_MIN) / (2 * NX),
                    X_MAX - (X_MAX - X_MIN) / (2 * NX), NX)

    def load(field):
        # Shape (NY=4, NX=800); 1D problem — all rows identical, take row 0 after flip.
        raw = np.loadtxt(f"{args.output_dir}/{args.prefix}_{args.tag}_{field}.csv", delimiter=",")
        return raw[::-1][0]

    rho = load("rho")
    p   = load("p")
    v   = load("v")
    By  = load("By")

    panels = [
        (rho, r"$\rho$"),
        (p,   r"$p$"),
        (v,   r"$v_y$"),
        (By,  r"$B_y$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    axes = axes.ravel()

    for ax, (data, ylabel) in zip(axes, panels):
        # Show every finite-volume cell value explicitly.  Connecting the
        # samples would imply a continuous reconstruction that is not stored.
        ax.plot(x, data, linestyle="none", marker=".", color="black",
                markersize=1.8)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(X_MIN, X_MAX)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    for ax in axes[2:]:
        ax.set_xlabel("$x$", fontsize=12)

    fig.suptitle(rf"Brio–Wu shock tube, $t = 0.1$, $\gamma = 2$  ({args.label})", fontsize=11)
    fig.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
