"""
Brio-Wu 1D MHD shock tube at t = 0.1, γ = 2.
Four-panel comparison: ρ, p, v_x, B_y vs x.
Reference: Brio & Wu (1988) Fig. 4; Dedner et al. (2002) Table IV.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

NX, NY   = 800, 4
X_MIN, X_MAX = 0.0, 1.0

OUTPUT_DIR = "outputs/gpu_brio_wu_hll_n1_12379"
PREFIX     = "brio_wu_gpu"
TAG        = "t10"

x = np.linspace(X_MIN + (X_MAX - X_MIN) / (2 * NX),
                X_MAX - (X_MAX - X_MIN) / (2 * NX), NX)

def load(field):
    # Shape (NY=4, NX=800); 1D problem — all rows identical, take row 0 after flip.
    raw = np.loadtxt(f"{OUTPUT_DIR}/{PREFIX}_{TAG}_{field}.csv", delimiter=",")
    return raw[::-1][0]

rho = load("rho")
p   = load("p")
u   = load("u")
By  = load("By")

panels = [
    (rho, r"$\rho$"),
    (p,   r"$p$"),
    (u,   r"$v_x$"),
    (By,  r"$B_y$"),
]

fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
axes = axes.ravel()

for ax, (data, ylabel) in zip(axes, panels):
    ax.plot(x, data, "k-", lw=0.9)
    ax.axvline(0.5, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(X_MIN, X_MAX)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

for ax in axes[2:]:
    ax.set_xlabel("$x$", fontsize=12)

fig.suptitle(r"Brio–Wu shock tube, $t = 0.1$, $\gamma = 2$  (HLL)", fontsize=11)
fig.tight_layout()

os.makedirs("figs", exist_ok=True)
fig.savefig("figs/brio_wu_profiles.png", dpi=300, bbox_inches="tight")
print("Saved figs/brio_wu_profiles.png")
plt.show()
