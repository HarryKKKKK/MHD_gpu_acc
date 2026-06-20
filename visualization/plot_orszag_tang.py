"""
Orszag-Tang 2D MHD vortex at t = π, γ = 5/3.
Left: density contour.  Right: pressure contour.
Reference: Dahlburg & Picone (1989) Fig. 1; Dedner et al. (2002) Fig. 3.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

NX = NY = 192
X_MIN, X_MAX = 0.0, 2 * np.pi
Y_MIN, Y_MAX = 0.0, 2 * np.pi

OUTPUT_DIR = "outputs/gpu_orszag_tang_hll_n1_12379"
PREFIX     = "orszag_tang_gpu"
TAG        = "t314"

N_LEVELS = 30

x = np.linspace(X_MIN + (X_MAX - X_MIN) / (2 * NX),
                X_MAX - (X_MAX - X_MIN) / (2 * NX), NX)
y = np.linspace(Y_MIN + (Y_MAX - Y_MIN) / (2 * NY),
                Y_MAX - (Y_MAX - Y_MIN) / (2 * NY), NY)

def load(field):
    raw = np.loadtxt(f"{OUTPUT_DIR}/{PREFIX}_{TAG}_{field}.csv", delimiter=",")
    return raw[::-1]

rho = load("rho")
p   = load("p")

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

for ax, (data, title) in zip(axes, [
        (rho, r"Density $\rho$"),
        (p,   r"Pressure $p$"),
]):
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

fig.suptitle(r"Orszag–Tang vortex, $t = \pi$, $\gamma = 5/3$  (HLL, $192^2$)", fontsize=11)
fig.tight_layout()

os.makedirs("figs", exist_ok=True)
fig.savefig("figs/orszag_tang_t_pi.png", dpi=300, bbox_inches="tight")
print("Saved figs/orszag_tang_t_pi.png")
plt.show()
