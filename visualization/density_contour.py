"""
Kelvin-Helmholtz instability: density contour at t = 0.5.
Compare with Dedner et al. (2002) Fig. 8.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

NX, NY   = 200, 400
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0

OUTPUT_DIR = "outputs/gpu_kelvin_helmholtz_hll_n1_12379"
PREFIX     = "kelvin_helmholtz_gpu"

DENSITY_LEVELS = np.linspace(0.85, 1.15, 30)

x = np.linspace(X_MIN + (X_MAX - X_MIN) / (2 * NX),
                X_MAX - (X_MAX - X_MIN) / (2 * NX), NX)
y = np.linspace(Y_MIN + (Y_MAX - Y_MIN) / (2 * NY),
                Y_MAX - (Y_MAX - Y_MIN) / (2 * NY), NY)

def load(tag, field):
    raw = np.loadtxt(f"{OUTPUT_DIR}/{PREFIX}_{tag}_{field}.csv", delimiter=",")
    return raw[::-1]

rho = load("t50", "rho")

fig, ax = plt.subplots(figsize=(3.2, 6.4))
ax.contour(x, y, rho, levels=DENSITY_LEVELS, colors="k", linewidths=0.5)

ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.set_title(r"KH instability, $t = 0.5$  (HLL)", fontsize=9)

os.makedirs("figs", exist_ok=True)
fig.tight_layout()
fig.savefig("figs/kh_density_t05.png", dpi=300, bbox_inches="tight")
print("Saved figs/kh_density_t05.png")
plt.show()
