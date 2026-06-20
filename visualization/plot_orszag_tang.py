"""
Orszag-Tang 2D MHD vortex, γ = 5/3.
Two rows: t=π (≡ Mignone et al. t=0.5) and t=2π (≡ Mignone t=1).
Left column: density.  Right column: pressure.
Reference: Mignone et al. (2010) §4.5, Fig. A.13.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

NX = NY = 192
X_MIN, X_MAX = 0.0, 2 * np.pi
Y_MIN, Y_MAX = 0.0, 2 * np.pi

OUTPUT_DIR = "outputs/gpu_orszag_tang_hlld_n1"
PREFIX     = "orszag_tang_gpu"

# Snapshot tags and their display labels
SNAPSHOTS = [
    ("tpi",  r"$t = \pi$  (Mignone $t = 0.5$)"),
    ("t2pi", r"$t = 2\pi$  (Mignone $t = 1$)"),
]

N_LEVELS = 30

x = np.linspace(X_MIN + (X_MAX - X_MIN) / (2 * NX),
                X_MAX - (X_MAX - X_MIN) / (2 * NX), NX)
y = np.linspace(Y_MIN + (Y_MAX - Y_MIN) / (2 * NY),
                Y_MAX - (Y_MAX - Y_MIN) / (2 * NY), NY)

def load(tag, field):
    path = f"{OUTPUT_DIR}/{PREFIX}_{tag}_{field}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return np.loadtxt(path, delimiter=",")[::-1]

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
        f"No snapshot files found in '{OUTPUT_DIR}'. "
        "Run the simulation with --out to generate them."
    )

n_rows = len(available)
fig, axes = plt.subplots(n_rows, 2, figsize=(9, 4.5 * n_rows))
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

fig.suptitle(r"Orszag–Tang vortex, $\gamma = 5/3$, $192^2$  (HLLD)", fontsize=11)
fig.tight_layout()

os.makedirs("figs", exist_ok=True)
out_path = "figs/orszag_tang_panels.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}  ({n_rows} rows)")
plt.show()
