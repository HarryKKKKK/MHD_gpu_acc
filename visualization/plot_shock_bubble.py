"""
Shock-bubble interaction: density contours at the 9 dimensionless times
used in Haas & Sturtevant (1987), Ms = 1.22.

Output files expected (one per snapshot):
  <OUTPUT_DIR>/<PREFIX>_t006_rho.csv  (t=0.6)
  <OUTPUT_DIR>/<PREFIX>_t012_rho.csv  (t=1.2)
  ...
  <OUTPUT_DIR>/<PREFIX>_t190_rho.csv  (t=19.0)

Tag format: "t" + int(t_paper * 10) zero-padded to 3 digits.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ────────────────────────────────────────────────────────────

NX, NY   = 500, 197
X_MIN, X_MAX = 0.0, 0.225
Y_MIN, Y_MAX = 0.0, 0.089

OUTPUT_DIR = "outputs/gpu_shock_bubble_hll_n1"
PREFIX     = "shock_bubble_gpu"

DENSITY_LEVELS = np.linspace(0.1, 2.8, 45)

BUBBLE_CX, BUBBLE_CY, R_BUBBLE = 0.035, 0.0445, 0.025

# Dimensionless times from the paper and their file tags
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

# ── Grid ─────────────────────────────────────────────────────────────────────

x = np.linspace(X_MIN + (X_MAX - X_MIN) / (2 * NX),
                X_MAX - (X_MAX - X_MIN) / (2 * NX), NX)
y = np.linspace(Y_MIN + (Y_MAX - Y_MIN) / (2 * NY),
                Y_MAX - (Y_MAX - Y_MIN) / (2 * NY), NY)

theta_circ = np.linspace(0.0, 2.0 * np.pi, 400)
bx = BUBBLE_CX + R_BUBBLE * np.cos(theta_circ)
by = BUBBLE_CY + R_BUBBLE * np.sin(theta_circ)

# ── Load & plot ───────────────────────────────────────────────────────────────

available = []
for t_paper, tag in SNAPSHOTS:
    path = f"{OUTPUT_DIR}/{PREFIX}_{tag}_rho.csv"
    if os.path.exists(path):
        available.append((t_paper, tag, path))
    else:
        print(f"  [skip] {path} not found")

if not available:
    raise FileNotFoundError(
        f"No snapshot files found in '{OUTPUT_DIR}'. "
        "Run the simulation with --out to generate them."
    )

n_panels = len(available)
fig_h    = 1.6 * n_panels
fig, axes = plt.subplots(n_panels, 1, figsize=(9, fig_h))
if n_panels == 1:
    axes = [axes]

for ax, (t_paper, tag, path) in zip(axes, available):
    rho = np.loadtxt(path, delimiter=",")[::-1]  # flip y so row 0 = y_min

    ax.contour(x, y, rho, levels=DENSITY_LEVELS, colors="k", linewidths=0.5)
    ax.plot(bx, by, "k--", linewidth=0.8)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.set_title(f"$t = {t_paper}$", fontsize=9, pad=2)

fig.suptitle(r"Shock–bubble interaction, $M_s = 1.22$ (HLL)", fontsize=11, y=1.002)
fig.tight_layout(h_pad=0.4)

os.makedirs("figs", exist_ok=True)
out_path = "figs/shock_bubble_panels.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}  ({n_panels} panels)")
plt.show()
