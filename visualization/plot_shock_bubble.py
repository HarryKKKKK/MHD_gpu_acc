"""
Shock-bubble interaction: density contours at the 9 dimensionless times
used in Haas & Sturtevant (1987), Ms = 1.22.

Output files expected (one per snapshot per solver):
  <OUTPUT_DIR>/<PREFIX>_t006_rho.csv  (t=0.6)
  ...
  <OUTPUT_DIR>/<PREFIX>_t190_rho.csv  (t=19.0)

Tag format: "t" + int(t_paper * 10) zero-padded to 3 digits.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────────────

NX, NY   = 500, 197
X_MIN, X_MAX = 0.0, 0.225
Y_MIN, Y_MAX = 0.0, 0.089

SOLVERS = [
    {"label": "CPU (HLLC)", "dir": "outputs/cpu_shock_bubble_hllc",    "prefix": "shock_bubble_cpu"},
    {"label": "GPU (HLLC)", "dir": "outputs/gpu_shock_bubble_hllc_n1", "prefix": "shock_bubble_gpu"},
]

DENSITY_LEVELS = np.linspace(0.1, 2.8, 45)

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

# ── Grid ─────────────────────────────────────────────────────────────────────

x = np.linspace(X_MIN + (X_MAX - X_MIN) / (2 * NX),
                X_MAX - (X_MAX - X_MIN) / (2 * NX), NX)
y = np.linspace(Y_MIN + (Y_MAX - Y_MIN) / (2 * NY),
                Y_MAX - (Y_MAX - Y_MIN) / (2 * NY), NY)

theta_circ = np.linspace(0.0, 2.0 * np.pi, 400)
bx = BUBBLE_CX + R_BUBBLE * np.cos(theta_circ)
by = BUBBLE_CY + R_BUBBLE * np.sin(theta_circ)

# ── Collect available snapshots per solver ───────────────────────────────────

solver_snaps = []
for solver in SOLVERS:
    snaps = []
    for t_paper, tag in SNAPSHOTS:
        path = f"{solver['dir']}/{solver['prefix']}_{tag}_rho.csv"
        if os.path.exists(path):
            snaps.append((t_paper, tag, path))
        else:
            print(f"  [skip] {path} not found")
    solver_snaps.append(snaps)

n_rows = max(len(s) for s in solver_snaps)
n_cols = len(SOLVERS)

if n_rows == 0:
    raise FileNotFoundError("No snapshot files found. Run the simulations first.")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 1.6 * n_rows))
if n_rows == 1:
    axes = [axes]

# column headers
for col, solver in enumerate(SOLVERS):
    axes[0][col].set_title(solver["label"], fontsize=11, pad=6, fontweight="bold")

for col, (solver, snaps) in enumerate(zip(SOLVERS, solver_snaps)):
    for row, (t_paper, tag, path) in enumerate(snaps):
        ax = axes[row][col]
        rho = np.loadtxt(path, delimiter=",")[::-1]

        ax.contour(x, y, rho, levels=DENSITY_LEVELS, colors="k", linewidths=0.5)
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

    # blank out any unused rows in this column
    for row in range(len(snaps), n_rows):
        axes[row][col].set_visible(False)

fig.suptitle(r"Shock–bubble interaction, $M_s = 1.22$", fontsize=12, y=1.002)
fig.tight_layout(h_pad=0.4, w_pad=0.6)

os.makedirs("figs", exist_ok=True)
out_path = "figs/shock_bubble_panels.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}  ({n_rows} rows x {n_cols} cols)")
plt.show()
