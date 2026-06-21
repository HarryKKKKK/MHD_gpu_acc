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
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── simulation parameters ────────────────────────────────────────────────────
NX = NY = 192
X_MIN, X_MAX = 0.0, 2 * np.pi
Y_MIN, Y_MAX = 0.0, 2 * np.pi

OUTPUT_DIR = "outputs/gpu_orszag_tang_hlld_n1"
PREFIX     = "orszag_tang_gpu"

SNAPSHOTS = [
    ("tpi",  r"$t = \pi$  (Mignone $t = 0.5$)"),
    ("t2pi", r"$t = 2\pi$  (Mignone $t = 1$)"),
]

N_LEVELS = 30

# Cell-centre coordinate arrays
dx = (X_MAX - X_MIN) / NX
dy = (Y_MAX - Y_MIN) / NY
x  = np.linspace(X_MIN + 0.5 * dx, X_MAX - 0.5 * dx, NX)
y  = np.linspace(Y_MIN + 0.5 * dy, Y_MAX - 0.5 * dy, NY)

# ── helpers ──────────────────────────────────────────────────────────────────

def load(tag, field):
    """Load a 2-D field CSV and flip rows so row-0 ↔ low-y."""
    path = f"{OUTPUT_DIR}/{PREFIX}_{tag}_{field}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return np.loadtxt(path, delimiter=",")[::-1]   # shape (NY, NX)

# ── load snapshots ───────────────────────────────────────────────────────────

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

os.makedirs("figs", exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — density & pressure contour maps  (Mignone Fig A.13 style)
# ════════════════════════════════════════════════════════════════════════════

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

fig1.suptitle(r"Orszag–Tang vortex, $\gamma = 5/3$, $192^2$  (HLLD)", fontsize=11)
fig1.tight_layout()
fig1.savefig("figs/orszag_tang_panels.png", dpi=300, bbox_inches="tight")
print("Saved figs/orszag_tang_panels.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — horizontal pressure cut at y=0.3125  (Mignone Fig A.15 style)
# ════════════════════════════════════════════════════════════════════════════

# Paper y=0.3125 on [0,1]² → simulation y = 0.3125 × 2π
Y_CUT_PAPER = 0.3125
y_cut_sim   = Y_CUT_PAPER * 2.0 * np.pi
j_cut       = int(np.argmin(np.abs(y - y_cut_sim)))
print(f"Pressure cut: paper y={Y_CUT_PAPER}, sim y={y_cut_sim:.4f}, "
      f"closest cell j={j_cut} (y={y[j_cut]:.4f})")

# x normalised to [0, 1] for direct comparison with the paper
x_norm = x / (2.0 * np.pi)

# Try to find the t=π snapshot
p_cut = None
for tag, label, rho, p_data in available:
    if tag == "tpi":
        p_cut = p_data[j_cut, :]
        break

if p_cut is None:
    # Fall back to the first available snapshot
    _, label, _, p_data = available[0]
    p_cut = p_data[j_cut, :]
    print("  [warn] t=π snapshot not found; using first available snapshot.")

# Optional reference solution (e.g. high-resolution CT or Stone et al.)
ref_path = f"{OUTPUT_DIR}/ref_pressure_cut.csv"
ref_data = None
if os.path.exists(ref_path):
    ref_data = np.loadtxt(ref_path, delimiter=",")   # columns: x_norm, pressure
    print(f"  [ref] loaded reference from {ref_path}")

# ── build the figure (single panel; paper uses two for different schemes) ──
fig2, ax = plt.subplots(figsize=(7, 4))

# Simulation result — red squares, every 4th point to reduce clutter at 192 pts
stride = max(1, NX // 48)   # ~48 markers across [0,1]
ax.plot(x_norm[::stride], p_cut[::stride],
        "rs", markersize=4, markerfacecolor="none", markeredgewidth=0.8,
        label=r"HLLD PLM  $192^2$  (this work)", zorder=3)

# Full line (thinner) underneath the markers so the curve shape is visible
ax.plot(x_norm, p_cut, "r-", linewidth=0.6, alpha=0.5, zorder=2)

# Reference line if available
if ref_data is not None:
    ax.plot(ref_data[:, 0], ref_data[:, 1],
            "k-", linewidth=1.0, label="Ref", zorder=4)

ax.set_xlim(0.0, 1.0)
ax.set_xlabel("position", fontsize=11)
ax.set_ylabel("gas pressure", fontsize=11)
ax.set_title(
    r"Horizontal cut at $y = 0.3125$,  Orszag–Tang,  $t = \pi$  (Mignone $t = 0.5$)",
    fontsize=9
)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.legend(fontsize=9, framealpha=0.9)

fig2.tight_layout()
fig2.savefig("figs/orszag_tang_pressure_cut.png", dpi=300, bbox_inches="tight")
print("Saved figs/orszag_tang_pressure_cut.png")

plt.show()
