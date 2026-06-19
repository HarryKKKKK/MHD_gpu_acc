"""
Reproduce the style of Dedner et al. (2002) Fig. 8:
  – density isolines of the Kelvin–Helmholtz instability at t = 0.5
  – black lines, white background, portrait layout matching the paper panel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── domain & grid (from CaseConfig in test_cases.cpp) ─────────────────────────
NX, NY   = 200, 400
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0

OUTPUT_DIR = "outputs/gpu_kelvin_helmholtz_hll_n1_12370"
PREFIX     = "kelvin_helmholtz_gpu"

# ── physical cell-centre coordinates ──────────────────────────────────────────
x = np.linspace(X_MIN + (X_MAX - X_MIN) / (2 * NX),
                X_MAX - (X_MAX - X_MIN) / (2 * NX), NX)
y = np.linspace(Y_MIN + (Y_MAX - Y_MIN) / (2 * NY),
                Y_MAX - (Y_MAX - Y_MIN) / (2 * NY), NY)

def load(tag, field):
    """CSV is written top→bottom (j=ny−1 first); flip so row 0 = y_min."""
    raw = np.loadtxt(f"{OUTPUT_DIR}/{PREFIX}_{tag}_{field}.csv", delimiter=",")
    return raw[::-1]

rho = load("t50", "rho")   # t = 0.5  (t_end × 100 = 50)

# ── contour levels: fixed range covering the physical variation ────────────────
# Paper uses ~20 isolines; we choose levels that span [rho_min, rho_max].
levels = np.linspace(rho.min(), rho.max(), 22)

# ── single-panel figure matching Fig. 8 portrait dimensions ───────────────────
fig, ax = plt.subplots(figsize=(3.2, 6.4))   # 1 : 2 aspect, like the paper panel

ax.contour(x, y, rho, levels=levels, colors="black", linewidths=0.55)

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_aspect("equal")
ax.set_xlabel("$x$", fontsize=10)
ax.set_ylabel("$y$", fontsize=10)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.tick_params(labelsize=9)

# Subtitle mimicking the paper's subfigure label
ax.set_title(r"HLL, $t = 0.5$", fontsize=9, pad=4)

fig.tight_layout()
fig.savefig("visualization/kh_fig8_comparison.pdf", dpi=200, bbox_inches="tight")
fig.savefig("visualization/kh_fig8_comparison.png", dpi=200, bbox_inches="tight")
print("Saved kh_fig8_comparison.{pdf,png}")
plt.show()
