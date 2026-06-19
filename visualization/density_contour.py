import numpy as np
import matplotlib.pyplot as plt

rho = np.loadtxt(
    "outputs/gpu_kelvin_helmholtz_hll_n1_12370/kelvin_helmholtz_gpu_t50_rho.csv",
    delimiter=","
)

plt.figure(figsize=(6,10))

plt.contour(
    rho,
    levels=30,
    linewidths=0.7
)

plt.gca().set_aspect("equal")
plt.title("Kelvin-Helmholtz density contours (t=0.5)")
plt.tight_layout()
plt.show()