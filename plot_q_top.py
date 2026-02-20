#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# -------------- USER SETTINGS --------------
path = "postProcessing/sample/1/nearTopWallLine_T.xy"
k = 400.0          # W/m/K (copper)
dy = 1.0e-4        # m  <-- distance wall -> sampling line (EDIT)
Twall = 430.0      # K  <-- wall temperature (EDIT)
plt.rcParams.update({'font.size': 18})
# -------------------------------------------

def read_xy(p):
    data = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            a = line.split()
            if len(a) >= 2:
                data.append((float(a[0]), float(a[1])))
    if not data:
        raise RuntimeError(f"No numeric data in {p}")
    arr = np.array(data, dtype=float)
    return arr[:, 0], arr[:, 1]

s, Tline = read_xy(path)

# Wall-normal gradient at wall (1st order):
# dT/dn ≈ (T_line - T_wall) / dy
dTdn = (Tline - Twall) / dy            # K/m

# Wall heat flux
q = -k * dTdn                          # W/m^2

# Means along the line
mean_dTdn = np.mean(dTdn)
mean_q = np.mean(q)

print("----- Steady wall results (from near-wall line) -----")
print(f"File: {path}")
print(f"Assumed Twall = {Twall:.6g} K, dy = {dy:.6e} m, k = {k:.6g} W/m/K")
print(f"Mean dT/dn  = {mean_dTdn:.6e} K/m")
print(f"Mean q_wall = {mean_q:.6e} W/m^2")

# Plot profiles along the sampled coordinate
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(s, dTdn, marker='s', label=r"$\partial T/\partial n$ (K/m)")
ax.plot(s, q, marker='s', label=r"$q$ (W/m$^2$)")
ax.set_xlabel("s (first column in .xy)")
ax.grid(True)

leg = ax.legend(loc="best", framealpha=0.3)
leg.set_zorder(0)

plt.show()

