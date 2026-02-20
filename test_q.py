#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# -------- USER SETTINGS --------
tfolder = "1"
base = f"postProcessing/sample/{tfolder}"
f1 = f"{base}/topSeeBeck_T_k.xy"      # y = y1
f2 = f"{base}/topSeeBeck__T_k.xy"     # y = y2

# IMPORTANT: set dy consistent with your sampling line y-values
# If your lines are at 0.00353 and 0.00343, dy = 1e-4.
dy = 5.0e-4   # <-- EDIT THIS to your actual y1-y2

# Mask threshold to keep only solid region (tune if needed)
k_threshold = 300.0  # W/m/K

plt.rcParams.update({'font.size': 18})
# -------------------------------

def read_xy_T_k(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#") or line.startswith("//"):
                continue
            nums = []
            for tok in line.split():
                try:
                    nums.append(float(tok))
                except ValueError:
                    pass
            if len(nums) == 3:
                rows.append(nums)  # s, T, k
            elif len(nums) == 2:
                # handle occasional missing s
                rows.append([np.nan, nums[0], nums[1]])

    if not rows:
        raise RuntimeError(f"No numeric rows found in {path}")

    arr = np.array(rows, dtype=float)
    s, T, k = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(~np.isfinite(s)):
        s = np.arange(len(T), dtype=float)
    return s, T, k

s1, T1, k1 = read_xy_T_k(f1)
s2, T2, k2 = read_xy_T_k(f2)

if len(T1) != len(T2):
    raise RuntimeError(f"Different point counts: {len(T1)} vs {len(T2)}")

s = s1

dTdn = (T1 - T2) / dy
k_mid = 0.5 * (k1 + k2)
q = -k_mid * dTdn

# Mask: keep only where both lines are in solid-ish region
mask = (k_mid > k_threshold) & np.isfinite(q)

k_mid_m = k_mid[mask]
q_m = q[mask]
s_m = s[mask]

print("----- Masked Top Seebeck results (steady) -----")
print(f"dy = {dy:.6e} m")
print(f"Mask: k_mid > {k_threshold} W/m/K")
print(f"Kept points: {mask.sum()} / {len(mask)}")
print(f"Mean k_mid (masked) = {np.mean(k_mid_m):.6e} W/m/K")
print(f"Mean q     (masked) = {np.mean(q_m):.6e} W/m^2")

# ---- Plot k and q on same figure with twin y-axis ----
fig = plt.figure()
axk = fig.add_subplot(111)

ln1 = axk.plot(s_m, k_mid_m, marker='s', label=r"$k_{mid}$ (W/m/K)")
axk.set_xlabel("s (first column in .xy)")
axk.set_ylabel(r"$k_{mid}$ [W/m/K]")
axk.grid(True)

axq = axk.twinx()
ln2 = axq.plot(s_m, q_m, marker='s', label=r"$q$ (W/m$^2$)")
axq.set_ylabel(r"$q$ [W/m$^2$]")

lines = ln1 + ln2
labels = [l.get_label() for l in lines]
leg = axk.legend(lines, labels, loc="best", framealpha=0.3)
leg.set_zorder(0)

plt.show()

