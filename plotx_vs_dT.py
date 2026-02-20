import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.size': 18})

p = Path("postProcessing/sample/1/centerLine_T.xy")
assert p.exists(), "File not found: /mnt/data/centerLine_T.xy"

# Load, skipping commented lines
with p.open("r", errors="ignore") as f:
    lines = [ln for ln in f if not ln.strip().startswith("#") and ln.strip()]

# Try parsing with whitespace splitting
rows = [ln.strip().split() for ln in lines]
# Convert to floats, keep rows that can be converted
num_rows = []
for r in rows:
    try:
        num_rows.append([float(x) for x in r])
    except:
        pass

arr = np.array(num_rows, dtype=float)
if arr.ndim != 2 or arr.shape[0] < 2:
    raise ValueError("Could not parse numeric data from file.")

# Heuristics for columns:
# - If 2 cols: x, T
# - If 4 cols: x, y, z, T
# - Otherwise: take first as x, last as T
if arr.shape[1] == 2:
    x = arr[:,0]
    T = arr[:,1]
elif arr.shape[1] == 4:
    x = arr[:,0]
    T = arr[:,3]
else:
    x = arr[:,0]
    T = arr[:,-1]

# Sort by x just in case
ordr = np.argsort(x)
x = x[ordr]
T = T[ordr]

# Build normalized coordinate n in [0,1] from extents
xmin, xmax = float(np.min(x)), float(np.max(x))
if xmax == xmin:
    n = np.zeros_like(x)
else:
    n = (x - xmin) / (xmax - xmin)

# Linear reference between end temperatures
Th, Tc = float(T[0]), float(T[-1])
Tlin = Th*(1.0 - n) + Tc*n
dT = T - Tlin


# --- reference data from paper (x, y) pairs ---
ref_data = np.array([
    [0, 0.032397803469973625],
    [0.011955614607982612, 1.020518351804806],
    [0.023911229215965224, 1.6684661162047016],
    [0.03586678127724494, 2.365011179023816],
    [0.04696844575133701, 3.0291572519444445],
    [0.0691716496061153, 4.357451770642724],
    [0.0828352270285819, 5.02159784356335],
    [0.09393682895597107, 5.685745102912491],
    [0.10760034383173489, 6.333692867312386],
    [0.12126392125420153, 7.014038435182256],
    [0.1349274361299653, 7.645787891061421],
    [0.1639624286959769, 8.893088494299024],
    [0.18104186920070886, 9.573434062168893],
    [0.2160547003440632, 10.804535763671506],
    [0.25704524497135467, 12.003239661704148],
    [0.3057215909904727, 13.03995691524394],
    [0.36464558880308945, 13.768898298497387],
    [0.43637915135757943, 13.995680253323053],
    [0.507258701231476, 13.477321923160286],
    [0.5644747987763112, 12.537796893601902],
    [0.6131511447954294, 11.582073555522788],
    [0.6558496522372048, 10.415766867745868],
    [0.6985482847723861, 9.281857390224664],
    [0.738684879265787, 8.083152305763509],
    [0.7754056732236253, 6.835853482168682],
    [0.8146883801298382, 5.653347299442515],
    [0.8505551614070831, 4.438444499674889],
    [0.888983855632703, 3.2073433913865323],
    [0.9282664374455102, 2.041036703609612],
    [0.9684031570323166, 0.8747300158326918],
    [0.9914603735676885, 0.29157667194422837]

])

xref, yref = ref_data[:,0], ref_data[:,1]

# --- plot both ---
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(n, dT, 'b-', lw=2, label="Simulation")
ax.plot(xref, yref, 'ro', ms=4, label="Reference (matlab)")

ax.set_xlabel("x/L")
ax.set_ylabel(r"$\Delta T$ [K]")
ax.set_xlim(0,1); ax.set_ylim(0,15)
ax.set_xticks(np.arange(0,1.01,0.2))
ax.set_yticks(np.arange(0,15.01, 5))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()
plt.tight_layout()
plt.show()


