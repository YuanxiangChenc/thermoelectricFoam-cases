import re
import numpy as np
from pathlib import Path

# ---------------- user settings ----------------
time_line  = "1"
time_patch = "1"

LINE_DIR  = Path("postProcessing") / "sample" / time_line
PATCH_DIR = Path("postProcessing") / "sample_patch" / time_patch

LINE_FILES = {
    "top_seebeck": LINE_DIR / "topSeebeck_T_gradTx_gradTy_gradTz_k_gradT_q.xy",
    "btm_seebeck": LINE_DIR / "btmSeebeck_T_gradTx_gradTy_gradTz_k_gradT_q.xy",
}

PATCH_QN = {
    "top_conductor": "TOP",
    "water": "WATER",
    "air": "AIR",
}

LOG_FILE = Path("log")
# ------------------------------------------------

def read_line_xy(fp: Path):
    if not fp.exists(): return None, None
    # Load columns manually with numpy
    data = np.loadtxt(fp)
    if data.shape[1] != 12:
        raise ValueError(f"{fp}: expected 12 columns")
    
    s = data[:, 0]
    qy = data[:, 10]
    L = s[-1] - s[0]
    
    if abs(L) < 1e-30: return 0.0, 0.0
    
    q_mean_w = float(np.trapz(qy, s) / L)
    Qline    = float(np.trapz(qy, s))
    return q_mean_w, Qline

def read_raw_scalar(fp: Path):
    vals = []
    with fp.open("r") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"): continue
            vals.append(float(parts[3]))
    return np.array(vals, dtype=float)

def patch_from_qn(patch: str):
    fp_qn = PATCH_DIR / f"qn_{patch}.raw"
    fp_A  = PATCH_DIR / f"magSf_{patch}.raw"
    if not fp_qn.exists() or not fp_A.exists(): return None, None

    qn, A = read_raw_scalar(fp_qn), read_raw_scalar(fp_A)
    Atot = float(A.sum())
    if Atot < 1e-30: return 0.0, 0.0
    
    Q = float(np.sum(qn * A))
    return Q / Atot, Q

def parse_electric_Pnet(log_fp: Path):
    if not log_fp.exists(): return None
    text = log_fp.read_text(errors="ignore")
    m = re.findall(r"\[electricPower\](?:.|\n)*?Pnet\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*W", text)
    return float(m[-1]) if m else None

def safe_pct(num, den):
    if num is None or den is None or abs(den) < 1e-30: return None
    return 100.0 * (num / den)

# --- Process Data ---
results = {}

# Patches
patch_Q_sum = 0.0
for name, patch in PATCH_QN.items():
    qmean, Q = patch_from_qn(patch)
    results[name] = {"q": qmean, "Q": Q, "ratio": None}
    if Q: patch_Q_sum += Q

# Lines
for name, fp in LINE_FILES.items():
    qmean, Qline = read_line_xy(fp)
    results[name] = {"q": qmean, "Q": Qline, "ratio": None}

# Electric and Net
Pnet = parse_electric_Pnet(LOG_FILE)
results["Q.electric"] = {"q": None, "Q": Pnet, "ratio": None}
results["Net(error)"] = {"q": None, "Q": (patch_Q_sum + (Pnet or 0.0)), "ratio": None}

# Calculate Ratios
results["top_seebeck"]["ratio"] = safe_pct(results["top_seebeck"]["q"], results["top_conductor"]["q"])
results["water"]["ratio"] = safe_pct(results["water"]["Q"], results["top_conductor"]["Q"])
results["Q.electric"]["ratio"] = safe_pct(
    results["top_seebeck"]["Q"] - results["btm_seebeck"]["Q"],
    results["btm_seebeck"]["Q"]
)

# --- Print Table ---
header = f"{'BC':<15} | {'q [W/m^2]':>12} | {'Q\' [W/m]':>12} | {'ratio [%]':>10}"
print(header)
print("-" * len(header))

for bc in ["top_conductor", "water", "air", "top_seebeck", "btm_seebeck", "Q.electric", "Net(error)"]:
    res = results.get(bc, {"q": None, "Q": None, "ratio": None})
    q_str = f"{res['q']:.6g}" if res['q'] is not None else ""
    Q_str = f"{res['Q']:.6g}" if res['Q'] is not None else ""
    r_str = f"{res['ratio']:.4f}" if res['ratio'] is not None else ""
    print(f"{bc:<15} | {q_str:>12} | {Q_str:>12} | {r_str:>10}")
