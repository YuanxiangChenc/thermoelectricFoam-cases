import re
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- user settings ----------------
time_line  = "1"
time_patch = "1"

LINE_DIR  = Path("postProcessing") / "sample" / time_line
PATCH_DIR = Path("postProcessing") / "sample_patch" / time_patch

# line (sets) files -> q and Q' from q_y along line
LINE_FILES = {
    "top_seebeck": LINE_DIR / "topSeebeck_T_gradTx_gradTy_gradTz_k_gradT_q.xy",
    "btm_seebeck": LINE_DIR / "btmSeebeck_T_gradTx_gradTy_gradTz_k_gradT_q.xy",
}

# patch-normal flux qn files -> q and Q from qn on patches
PATCH_QN = {
    "top_conductor": "TOP",
    "water": "WATER",
    "air": "AIR",
}

LOG_FILE = Path("log")   # change if your solver log filename differs
# ------------------------------------------------


def read_line_xy(fp: Path):
    """
    12 cols:
    s, T, gradTx, gradTy, gradTz, k, gradT_x, gradT_y, gradT_z, q_x, q_y, q_z
    Returns:
      q_mean = length-weighted mean q_y [W/m^2]
      Qline  = ∫ q_y ds                 [W/m]
    """
    df = pd.read_csv(fp, delim_whitespace=True, header=None)
    if df.shape[1] != 12:
        raise ValueError(f"{fp}: expected 12 columns, got {df.shape[1]}")
    df.columns = ["s","T","gradTx","gradTy","gradTz","k","gradT_x","gradT_y","gradT_z","q_x","q_y","q_z"]

    s  = df["s"].values
    qy = df["q_y"].values
    L = s[-1] - s[0]
    if abs(L) < 1e-30:
        raise ValueError(f"{fp}: line length ~0")

    q_mean_w = float(np.trapz(qy, s) / L)   # W/m^2
    Qline    = float(np.trapz(qy, s))       # W/m

    return q_mean_w, Qline


def read_raw_scalar(fp: Path):
    """raw file: x y z val ; returns (N,) scalar values"""
    vals = []
    with fp.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vals.append(float(parts[3]))
    return np.array(vals, dtype=float)


def patch_from_qn(patch: str):
    """
    Uses qn_PATCH.raw and magSf_PATCH.raw (aligned FACE_DATA).
    Returns:
      qmean = area-weighted mean qn [W/m^2]
      Q     = Σ(qn*Af)             [W]
    """
    fp_qn = PATCH_DIR / f"qn_{patch}.raw"
    fp_A  = PATCH_DIR / f"magSf_{patch}.raw"

    if not fp_qn.exists() or not fp_A.exists():
        return None, None

    qn = read_raw_scalar(fp_qn)
    A  = read_raw_scalar(fp_A)

    if len(qn) != len(A):
        raise ValueError(f"Size mismatch on {patch}: qn N={len(qn)} vs magSf N={len(A)}")

    Atot = float(A.sum())
    if Atot < 1e-30:
        raise ValueError(f"{patch}: total area ~0")

    Q = float(np.sum(qn * A))
    qmean = float(Q / Atot)
    return qmean, Q


def parse_electric_Pnet(log_fp: Path):
    """
    Reads last [electricPower] block and returns Pnet [W] or None.
    """
    if not log_fp.exists():
        return None
    text = log_fp.read_text(errors="ignore")
    pat = re.compile(r"\[electricPower\](?:.|\n)*?Pnet\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*W")
    m = pat.findall(text)
    return float(m[-1]) if m else None


def safe_pct(num, den):
    if num is None or den is None:
        return np.nan
    if isinstance(num, float) and np.isnan(num):
        return np.nan
    if isinstance(den, float) and np.isnan(den):
        return np.nan
    if abs(den) < 1e-30:
        return np.nan
    return 100.0 * (num / den)


def fmt_num(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return f"{x:.6g}"


# ---------------- build table ----------------
rows = []

# patch-based (qn) rows
patch_Q_sum = 0.0
patch_any = False
for name, patch in PATCH_QN.items():
    qmean, Q = patch_from_qn(patch)
    rows.append({"BC": name, "q [W/m^2]": qmean, "Q' [W/m]": Q, "ratio [%]": np.nan})
    if Q is not None:
        patch_Q_sum += Q
        patch_any = True

# line-based rows
for name, fp in LINE_FILES.items():
    if fp.exists():
        qmean, Qline = read_line_xy(fp)
        rows.append({"BC": name, "q [W/m^2]": qmean, "Q' [W/m]": Qline, "ratio [%]": np.nan})
    else:
        rows.append({"BC": name, "q [W/m^2]": np.nan, "Q' [W/m]": np.nan, "ratio [%]": np.nan})

# electric from log
Pnet = parse_electric_Pnet(LOG_FILE)
rows.append({"BC": "Q.electric", "q [W/m^2]": np.nan, "Q' [W/m]": Pnet, "ratio [%]": np.nan})

# net = patch sum + electric
net = np.nan
if patch_any and (Pnet is not None):
    net = patch_Q_sum + Pnet
rows.append({"BC": "Net(error)", "q [W/m^2]": np.nan, "Q' [W/m]": net, "ratio [%]": np.nan})

df = pd.DataFrame(rows, columns=["BC", "q [W/m^2]", "Q' [W/m]", "ratio [%]"])

# ---------------- fill ratios INTO existing rows ----------------
# Ratios requested:
# 1) top_seebeck row: q_top_seebeck / q_top_conductor
# 2) water row:       Q'_water / Q'_conductor  (conductor = top_conductor)

# Fetch needed values
q_top_cond = df.loc[df["BC"] == "top_conductor", "q [W/m^2]"].values
q_top_seeb = df.loc[df["BC"] == "top_seebeck",   "q [W/m^2]"].values
Qw         = df.loc[df["BC"] == "water",         "Q' [W/m]"].values
Qc         = df.loc[df["BC"] == "top_conductor", "Q' [W/m]"].values

q_top_cond = q_top_cond[0] if len(q_top_cond) else np.nan
q_top_seeb = q_top_seeb[0] if len(q_top_seeb) else np.nan
Qw         = Qw[0]         if len(Qw)         else np.nan
Qc         = Qc[0]         if len(Qc)         else np.nan

ratio_q_pct = safe_pct(q_top_seeb, q_top_cond)   # %
ratio_Q_pct = safe_pct(Qw, Qc)                   # %

# Put into the 4th column of the corresponding rows
df.loc[df["BC"] == "top_seebeck", "ratio [%]"] = ratio_q_pct
df.loc[df["BC"] == "water",      "ratio [%]"] = ratio_Q_pct

# ---------------- print ----------------
print(df.to_string(index=False, formatters={
    "q [W/m^2]": fmt_num,
    "Q' [W/m]": fmt_num,
    "ratio [%]": fmt_num,
}))

