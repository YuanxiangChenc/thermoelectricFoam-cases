#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
from pathlib import Path

# ---------------- user settings ----------------
time_line  = "1"
time_patch = "1"

LINE_DIR  = Path("postProcessing") / "sample" / time_line
PATCH_DIR = Path("postProcessing") / "sample_patch" / time_patch

LINE_FILES = {
    "seeb_top": LINE_DIR / "topSeebeck_T_alpha_gradTx_gradTy_gradTz_k_gradT_q.xy",
    "seeb_btm": LINE_DIR / "btmSeebeck_T_alpha_gradTx_gradTy_gradTz_k_gradT_q.xy",
}

TEMP_LINE_FILES = {
    "cond_top": LINE_DIR / "topConductor_T_alpha_gradTx_gradTy_gradTz_k_gradT_q.xy",
    "seeb_top": LINE_FILES["seeb_top"],
    "seeb_btm": LINE_FILES["seeb_btm"],
}

PATCH_QN = {
    "cond_top": "TOP",
    "water": "WATER",
    "air": "AIR",
}

LOG_FILE = Path("log")

NORMAL_COMP = "y"  # "x" or "y" or "z"

MATPROPS = Path("constant") / "materialProperties"
MAT_NAME = "seebeckcell"
# ------------------------------------------------


def _idx_for_component(comp: str) -> int:
    comp = comp.lower()
    if comp == "x":
        return 0
    if comp == "y":
        return 1
    if comp == "z":
        return 2
    raise ValueError(f"Unknown component '{comp}', use 'x'/'y'/'z'.")


def fmt_signed(x, nd=6):
    if x is None:
        return ""
    return f"{x:+.{nd}g}"


def fmt_T(x):
    """No sign, fixed 1 decimal: xxx.x"""
    if x is None:
        return ""
    return f"{x:.1f}"


def mean_along_s(s: np.ndarray, y: np.ndarray) -> float:
    if len(s) < 2 or abs(s[-1] - s[0]) < 1e-30:
        return float(np.mean(y))
    L = s[-1] - s[0]
    return float(np.trapz(y, s) / L)


def integ_along_s(s: np.ndarray, y: np.ndarray) -> float:
    if len(s) < 2:
        return 0.0
    return float(np.trapz(y, s))


def safe_pct(num, den):
    if num is None or den is None or abs(den) < 1e-30:
        return None
    return 100.0 * (num / den)


def read_material_J(matprops_path: Path, material_name: str) -> np.ndarray:
    if not matprops_path.exists():
        raise FileNotFoundError(f"Missing {matprops_path}")

    txt = matprops_path.read_text(errors="ignore")

    m_block = re.search(rf"\b{re.escape(material_name)}\s*\{{(.*?)\n\s*\}}", txt, flags=re.S)
    if not m_block:
        raise ValueError(f"Cannot find material block '{material_name}' in {matprops_path}")

    block = m_block.group(1)

    m_j = re.search(
        r"\bj\s*\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)\s*;",
        block
    )
    if not m_j:
        raise ValueError(f"Cannot find 'j (.. .. ..);' inside '{material_name}' in {matprops_path}")

    return np.array([float(m_j.group(1)), float(m_j.group(2)), float(m_j.group(3))], dtype=float)


def read_line_T_stats(fp: Path):
    if not fp.exists():
        return None
    data = np.loadtxt(fp)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    s = data[:, 0]
    T = data[:, 1]
    return {"T_mean": mean_along_s(s, T), "T_max": float(np.max(T))}


def read_line_xy_seebeck_parts(fp: Path, Jn: float):
    if not fp.exists():
        return None

    data = np.loadtxt(fp)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 7:
        raise ValueError(f"{fp}: too few columns ({data.shape[1]}). Need at least 7.")

    s = data[:, 0]
    T = data[:, 1]
    alpha = data[:, 2]
    gradT = data[:, 3:6]
    k = data[:, 6]

    comp = _idx_for_component(NORMAL_COMP)
    dTdn = gradT[:, comp]

    q_c = -k * dTdn
    q_e = alpha * Jn * T
    q_n = q_c + q_e

    return {
        "T_mean": mean_along_s(s, T),
        "T_max": float(np.max(T)),
        "q_c": mean_along_s(s, q_c),
        "q_e": mean_along_s(s, q_e),
        "q_n": mean_along_s(s, q_n),
        "Q": integ_along_s(s, q_n),
    }


def read_raw_scalar(fp: Path) -> np.ndarray:
    vals = []
    with fp.open("r") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            vals.append(float(parts[3]))
    return np.array(vals, dtype=float)


def patch_from_qn(patch: str):
    fp_qn = PATCH_DIR / f"qn_{patch}.raw"
    fp_A  = PATCH_DIR / f"magSf_{patch}.raw"

    if not fp_qn.exists() or not fp_A.exists():
        return None, None

    qn = read_raw_scalar(fp_qn)
    A  = read_raw_scalar(fp_A)

    Atot = float(A.sum())
    if Atot < 1e-30:
        return 0.0, 0.0

    Q = float(np.sum(qn * A))
    qmean = Q / Atot
    return qmean, Q


def parse_electric_Pnet(log_fp: Path):
    if not log_fp.exists():
        return None
    text = log_fp.read_text(errors="ignore")
    m = re.findall(
        r"\[electricPower\](?:.|\n)*?Pnet\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*W",
        text,
    )
    return float(m[-1]) if m else None


# ---------------- main ----------------
ORDER = ["cond_top", "water", "air", "seeb_top", "seeb_btm", "Q.e", "net"]

Jvec = read_material_J(MATPROPS, MAT_NAME)
Jn = float(Jvec[_idx_for_component(NORMAL_COMP)])
print(f"[info] J = ({Jvec[0]:g} {Jvec[1]:g} {Jvec[2]:g}), normal={NORMAL_COMP}, Jn={Jn:g}")

results = {bc: {"T_mean": None, "T_max": None, "q_c": None, "q_e": None, "q_n": None, "Q": None, "ratio": None}
           for bc in ORDER}

patch_Q_sum = 0.0
for bc, patch in PATCH_QN.items():
    qmean, Q = patch_from_qn(patch)
    results[bc]["q_n"] = qmean
    results[bc]["Q"] = Q
    if Q is not None:
        patch_Q_sum += Q

for bc, fp in LINE_FILES.items():
    out = read_line_xy_seebeck_parts(fp, Jn=Jn)
    if out is not None:
        results[bc].update(out)

for bc, fp in TEMP_LINE_FILES.items():
    ts = read_line_T_stats(fp)
    if ts is not None:
        results[bc]["T_mean"] = ts["T_mean"]
        results[bc]["T_max"] = ts["T_max"]

Pnet = parse_electric_Pnet(LOG_FILE)
results["Q.e"]["Q"] = Pnet
results["net"]["Q"] = patch_Q_sum + (Pnet or 0.0)

# ratios same as old code
results["seeb_top"]["ratio"] = safe_pct(results["seeb_top"]["q_n"], results["cond_top"]["q_n"])
results["water"]["ratio"] = safe_pct(results["water"]["Q"], results["cond_top"]["Q"])
results["Q.e"]["ratio"] = safe_pct(
    (results["seeb_top"]["Q"] - results["seeb_btm"]["Q"])
    if (results["seeb_top"]["Q"] is not None and results["seeb_btm"]["Q"] is not None)
    else None,
    results["seeb_top"]["Q"]
)

# ---------------- Print tables ----------------

def fmt_q_int(x):
    """Format q values as signed integer."""
    if x is None:
        return ""
    return f"{x:+.0f}"

def fmt_Q(x, bc=None):
    """Format Q' values."""
    if x is None:
        return ""
    if bc == "net":
        return f"{x:+.6f}"
    return f"{x:+.0f}"

def fmt_ratio(x):
    if x is None:
        return ""
    return f"{x:+.3g}"

def fmt_expr(num, den):
    if num is None or den is None:
        return ""
    return f"{num:+.0f} / {den:+.0f}"

def fmt_pct(num, den):
    if num is None or den is None or abs(den) < 1e-30:
        return ""
    return f"{100.0 * num / den:+.3g}"

def print_block_title(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# 1) SYSTEM BALANCE
# ============================================================
print_block_title("SYSTEM BALANCE")

header1 = "{:<12} | {:>10} | {:>10}".format("BC", "q_n", "Q'")
print(header1)
print("-" * len(header1))

for bc in ["cond_top", "water", "air", "Q.e"]:
    r = results[bc]
    qn = "" if r["q_n"] is None else f"{r['q_n']:+.0f}"
    Q  = fmt_Q(r["Q"], bc)
    print(f"{bc:<12} | {qn:>10} | {Q:>10}")


# ============================================================
# 2) SEEBECK
# ============================================================
print_block_title("SEEBECK")

header2 = "{:<12} | {:>8} | {:>10} | {:>10} | {:>10}".format(
    "side", "Tmax", "q_cond", "q_ele", "q_n"
)
print(header2)
print("-" * len(header2))

side_map = {
    "hot side": "seeb_top",
    "cold side": "seeb_btm",
}

for side_name, key in side_map.items():
    r = results[key]
    Tmax = fmt_T(r["T_max"])
    qcond = fmt_q_int(r["q_c"])
    qele  = fmt_q_int(r["q_e"])
    qn    = "" if r["q_n"] is None else f"{r['q_n']:+.0f}"
    print(f"{side_name:<12} | {Tmax:>8} | {qcond:>10} | {qele:>10} | {qn:>10}")


# ============================================================
# 3) RATIOS
# ============================================================
print_block_title("RATIOS")

Q_cond  = results["cond_top"]["Q"]
Q_water = results["water"]["Q"]
Q_air   = results["air"]["Q"]
Qe      = results["Q.e"]["Q"]

q_cond_top = results["cond_top"]["q_n"]
q_seeb_cond = results["seeb_top"]["q_c"]
q_seeb_net  = results["seeb_top"]["q_n"]

# conductor heat flux ratio
cond_num_wo = q_seeb_cond
cond_den_wo = q_cond_top
cond_num_w  = q_seeb_net
cond_den_w  = q_cond_top

# system heat flow ratio
# without Q.e : (Q_water + Q_air) / Q_cond
sys_num_wo = Q_water
sys_den_wo = Q_cond

# with Q.e : Q_water / (Q_cond + Q.e)
sys_num_w = Q_water
sys_den_w = (Q_cond or 0.0) + (Qe or 0.0)

header3 = "{:<28} | {:>22} | {:>10} | {:>22} | {:>10}".format(
    "value [%]", "without Q.e", "result", "with Q.e", "result"
)
print(header3)
print("-" * len(header3))

print(
    f"{'conductor heat flux ratio':<28} | "
    f"{fmt_expr(cond_num_wo, cond_den_wo):>22} | "
    f"{fmt_pct(cond_num_wo, cond_den_wo):>10} | "
    f"{fmt_expr(cond_num_w, cond_den_w):>22} | "
    f"{fmt_pct(cond_num_w, cond_den_w):>10}"
)

print(
    f"{'system heat flow ratio':<28} | "
    f"{fmt_expr(sys_num_wo, sys_den_wo):>22} | "
    f"{fmt_pct(sys_num_wo, sys_den_wo):>10} | "
    f"{fmt_expr(sys_num_w, sys_den_w):>22} | "
    f"{fmt_pct(sys_num_w, sys_den_w):>10}"
)

print(
    f"{'seebeck efficiency':<28} | "
    f"{'':>22} | "
    f"{'':>10} | "
    f"{'':>22} | "
    f"{fmt_ratio(results['Q.e']['ratio']):>10}"
)
