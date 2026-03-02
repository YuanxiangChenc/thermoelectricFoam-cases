#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- settings --------
time_line = "1"
LINE_DIR = Path("postProcessing") / "sample" / time_line

FILES = {
    "seeb_top": LINE_DIR / "topSeebeck_T_alpha_gradTx_gradTy_gradTz_k_gradT_q.xy",
    "seeb_btm": LINE_DIR / "btmSeebeck_T_alpha_gradTx_gradTy_gradTz_k_gradT_q.xy",
}

# Column mapping for your sampled *.xy:
# 0:s, 1:T, 2:alpha, 3:gradTx, 4:gradTy, 5:gradTz, 6:k, ...
COL_S = 0
COL_K = 6
# --------------------------


def load_s_k(fp: Path):
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")

    data = np.loadtxt(fp)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] <= COL_K:
        raise ValueError(f"{fp}: expected at least {COL_K+1} columns, got {data.shape[1]}")

    s = data[:, COL_S]
    k = data[:, COL_K]
    return s, k


def mean_along_s(s, y):
    if len(s) < 2 or abs(s[-1] - s[0]) < 1e-30:
        return float(np.mean(y))
    L = s[-1] - s[0]
    return float(np.trapz(y, s) / L)


def main():
    plt.rcParams.update({"font.size": 18})

    # Load both
    curves = {}
    for name, fp in FILES.items():
        s, k = load_s_k(fp)
        k_mean = mean_along_s(s, k)
        curves[name] = (s, k, k_mean)

        # Also print to terminal
        print(f"{name}: k_mean = {k_mean:.6g}")

    # ---- One figure with both lines ----
    plt.figure()

    # Plot both datasets
    for name in ["seeb_top", "seeb_btm"]:
        s, k, k_mean = curves[name]
        # put mean in legend text
        plt.plot(s, k, marker="s", linestyle="-", label=f"{name} (k_mean={k_mean:.4g})")

    plt.xlabel("s")
    plt.ylabel("k")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Also write means as text on the plot (window)
    text_lines = [f"{name}: k_mean = {curves[name][2]:.6g}" for name in ["seeb_top", "seeb_btm"]]
    plt.text(
        0.02, 0.98,
        "\n".join(text_lines),
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(alpha=0.15)  # no fixed color
    )

    plt.show()


if __name__ == "__main__":
    main()
