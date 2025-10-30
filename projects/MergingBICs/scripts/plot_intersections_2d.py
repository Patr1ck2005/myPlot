#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified 2D contour plotter for band surfaces.
Focuses on S2 contour background with S2∩S3 overlays for varying E_offset.
Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Physics parameters
# =========================
KMAX = 1.0  # k-space range [-KMAX, KMAX]
NGRID = 240  # grid density

# ---------- S2 ----------
S2_iso_coeff = -0.7  # S2 isotropic coefficient
S2_aniso_strength = -0  # S2 fourfold anisotropy strength
# ---------- S3 ----------
S3_iso_coeff = -0.3  # S3 isotropic coefficient (originally inv_mass)
S3_aniso_strength = 0.8  # S3 fourfold anisotropy strength (NEW)
S3_phase_shift = np.pi
S3_energy_shift_values = [-0.2, -0.1, 0.1, 0.2]  # Multiple energy shifts for overlays

# Plotting configuration
PLOT_CONFIG = {
    "contour": {
        "cmap": "twilight",  # For S2 contour background
        "alpha": 0.7,  # Transparency for filled contours
        "levels": np.linspace(-2.5, 0, 8),  # Contour levels for S2
        "zero_color": "black",  # Zero contour color (for S2∩S1 reference)
        "zero_line_width": 1.0,
        "intersection_color": "green",  # For S2∩S3
        "intersection_line_width": 1.0,
        "contour_line_width": 1.0*0,
    },
}

fs = 9
plt.rcParams.update({"font.size": fs})

# =========================
# Surface definitions
# =========================
def S2_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Real dispersion surface with isotropic and fourfold anisotropic terms."""
    r2 = kx ** 2 + ky ** 2
    th = np.arctan2(ky, kx)
    return r2 * (S2_iso_coeff + S2_aniso_strength * np.cos(4.0 * th))

def S3_func(kx: np.ndarray, ky: np.ndarray, energy_shift: float) -> np.ndarray:
    """Auxiliary surface with isotropic, anisotropic terms, and energy shift."""
    r2 = kx ** 2 + ky ** 2
    th = np.arctan2(ky, kx)
    return r2 * (S3_iso_coeff + S3_aniso_strength * np.cos(4.0 * th + S3_phase_shift)) + energy_shift

from matplotlib.path import Path as MplPath

# =========================
# Intersection calculation (simplified for S2∩S3 only)
# =========================
def compute_s2_s3_intersections(energy_shift: float):
    # Create kx-ky grid
    k = np.linspace(-KMAX, KMAX, NGRID)
    KX, KY = np.meshgrid(k, k, indexing="xy")

    # Compute diff = S2 - S3
    S2 = S2_func(KX, KY)
    S3 = S3_func(KX, KY, energy_shift)
    diff = S2 - S3

    # Use contour to find zero level
    cs = plt.contour(KX, KY, diff, levels=[0])
    plt.close()

    paths = cs.get_paths()
    if not paths:
        print(f"[WARN] S2∩S3: No intersection found for energy_shift={energy_shift}")
        return []

    segments = []
    for p in paths:
        verts = p.vertices
        codes = p.codes

        if verts is None or len(verts) < 2:
            continue

        # 若 codes 为空，视为单段开放曲线
        if codes is None:
            segments.append(("S2∩S3", verts.copy(), energy_shift, False))
            continue

        # --- 关键：按 codes 拆分子路径 ---
        cur = []
        for v, c in zip(verts, codes):
            if c == MplPath.MOVETO:
                # 开启新段，先把上一段收集起来
                if len(cur) >= 2:
                    is_closed = np.allclose(cur[0], cur[-1], atol=1e-12)
                    segments.append(("S2∩S3", np.asarray(cur), energy_shift, is_closed))
                cur = [v]
            elif c == MplPath.LINETO:
                cur.append(v)
            elif c == MplPath.CLOSEPOLY:
                # CLOSEPOLY：闭合当前段
                if len(cur) >= 2:
                    # Matplotlib 有时会在 CLOSEPOLY 处给一个“哑”点；几何上首尾视为闭合
                    is_closed = True
                    segments.append(("S2∩S3", np.asarray(cur), energy_shift, is_closed))
                cur = []
            else:
                # 其他曲线代码在等值线中基本不会出现；保守处理为断开
                if len(cur) >= 2:
                    is_closed = np.allclose(cur[0], cur[-1], atol=1e-12)
                    segments.append(("S2∩S3", np.asarray(cur), energy_shift, is_closed))
                cur = []

        # 收尾：最后一段（若没有 CLOSEPOLY）
        if len(cur) >= 2:
            is_closed = np.allclose(cur[0], cur[-1], atol=1e-12)
            segments.append(("S2∩S3", np.asarray(cur), energy_shift, is_closed))

    return segments


# =========================
# Plot contour map (simplified)
# =========================
def plot_contour_map():
    # Create kx-ky grid
    k = np.linspace(-KMAX, KMAX, NGRID)
    KX, KY = np.meshgrid(k, k, indexing="xy")

    # Compute S2 for background
    S2 = S2_func(KX, KY)

    # Plot configuration
    cmap = PLOT_CONFIG["contour"]["cmap"]
    alpha = PLOT_CONFIG["contour"]["alpha"]
    levels = PLOT_CONFIG["contour"]["levels"]
    inter_color = PLOT_CONFIG["contour"]["intersection_color"]
    inter_width = PLOT_CONFIG["contour"]["intersection_line_width"]
    contour_line_width = PLOT_CONFIG["contour"]["contour_line_width"]

    # 收集所有能量位移的分段
    intersections = []
    for shift in S3_energy_shift_values:
        segs = compute_s2_s3_intersections(shift)
        if segs:
            intersections.extend(segs)

    # ------------------------------
    # 1. With axes
    # ------------------------------
    fig, ax = plt.subplots(figsize=(1, 1))
    cs = ax.contourf(KX, KY, S2, levels=levels, cmap=cmap, alpha=alpha)
    if contour_line_width > 0:
        ax.contour(KX, KY, S2, levels=levels, colors="black", linewidths=contour_line_width)

    # 每一段单独画，绝不跨段连线；只在 is_closed=True 且首尾未重复时补闭合
    for i, (label, pts, param, is_closed) in enumerate(intersections):
        if len(pts) < 2:
            continue
        # 只给第一段加图例，避免重复
        lbl = f"{label} (shift={param:.1f})" if i == 0 else None
        ax.plot(pts[:, 0], pts[:, 1], color=inter_color, linewidth=inter_width, label=lbl)
        if is_closed and not np.allclose(pts[0], pts[-1], atol=1e-12):
            ax.plot([pts[0, 0], pts[-1, 0]], [pts[0, 1], pts[-1, 1]], color=inter_color, linewidth=inter_width)

    ax.set_xlim(-KMAX, KMAX)
    ax.set_ylim(-KMAX, KMAX)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    plt.savefig("contour_with_axes.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Saved contour_with_axes.svg")

    # ------------------------------
    # 2. Without axes
    # ------------------------------
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.contourf(KX, KY, S2, levels=levels, cmap=cmap, alpha=alpha)
    if contour_line_width > 0:
        ax.contour(KX, KY, S2, levels=levels, colors="black", linewidths=contour_line_width)

    for (_, pts, _, is_closed) in intersections:
        if len(pts) < 2:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color=inter_color, linewidth=inter_width)
        if is_closed and not np.allclose(pts[0], pts[-1], atol=1e-12):
            ax.plot([pts[0, 0], pts[-1, 0]], [pts[0, 1], pts[-1, 1]], color=inter_color, linewidth=inter_width)

    ax.axis("off")
    ax.set_aspect("equal")
    plt.savefig("contour_clean.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Saved contour_clean.svg")


# =========================
# Main execution (simplified)
# =========================
if __name__ == "__main__":
    plot_contour_map()
