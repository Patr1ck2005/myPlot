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
A_iso = -1.5  # S2 isotropic coefficient
A_4 = -1.0  # S2 fourfold anisotropy strength
inv_mass = 0.50  # S3 ~ ħ^2/(2m*)
E_offset_values = [-1.0, -.5, -0.2]  # Multiple E_offset for overlays

# Plotting configuration
PLOT_CONFIG = {
    "contour": {
        "cmap": "twilight",  # For S2 contour background
        "alpha": 0.7,  # Transparency for filled contours
        "levels": np.linspace(-2.5, 0, 10),  # Contour levels for S2
        "zero_color": "black",  # Zero contour color (for S2∩S1 reference)
        "zero_line_width": 1.0,
        "intersection_color": "green",  # For S2∩S3
        "intersection_line_width": 1.0,
        "contour_line_width": 1.0,
    },
}

fs = 9
plt.rcParams.update({"font.size": fs})


# =========================
# Surface definitions
# =========================
def S2_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    r2 = kx ** 2 + ky ** 2
    th = np.arctan2(ky, kx)
    return r2 * (A_iso + A_4 * np.cos(4.0 * th))


def S3_func(kx: np.ndarray, ky: np.ndarray, E_offset: float) -> np.ndarray:
    r2 = kx ** 2 + ky ** 2
    return inv_mass * r2 + E_offset


# =========================
# Intersection calculation (simplified for S2∩S3 only)
# =========================
def compute_s2_s3_intersections(E_offset: float) -> tuple:
    # Create kx-ky grid
    k = np.linspace(-KMAX, KMAX, NGRID)
    KX, KY = np.meshgrid(k, k, indexing="xy")

    # Compute diff = S2 - S3
    S2 = S2_func(KX, KY)
    S3 = S3_func(KX, KY, E_offset)
    diff = S2 - S3

    # Use contour to find zero level
    cs = plt.contour(KX, KY, diff, levels=[0])
    plt.close()  # Close temporary figure

    # Extract points
    paths = cs.get_paths()
    if not paths:
        print(f"[WARN] S2∩S3: No intersection found for E_offset={E_offset}")
        return None

    # Take the first path (assuming single closed curve)
    points = paths[0].vertices  # Shape: (N, 2) for kx, ky
    if len(points) < 2:
        print(f"[WARN] S2∩S3: Too few points ({len(points)}) for E_offset={E_offset}")
        return None

    # Sort points by polar angle
    kx, ky = points[:, 0], points[:, 1]
    theta = np.arctan2(ky, kx)
    sorted_indices = np.argsort(theta)
    sorted_points = points[sorted_indices]

    return ("S2∩S3", sorted_points, E_offset)


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
    zero_color = PLOT_CONFIG["contour"]["zero_color"]
    zero_line_width = PLOT_CONFIG["contour"]["zero_line_width"]
    inter_color = PLOT_CONFIG["contour"]["intersection_color"]
    inter_width = PLOT_CONFIG["contour"]["intersection_line_width"]
    contour_line_width = PLOT_CONFIG["contour"]["contour_line_width"]

    # Compute S2∩S3 intersections for multiple E_offset
    intersections = []
    for E_offset in E_offset_values:
        inter = compute_s2_s3_intersections(E_offset)
        if inter:
            intersections.append(inter)

    # 1. With axes
    fig, ax = plt.subplots(figsize=(2, 2))

    # Background: S2 contour (filled)
    cs = ax.contourf(KX, KY, S2, levels=levels, cmap=cmap, alpha=alpha)
    ax.contour(KX, KY, S2, levels=levels, colors="black", linewidths=contour_line_width)  # Outlines

    # Overlay S2∩S3 intersections
    for label, points, param in intersections:
        ax.plot(points[:, 0], points[:, 1], color=inter_color, linewidth=inter_width,
                label=f"{label} (E_offset={param:.1f})")
        ax.plot(points[[0, -1], 0], points[[0, -1], 1], color=inter_color, linewidth=inter_width)  # Close

    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title("S2 Contour with S2∩S3 Overlays")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(-KMAX, KMAX)
    ax.set_ylim(-KMAX, KMAX)
    ax.set_aspect("equal")
    plt.colorbar(cs, ax=ax, label="S2 Energy")
    plt.savefig("contour_with_axes.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Saved contour_with_axes.svg")

    # 2. Without axes
    fig, ax = plt.subplots(figsize=(1.5, 1.5))

    # Background: S2 contour
    ax.contourf(KX, KY, S2, levels=levels, cmap=cmap, alpha=alpha)
    # ax.contour(KX, KY, S2, levels=levels, colors="black", linewidths=contour_line_width)

    # Overlay S2∩S3 intersections
    for _, points, _ in intersections:
        ax.plot(points[:, 0], points[:, 1], color=inter_color, linewidth=inter_width)
        ax.plot(points[[0, -1], 0], points[[0, -1], 1], color=inter_color, linewidth=inter_width)  # Close

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
