#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular & highly-configurable band-surface visualizer (PyVista/VTK, no argparse).

Surfaces in k-space (kx, ky):
  S1 (reference plane, imaginary):            E = S1_energy_level
  S2 (real dispersion):                       E2(k) = k^2 * (S2_iso_coeff + S2_aniso_strength * cos(4θ))
  S3 (auxiliary surface, imaginary):          E3(k) = k^2 * (S3_iso_coeff + S3_aniso_strength * cos(4θ)) + S3_energy_shift

Key ideas:
- All behavior driven by CONFIG (visibility, color mapping, opacity, etc.).
- Pluggable colorizers; per-surface polar mode; robust PyVista axes customization.
- Outputs: PNG screenshot with PyVista's native, clean axes.
- No Matplotlib post-processing, no SVG output.

Requires: pip install pyvista pyvistaqt
"""

from __future__ import annotations
import os
import csv
import numpy as np
import pyvista as pv
from typing import Callable, Dict, Tuple, List, Optional

# =========================
# Physics parameters
# =========================
# S1_energy_level = -0.6  # S1 reference plane energy (imaginary helper)
# S2_iso_coeff = -1.5  # S2 isotropic coefficient for k^2 term
# S2_aniso_strength = -1.0  # S2 fourfold anisotropy strength for cos(4θ)
# S3_iso_coeff = 0.8  # S3 isotropic coefficient for k^2 term
# S3_aniso_strength = 0.0  # S3 fourfold anisotropy strength for cos(4θ)
# S3_energy_shift = -1.0  # S3 energy offset for the surface

S1_energy_level = -0.6  # S1 reference plane energy (imaginary helper)
S2_iso_coeff = -0.7  # S2 isotropic coefficient for k^2 term
S2_aniso_strength = -0.0  # S2 fourfold anisotropy strength for cos(4θ)
S3_iso_coeff = -0.3  # S3 isotropic coefficient for k^2 term
S3_aniso_strength = 0.8  # S3 fourfold anisotropy strength for cos(4θ)
S3_energy_shift = 0.2-0.5  # S3 energy offset for the surface

# k-space sampling
KMAX = 1.0  # Range [-KMAX, KMAX]
NGRID = 240  # Grid density

# =========================
# Colorizers (return [0,1] scalar field)
# =========================
def colorize_height(kx, ky, z):
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if np.isclose(zmax, zmin): return np.zeros_like(z, dtype=float)
    return (z - zmin) / (zmax - zmin)

def colorize_radial(kx, ky, z):
    r = np.sqrt(kx ** 2 + ky ** 2)
    rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
    if np.isclose(rmax, rmin): return np.zeros_like(r, dtype=float)
    return (r - rmin) / (rmax - rmin)

COLORIZER_REGISTRY: Dict[str, Callable] = {
    "height": colorize_height,
    "radial": colorize_radial,
}

# =========================
# Surface definitions
# =========================
def S1_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Reference plane at fixed energy level."""
    return np.full_like(kx, S1_energy_level, dtype=float)

def S2_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Real dispersion surface with isotropic and fourfold anisotropic terms."""
    r2 = kx ** 2 + ky ** 2
    th = np.arctan2(ky, kx)
    return r2 * (S2_iso_coeff + S2_aniso_strength * np.cos(4.0 * th))

def S3_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Auxiliary surface with isotropic, anisotropic terms, and energy shift."""
    r2 = kx ** 2 + ky ** 2
    th = np.arctan2(ky, kx)
    return r2 * (S3_iso_coeff + S3_aniso_strength * np.cos(4.0 * th)) + S3_energy_shift

SURFACE_FUNCS = {
    "S1": S1_func,
    "S2": S2_func,
    "S3": S3_func,
}

# =========================
# MASTER CONFIG
# =========================
# # For isotropic case
# r_X = 0.66
# r_M = 0.66

# # for anisotropic case
# r_X = 0.55
# r_M = 0.90

# # for ani-anisotropic case 1
# r_X = 2
# r_M = 0.70

# for ani-anisotropic case 2
r_X = 0.50
r_M = 2

CONFIG = {
    "scene": {
        "window_size": (1024, 1024),
        "background": "white",
        "theme": "document",
        "depth_peeling": True,
        "camera": {"preset": "iso", "elev": 25, "azim": 35},
        "show_bounds": True,
        "axes": True,
        "clean_view": True,
        "pure_surface_only": False,
        "anti_aliasing": "msaa",
        "pyvista_axes_config": {
            "show_axes_labels": True,
            "axes_line_width": 2,
            "bounds_grid": False,
            "bounds_location": "outer",
            "bounds_all_edges": False,
            "bounds_font_size": 18,
            "bounds_font_family": "arial",
            "bounds_color": "black",
            "bounds_ticks": "outside",
            "bounds_xtitle": "k_x",
            "bounds_ytitle": "k_y",
            "bounds_ztitle": "E",
        },
    },
    "surfaces": {
        "S1": {
            "visible": True,
            "mode": "solid",
            "solid_color": (0.55, 0.55, 0.55),
            "opacity": 0.30,
            "smooth_shading": True,
            "cmap": "viridis",
            "colorizer": None,
            "scalar_bar": False,
            "line_width": 1.0,
            "lighting": True,
            "ambient": 0.2,
            "specular": 0.0,
            "polar_mode": False,
        },
        "S2": {
            "visible": True,
            "mode": "colormap",
            "solid_color": (0.2, 0.2, 0.2),
            "opacity": 0.95,
            "smooth_shading": True,
            "cmap": "twilight",
            "colorizer": "height",
            "scalar_bar": False,
            "line_width": 1.0,
            "lighting": True,
            "ambient": 0.2,
            "specular": 0.0,
            "polar_mode": False,
        },
        "S3": {
            "visible": True,
            "mode": "solid",
            "solid_color": "lawngreen",
            "opacity": 0.40,
            "smooth_shading": True,
            "cmap": "viridis",
            "colorizer": None,
            "scalar_bar": False,
            "line_width": 1.0,
            "lighting": True,
            "ambient": 0.2,
            "specular": 0.0,
            "polar_mode": False,
        },
    },
    "intersections": {
        "draw": True,
        "pairs": [
            {"surface1": "S2", "surface2": "S1", "color": "white", "line_width": 15.0},
            {"surface1": "S2", "surface2": "S3", "color": "lawngreen", "line_width": 15.0},
        ],
        "tol_rel": 1e-3,
        "export": {
            "folder": None,
            "vtk": False,
            "csv": False,
        }
    },
    "spheres": [
        {"center": (r_M/np.sqrt(2), r_M/np.sqrt(2), None), "radius": 0.1, "color": "magenta", "opacity": 1},
        {"center": (r_M/np.sqrt(2), -r_M/np.sqrt(2), None), "radius": 0.1, "color": "magenta", "opacity": 1},
        {"center": (-r_M/np.sqrt(2), -r_M/np.sqrt(2), None), "radius": 0.1, "color": "magenta", "opacity": 1},
        {"center": (-r_M/np.sqrt(2), r_M/np.sqrt(2), None), "radius": 0.1, "color": "magenta", "opacity": 1},
        {"center": (r_X, 0, None), "radius": 0.1, "color": "deepskyblue", "opacity": 1},
        {"center": (-r_X, 0, None), "radius": 0.1, "color": "deepskyblue", "opacity": 1},
        {"center": (0, -r_X, None), "radius": 0.1, "color": "deepskyblue", "opacity": 1},
        {"center": (0, r_X, None), "radius": 0.1, "color": "deepskyblue", "opacity": 1},
    ],
    "output": {
        "screenshot": {
            "path": './temp.png',
            "transparent_background": True,
            "camera_azimuth": 10,
            "camera_elevation": 15,
            "parallel_projection": True,
        },
        "allow_empty_mesh": False,
    },
}

# =========================
# Utility functions
# =========================
def make_grid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> pv.StructuredGrid:
    return pv.StructuredGrid(X, Y, Z)

def compute_color_scalars(kind, kx, ky, z):
    if kind is None:
        return None
    fn = kind if callable(kind) else COLORIZER_REGISTRY.get(kind)
    if fn is None:
        raise ValueError(f"Unknown colorizer: {kind}")
    scal = fn(kx, ky, z)
    return np.ascontiguousarray(scal.ravel(order="F"))

def _zero_contour(mesh: pv.StructuredGrid, F: np.ndarray, name: str, tol_rel: float):
    F = F.astype(np.float32)
    m = mesh.copy()
    m[name] = F.ravel(order="F")
    mn, mx = float(np.nanmin(F)), float(np.nanmax(F))
    rng = mx - mn
    if rng <= 0 or np.isnan(rng):
        return None, None
    if mn <= 0.0 <= mx:
        poly = m.contour([0.0], scalars=name)
        if poly and poly.n_points > 0: return poly, 0.0
    eps = tol_rel * rng
    trials = [eps, -eps, 0.5 * eps, -0.5 * eps, float(F.ravel()[np.nanargmin(np.abs(F))])]
    for val in trials:
        poly = m.contour([val], scalars=name)
        if poly and poly.n_points > 0: return poly, val
    return None, None

def extract_intersections(meshes: Dict[str, pv.StructuredGrid],
                          fields: Dict[str, np.ndarray],
                          pairs: List[Dict[str, any]], tol_rel: float) -> List[Tuple[str, pv.PolyData, Optional[float], Dict]]:
    out = []
    for pconf in pairs:
        a, b = pconf["surface1"], pconf["surface2"]
        grid_a = meshes[a]
        Za = fields[a]
        if meshes[b].dimensions != grid_a.dimensions:
            print(f"[WARN] {b} grid differs from {a}; recomputing {b} on {a}'s grid.")
            x, y, _ = grid_a.points.T.reshape(3, *grid_a.dimensions[:2])
            Zb = SURFACE_FUNCS[b](x, y)
        else:
            Zb = fields[b]
        poly, used = _zero_contour(grid_a, Za - Zb, f"f_{a}_{b}", tol_rel)
        out.append((f"{a}∩{b}", poly, used, pconf))
    return out

def ensure_folder(path: Optional[str]):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def export_poly(poly: pv.PolyData, basename: str, folder: str, want_vtk: bool, want_csv: bool):
    if not poly or poly.n_points == 0: return
    ensure_folder(folder)
    if want_vtk:
        poly.save(os.path.join(folder, f"{basename}.vtp"))
    if want_csv:
        pts = np.asarray(poly.points)
        with open(os.path.join(folder, f"{basename}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kx", "ky", "E"])
            w.writerows(pts.tolist())

# =========================
# Build & render
# =========================
def main():
    out = CONFIG["output"]
    if out.get("allow_empty_mesh", False):
        pv.global_theme.allow_empty_mesh = True

    sc = CONFIG["scene"]
    pv.set_plot_theme(sc.get("theme", "document"))
    off_screen = out.get("screenshot", {}).get("path", None)
    plotter = pv.Plotter(off_screen=off_screen, window_size=sc.get("window_size", (1200, 900)))
    plotter.set_background(sc.get("background", "white"))
    if sc.get("depth_peeling", False):
        plotter.enable_depth_peeling()
    aa_mode = sc.get("anti_aliasing", None)
    if aa_mode:
        plotter.enable_anti_aliasing(aa_mode)

    # per-surface grids
    Z_fields = {}
    meshes = {}
    for name, sconf in CONFIG["surfaces"].items():
        if not sconf.get("visible", True):
            continue
        polar = sconf.get("polar_mode", False)
        if polar:
            print(f"[INFO] {name}: Using polar mode (r <= KMAX)")
            r = np.linspace(0, KMAX, NGRID)
            theta = np.linspace(0, 2 * np.pi, int(NGRID * 1.5))
            R, THETA = np.meshgrid(r, theta, indexing="ij")
            KX = R * np.cos(THETA)
            KY = R * np.sin(THETA)
        else:
            k = np.linspace(-KMAX, KMAX, NGRID)
            KX, KY = np.meshgrid(k, k, indexing="xy")
        Z = SURFACE_FUNCS[name](KX, KY)
        Z_fields[name] = Z
        meshes[name] = make_grid(KX, KY, Z)

    # add surfaces
    for name, sconf in CONFIG["surfaces"].items():
        if not sconf.get("visible", True):
            continue
        grid = meshes[name]
        mode = sconf.get("mode", "colormap")
        opacity = float(sconf.get("opacity", 1.0))
        smooth = bool(sconf.get("smooth_shading", True))
        line_width = float(sconf.get("line_width", 1.0))
        lighting = bool(sconf.get("lighting", False))
        ambient = float(sconf.get("ambient", 0.2))
        specular = float(sconf.get("specular", 0.0))

        if mode == "colormap":
            scal = compute_color_scalars(sconf.get("colorizer", None), grid.x, grid.y, Z_fields[name])
            if scal is None:
                raise ValueError(f"{name}: mode 'colormap' requires a colorizer")
            grid["color"] = scal
            plotter.add_mesh(
                grid,
                scalars="color",
                cmap=sconf.get("cmap", "viridis"),
                opacity=opacity,
                smooth_shading=smooth,
                show_scalar_bar=bool(sconf.get("scalar_bar", False)),
                name=name,
                lighting=lighting,
                ambient=ambient,
                specular=specular,
            )
        elif mode == "solid":
            plotter.add_mesh(
                grid,
                color=sconf.get("solid_color", (0.6, 0.6, 0.6)),
                opacity=opacity,
                smooth_shading=smooth,
                name=name,
                lighting=lighting,
                ambient=ambient,
                specular=specular,
            )
        elif mode == "wireframe":
            plotter.add_mesh(
                grid,
                color=sconf.get("solid_color", (0.0, 0.0, 0.0)),
                opacity=opacity,
                style="wireframe",
                line_width=line_width,
                smooth_shading=smooth,
                name=name,
                lighting=lighting,
                ambient=ambient,
                specular=specular,
            )
        else:
            raise ValueError(f"Unknown mode for {name}: {mode}")

    # intersections
    iconf = CONFIG["intersections"]
    if iconf.get("draw", True):
        pairs = iconf.get("pairs", [])
        tol_rel = float(iconf.get("tol_rel", 1e-3))
        curves = extract_intersections(meshes, Z_fields, pairs, tol_rel)
        for label, poly, used, pconf in curves:
            if poly is not None and poly.n_points > 0:
                plotter.add_mesh(poly,
                                 color=pconf.get("color", "black"),
                                 line_width=float(pconf.get("line_width", 3.0)),
                                 name=label)
                print(f"[INFO] drew {label} (iso ~ {0.0 if used is None else used:.3e})")
                exp = iconf.get("export", {})
                folder = exp.get("folder")
                if folder:
                    export_poly(poly, label.replace("∩", "_"), folder,
                                want_vtk=bool(exp.get("vtk", False)),
                                want_csv=bool(exp.get("csv", False)))
            else:
                print(f"[WARN] {label}: empty. Skipped.")

    # spheres
    if not sc.get("pure_surface_only", False):
        for i, sph in enumerate(CONFIG.get("spheres", []), start=1):
            cx, cy, cz = sph.get("center", (0.0, 0.0, None))
            r_sph = np.sqrt(cx ** 2 + cy ** 2)
            if r_sph > KMAX:
                print(f"[WARN] Sphere {i} at r={r_sph:.2f} > KMAX; skipped.")
                continue
            if cz is None:
                zval = float(S2_func(np.array([[cx]]), np.array([[cy]]))[0, 0])
            else:
                zval = float(cz)
            sphere = pv.Sphere(
                radius=float(sph.get("radius", 0.06)),
                center=(cx, cy, zval),
                theta_resolution=48, phi_resolution=48,
            )
            plotter.add_mesh(
                sphere,
                color=sph.get("color", None),
                opacity=float(sph.get("opacity", 0.9)),
                smooth_shading=True,
                name=f"sphere{i}",
                lighting=False
            )

    # axes and bounds
    clean_view = sc.get("clean_view", False) or sc.get("pure_surface_only", False)
    pyvista_ax_conf = sc.get("pyvista_axes_config", {})
    if not clean_view:
        if sc.get("axes", True):
            plotter.add_axes(
                line_width=pyvista_ax_conf.get("axes_line_width", 2),
                labels_off=not pyvista_ax_conf.get("show_axes_labels", True)
            )
        if sc.get("show_bounds", True):
            plotter.show_bounds(
                grid=pyvista_ax_conf.get("bounds_grid", False),
                location=pyvista_ax_conf.get("bounds_location", "outer"),
                all_edges=pyvista_ax_conf.get("bounds_all_edges", False),
                font_size=pyvista_ax_conf.get("bounds_font_size", 18),
                font_family=pyvista_ax_conf.get("bounds_font_family", "arial"),
                color=pyvista_ax_conf.get("bounds_color", "black"),
                ticks=pyvista_ax_conf.get("bounds_ticks", "outside"),
                xtitle=pyvista_ax_conf.get("bounds_xtitle", "k_x"),
                ytitle=pyvista_ax_conf.get("bounds_ytitle", "k_y"),
                ztitle=pyvista_ax_conf.get("bounds_ztitle", "E"),
            )

    # camera
    cam = sc.get("camera", {})
    if cam.get("preset", "iso") == "iso":
        plotter.camera_position = "iso"
    plotter.camera.elevation = float(cam.get("elev", 25))
    plotter.camera.azimuth = float(cam.get("azim", 35))

    # output
    ss_conf = out.get("screenshot", {})
    screenshot_path = ss_conf.get("path", None)
    if screenshot_path:
        plotter.camera.azimuth = float(ss_conf.get("camera_azimuth", 10))
        plotter.camera.elevation = float(ss_conf.get("camera_elevation", 15))
        if ss_conf.get("parallel_projection", True):
            plotter.enable_parallel_projection()
        plotter.screenshot(screenshot_path, transparent_background=ss_conf.get("transparent_background", True))
        print(f"[INFO] PNG saved to {screenshot_path}")
    else:
        plotter.show()

# Entry
if __name__ == "__main__":
    main()
