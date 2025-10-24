#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular & highly-configurable band-surface visualizer (PyVista/VTK, no argparse).

Surfaces in k-space (kx, ky):
  S1 (reference plane, imaginary):            E_ref
  S2 (real dispersion):                       E2(k)=k^2*(A_iso + A_4*cos(4θ))
  S3 (auxiliary paraboloid, imaginary):       inv_mass*k^2 + E_offset

Key ideas:
- All behavior is driven by CONFIG (per-surface visibility, color mapping, scalar bars, opacity, etc.)
- Colorizers are pluggable; per-surface choose 'height' / 'radial' / 'none' / custom function
- Intersections are modular: choose pairs, robust zero-crossing with tolerance & graceful fallback
- Optional exports: screenshot, intersection polylines to VTK/CSV
- Enhanced: per-intersection colors, detailed surface controls (lighting, ambient), clean view, camera for screenshot

Requires:   pip install pyvista pyvistaqt
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
E_ref      = -2     # S1 reference plane (imaginary helper)
A_iso      = -0.80  # S2 isotropic coefficient
A_4        = -0.40  # S2 fourfold anisotropy strength
inv_mass   = 0.50   # S3 ~ ħ^2/(2m*)
E_offset   = -3     # S3 energy offset

# k-space sampling
KMAX   = 2.0          # range [-KMAX, KMAX]
NGRID  = 240          # grid density

# =========================
# Colorizers (return [0,1] scalar field)
# =========================
def colorize_height(kx, ky, z):
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if np.isclose(zmax, zmin): return np.zeros_like(z, dtype=float)
    return (z - zmin) / (zmax - zmin)

def colorize_radial(kx, ky, z):
    r = np.sqrt(kx**2 + ky**2)
    rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
    if np.isclose(rmax, rmin): return np.zeros_like(r, dtype=float)
    return (r - rmin) / (rmax - rmin)

# 你可在此注册更多色标器；也可直接在 CONFIG 里传自定义函数
COLORIZER_REGISTRY: Dict[str, Callable] = {
    "height": colorize_height,
    "radial": colorize_radial,
}

# =========================
# Surface definitions
# =========================
def S1_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    return np.full_like(kx, E_ref, dtype=float)

def S2_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    r2 = kx**2 + ky**2
    th = np.arctan2(ky, kx)
    return r2 * (A_iso + A_4 * np.cos(4.0 * th))

def S3_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    r2 = kx**2 + ky**2
    return inv_mass * r2 + E_offset

SURFACE_FUNCS = {
    "S1": S1_func,
    "S2": S2_func,
    "S3": S3_func,
}

# =========================
# MASTER CONFIG
# =========================
CONFIG = {
    "scene": {
        "window_size": (1200, 900),
        "background": "white",
        "theme": "document",
        "depth_peeling": True,        # better translucency in VTK backends
        "camera": {"preset": "iso", "elev": 25, "azim": 35},
        "show_bounds": True,
        "axes": True,
        "clean_view": False,          # 新增: 如果True，隐藏轴和边界，使曲面更纯净
        "pure_surface_only": False,   # 新增: 如果True，只渲染曲面和交线，不加球或其他
    },

    # Per-surface settings
    # mode: "colormap" | "solid" | "wireframe"
    # colorizer: name in registry | callable | None
    "surfaces": {
        "S1": {
            "visible": True,
            "mode": "solid",
            "solid_color": (0.55, 0.55, 0.55),
            "opacity": 0.30,
            "smooth_shading": True,
            "cmap": "viridis",
            "colorizer": None,        # no scalar mapping on S1
            "scalar_bar": False,
            "line_width": 1.0,        # used if wireframe
            "lighting": False,        # 新增: 默认False
            "ambient": 0.2,           # 新增: 默认0.2
            "specular": 0.0,          # 新增: 可选，默认0.0
        },
        "S2": {
            "visible": True,
            "mode": "colormap",       # only S2 is colored by default
            "solid_color": (0.2, 0.2, 0.2),
            "opacity": 0.95,
            "smooth_shading": True,
            "cmap": "Blues",
            "colorizer": "height",    # 'height'/'radial'/callable/None
            "scalar_bar": False,
            "line_width": 1.0,
            "lighting": False,        # 新增
            "ambient": 0.2,           # 新增
            "specular": 0.0,          # 新增
        },
        "S3": {
            "visible": True,
            "mode": "solid",
            "solid_color": (0.3, 0.5, 0.85),
            "opacity": 0.40,
            "smooth_shading": True,
            "cmap": "viridis",
            "colorizer": None,
            "scalar_bar": False,
            "line_width": 1.0,
            "lighting": False,        # 新增
            "ambient": 0.2,           # 新增
            "specular": 0.0,          # 新增
        },
    },

    # Intersection settings
    # pairs: 现在是列表的字典，每个支持独立color和line_width
    "intersections": {
        "draw": True,
        "pairs": [
            {"surface1": "S2", "surface2": "S1", "color": "white", "line_width": 3.0},
            {"surface1": "S2", "surface2": "S3", "color": "red", "line_width": 4.0},  # 示例不同颜色
        ],
        "tol_rel": 1e-3,          # relative tolerance for near-zero fallback
        "export": {               # optional exports
            "folder": None,       # e.g., "exports" or None
            "vtk": False,         # save polyline as .vtp
            "csv": False,         # save as .csv (x,y,z)
        }
    },

    # Spheres (annotation objects)
    "spheres": [
        {"center": (0.5, 0.5, None), "radius": 0.06, "color": (1.0, 0.4, 0.4), "opacity": 0.9},
        {"center": (1.0, 0.0, None), "radius": 0.06, "color": (0.4, 1.0, 0.4), "opacity": 0.9},
        {"center": (-0.8, 0.6, None), "radius": 0.06, "color": (0.4, 0.6, 1.0), "opacity": 0.9},
    ],

    # Output options
    "output": {
        "screenshot": {           # 新增: screenshot配置
            "path": None,         # e.g., './rsl/3d_surface.png' or None
            "transparent_background": True,
            "camera_azimuth": 10,
            "camera_elevation": -15,
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
    """Return None (no mapping) or flat array in Fortran order."""
    if kind is None:
        return None
    fn = kind
    if isinstance(kind, str):
        fn = COLORIZER_REGISTRY.get(kind)
        if fn is None:
            raise ValueError(f"Unknown colorizer: {kind}")
    scal = fn(kx, ky, z)
    return np.ascontiguousarray(scal.ravel(order="F"))

def _zero_contour(mesh: pv.StructuredGrid, F: np.ndarray, name: str, tol_rel: float):
    """Robust zero contour with tolerance fallback; return (poly, used_iso or None)."""
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
    trials = [eps, -eps, 0.5*eps, -0.5*eps]
    absmin_val = float(F.ravel()[np.nanargmin(np.abs(F))])
    trials.append(absmin_val)
    for val in trials:
        poly = m.contour([val], scalars=name)
        if poly and poly.n_points > 0: return poly, val
    return None, None

def extract_intersections(meshes: Dict[str, pv.StructuredGrid],
                          fields: Dict[str, np.ndarray],
                          pairs: List[Dict[str, any]],
                          tol_rel: float
                          ) -> List[Tuple[str, pv.PolyData, Optional[float], Dict]]:
    """Return list of (label, poly, used_iso, pair_conf)."""
    out = []
    for pconf in pairs:
        a, b = pconf["surface1"], pconf["surface2"]
        Za = fields[a]
        Zb = fields[b]
        base = a  # extract on surface a
        poly, used = _zero_contour(meshes[base], Za - Zb, f"f_{a}_{b}", tol_rel)
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
    # global toggles
    out = CONFIG["output"]
    if out.get("allow_empty_mesh", False):
        pv.global_theme.allow_empty_mesh = True

    sc = CONFIG["scene"]
    pv.set_plot_theme(sc.get("theme", "document"))
    plotter = pv.Plotter(window_size=sc.get("window_size", (1200, 900)))
    plotter.set_background(sc.get("background", "white"))
    if sc.get("depth_peeling", False):
        plotter.enable_depth_peeling()

    # grid
    k = np.linspace(-KMAX, KMAX, NGRID)
    KX, KY = np.meshgrid(k, k, indexing="xy")

    # fields and meshes
    Z_fields: Dict[str, np.ndarray] = {name: func(KX, KY)
                                       for name, func in SURFACE_FUNCS.items()}
    meshes: Dict[str, pv.StructuredGrid] = {name: make_grid(KX, KY, Z_fields[name])
                                            for name in SURFACE_FUNCS.keys()}

    # add surfaces
    for name, sconf in CONFIG["surfaces"].items():
        if not sconf.get("visible", True):
            continue
        grid = meshes[name]
        mode = sconf.get("mode", "colormap")
        opacity = float(sconf.get("opacity", 1.0))
        smooth = bool(sconf.get("smooth_shading", True))
        line_width = float(sconf.get("line_width", 1.0))
        lighting = bool(sconf.get("lighting", False))  # 新增
        ambient = float(sconf.get("ambient", 0.2))     # 新增
        specular = float(sconf.get("specular", 0.0))   # 新增

        if mode == "colormap":
            scal = compute_color_scalars(sconf.get("colorizer", None), KX, KY, Z_fields[name])
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
                lighting=lighting,      # 新增
                ambient=ambient,        # 新增
                specular=specular,      # 新增
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
                                 color=pconf.get("color", "black"),  # 新增: per-pair color
                                 line_width=float(pconf.get("line_width", 3.0)),  # 新增: per-pair
                                 name=label)
                print(f"[INFO] drew {label} (iso ~ {0.0 if used is None else used:.3e})")
                # optional export
                exp = iconf.get("export", {})
                folder = exp.get("folder")
                if folder:
                    export_poly(poly, label.replace("∩", "_"), folder,
                                want_vtk=bool(exp.get("vtk", False)),
                                want_csv=bool(exp.get("csv", False)))
            else:
                print(f"[WARN] {label}: empty (no zero-crossing within tolerance). Skipped.")

    # spheres (skip if pure_surface_only)
    if not sc.get("pure_surface_only", False):
        for i, sph in enumerate(CONFIG.get("spheres", []), start=1):
            cx, cy, cz = sph.get("center", (0.0, 0.0, None))
            if cz is None:
                # place on S2 by default
                zval = float(S2_func(np.array([[cx]]), np.array([[cy]]))[0, 0])
            else:
                zval = float(cz)
            sphere = pv.Sphere(radius=float(sph.get("radius", 0.06)),
                               center=(cx, cy, zval),
                               theta_resolution=48, phi_resolution=48)
            plotter.add_mesh(sphere,
                             color=sph.get("color", None),
                             opacity=float(sph.get("opacity", 0.9)),
                             smooth_shading=True,
                             name=f"sphere{i}")

    # axes, bounds, camera (skip if clean_view or pure_surface_only)
    clean_view = sc.get("clean_view", False) or sc.get("pure_surface_only", False)
    if not clean_view:
        if sc.get("axes", True):
            plotter.add_axes(line_width=2, labels_off=False)
        if sc.get("show_bounds", True):
            plotter.show_bounds(grid="front", location="outer", all_edges=True,
                                xtitle="k_x", ytitle="k_y", ztitle="E")
    cam = sc.get("camera", {})
    if cam.get("preset", "iso") == "iso":
        plotter.camera_position = "iso"
    plotter.camera.elevation = float(cam.get("elev", 25))
    plotter.camera.azimuth   = float(cam.get("azim", 35))

    # screenshot settings (新增)
    ss_conf = out.get("screenshot", {})
    screenshot_path = ss_conf.get("path", None)
    if screenshot_path:
        # 设置自定义相机
        plotter.camera.azimuth = float(ss_conf.get("camera_azimuth", 10))
        plotter.camera.elevation = float(ss_conf.get("camera_elevation", -15))
        if ss_conf.get("parallel_projection", True):
            plotter.enable_parallel_projection()
        # 保存
        plotter.screenshot(screenshot_path, transparent_background=ss_conf.get("transparent_background", True))
        print(f"[INFO] Screenshot saved to {screenshot_path}")
    else:
        plotter.show()

# =========================
# Entry
# =========================
if __name__ == "__main__":
    main()
