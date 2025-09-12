import math
import itertools
import numpy as np
import ezdxf
from joblib import Parallel, delayed

# 缓存窗口多边形
_window_cache = None


def projection_matrices():
    a = 1 / np.sqrt(2)
    P_par = np.array([[1.0, a, 0.0, -a], [0.0, a, 1.0, a]], dtype=float)
    P_perp = np.array([[1.0, -a, 0.0, a], [0.0, -a, 1.0, -a]], dtype=float)
    return P_par, P_perp


def convex_hull(points):
    pts = sorted(points.tolist())
    if len(pts) <= 1:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


def canonical_octagon_window(P_perp):
    global _window_cache
    if _window_cache is not None:
        return _window_cache
    corners4d = np.array(list(itertools.product([-0.5, 0.5], repeat=4)), dtype=float)
    projected = corners4d @ P_perp.T
    hull = convex_hull(projected)
    c = hull.mean(axis=0)
    ang = np.arctan2(hull[:, 1] - c[1], hull[:, 0] - c[0])
    order = np.argsort(ang)
    poly = hull[order]
    cleaned = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-10:
            cleaned.append(p)
    if np.linalg.norm(cleaned[0] - cleaned[-1]) < 1e-10 and len(cleaned) > 1:
        cleaned.pop()
    _window_cache = np.array(cleaned, dtype=float)
    return _window_cache


def process_chunk(n_chunk, P_par, P_perp, window_poly, R):
    pts_perp = n_chunk @ P_perp.T
    pts_par = n_chunk @ P_par.T
    in_circle = np.sum(pts_par ** 2, axis=1) <= R ** 2 + 1e-12
    in_window = np.ones(len(n_chunk), dtype=bool)
    for i in range(len(window_poly)):
        x1, y1 = window_poly[i]
        x2, y2 = window_poly[(i + 1) % len(window_poly)]
        cross = (x2 - x1) * (pts_perp[:, 1] - y1) - (y2 - y1) * (pts_perp[:, 0] - x1)
        in_window &= cross >= -1e-12
    return pts_par[in_circle & in_window]


def ammann_beenker_points(R=12.0, max_coeff=None, return_internal=False, n_jobs=4):
    P_par, P_perp = projection_matrices()
    window_poly = canonical_octagon_window(P_perp)

    if max_coeff is None:
        norm_perp = np.linalg.norm(P_perp, ord='fro')
        window_radius = 0.99
        max_coeff = int(math.ceil(R / norm_perp + window_radius))

    rng = np.arange(-max_coeff, max_coeff + 1)
    N = np.array(list(np.ndindex((2 * max_coeff + 1,) * 4))) - max_coeff
    chunks = np.array_split(N, n_jobs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk, P_par, P_perp, window_poly, R) for chunk in chunks
    )
    pts = np.vstack([r for r in results if len(r) > 0])

    rounded = np.round(pts, 12)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    pts = pts[idx]

    if return_internal:
        pts_perp = N @ P_perp.T
        valid = np.isin(np.arange(len(N)), idx)
        return pts, pts_perp[valid]
    return pts


def export_to_dxf(pts, filename='quasicrystal.dxf', shape='circle', radius=0.1):
    doc = ezdxf.new()
    msp = doc.modelspace()
    for pt in pts:
        if shape == 'circle':
            msp.add_circle(center=pt, radius=radius)
        elif shape == 'square':
            half_size = radius / np.sqrt(2)
            msp.add_lwpolyline([
                (pt[0] - half_size, pt[1] - half_size),
                (pt[0] + half_size, pt[1] - half_size),
                (pt[0] + half_size, pt[1] + half_size),
                (pt[0] - half_size, pt[1] + half_size),
                (pt[0] - half_size, pt[1] - half_size),
            ], close=True)
    doc.saveas(filename)
    print(f"DXF 文件已保存为 {filename}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    R = 45.0
    start_time = time.time()
    pts = ammann_beenker_points(R=R, n_jobs=4)
    end_time = time.time()
    print(f"生成点数: {len(pts)}，耗时: {end_time - start_time:.2f}秒")

    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=6, c='red')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Ammann–Beenker octagonal quasicrystal (R={R})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig('quasicrystal_plot.svg', transparent=True)
    plt.show()

    export_to_dxf(pts, filename='quasicrystal.dxf', shape='circle', radius=0.2)

    # 生成 Lumerical 脚本，简化为一串 addcircle() 命令
    lumerical_script_simplified = ''

    # 将每个点转换为 addcircle() 命令
    for i, pt in enumerate(pts):
        lumerical_script_simplified += (
            f'addcircle;'
            f'set("name", "AB_cyl_{i + 1}");'
            f'set("x", {pt[0] * 1e-6});'
            f'set("y", {pt[1] * 1e-6});'
            f'set("z", t/2);'
            f'set("radius", r);'
            f'set("z span", t);'
            f'set("index", index_disk);'
            f'addtogroup("AB_quasicrystal_group");'
        )
    # 保存生成的 Lumerical 脚本
    lumerical_script_file_simplified = './ammann_beenker_simplified_lumerical_script.txt'
    with open(lumerical_script_file_simplified, 'w') as f:
        f.write(lumerical_script_simplified)
    print(lumerical_script_simplified)
