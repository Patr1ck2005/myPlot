import math
import itertools
import numpy as np
import ezdxf


# ===================== 基础：投影矩阵 =====================

def projection_matrices():
    """返回 (P_par, P_perp) 两个 2x4 投影矩阵。"""
    a = 1 / np.sqrt(2)
    # a = 2 / 3
    P_par = np.array([
        [1.0, a, 0.0, -a],
        [0.0, a, 1.0, a],
    ], dtype=float)
    P_perp = np.array([
        [1.0, -a, 0.0, a],
        [0.0, -a, 1.0, -a],
    ], dtype=float)
    return P_par, P_perp


# ===================== 几何工具：凸包与点在多边形内 =====================

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


def point_in_convex_polygon(pt, poly, eps=1e-12):
    x, y = pt
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) < -eps:
            return False
    return True


# ===================== 构造典范八边形窗 =====================

def canonical_octagon_window(P_perp):
    corners4d = np.array(list(itertools.product([-0.5, 0.5], repeat=4)), dtype=float)  # (16,4)
    projected = corners4d @ P_perp.T  # (16,2)
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
    return np.array(cleaned, dtype=float)  # 期望长度为 8


# ===================== 生成格点 =====================

def ammann_beenker_points(R=12.0, max_coeff=None, return_internal=False):
    P_par, P_perp = projection_matrices()
    window_poly = canonical_octagon_window(P_perp)

    if max_coeff is None:
        max_coeff = int(math.ceil(R))

    pts = []
    pts_perp = []

    rng = range(-max_coeff, max_coeff + 1)
    for n0 in rng:
        for n1 in rng:
            for n2 in rng:
                for n3 in rng:
                    n = np.array([n0, n1, n2, n3], dtype=float)
                    p_perp = P_perp @ n
                    if not point_in_convex_polygon(p_perp, window_poly, eps=1e-12):
                        continue

                    p = P_par @ n
                    if p[0] * p[0] + p[1] * p[1] <= R * R + 1e-12:
                        pts.append(p)
                        if return_internal:
                            pts_perp.append(p_perp)

    if len(pts) == 0:
        return (np.empty((0, 2)), np.empty((0, 2))) if return_internal else np.empty((0, 2))

    pts = np.vstack(pts)

    # 去重
    rounded = np.round(pts, 12)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    pts = pts[idx]

    if return_internal:
        pts_perp = np.vstack(pts_perp)[idx]
        return pts, pts_perp
    else:
        return pts


# ===================== 导出为 DXF 文件 =====================

def export_to_dxf(pts, filename='quasicrystal.dxf', shape='circle', radius=0.1):
    """将点导出为 DXF 文件，支持圆形或方形图案。"""
    doc = ezdxf.new()
    msp = doc.modelspace()

    for pt in pts:
        if shape == 'circle':
            msp.add_circle(center=pt, radius=radius)  # 圆形
        elif shape == 'square':
            half_size = radius / np.sqrt(2)
            msp.add_lwpolyline([
                (pt[0] - half_size, pt[1] - half_size),
                (pt[0] + half_size, pt[1] - half_size),
                (pt[0] + half_size, pt[1] + half_size),
                (pt[0] - half_size, pt[1] + half_size),
                (pt[0] - half_size, pt[1] - half_size),
            ], close=True)  # 方形

    doc.saveas(filename)
    print(f"DXF 文件已保存为 {filename}")


# ===================== 示例与绘图（可选） =====================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    R = 30.0
    pts = ammann_beenker_points(R=R)

    # 可选择绘制图形
    print(f"生成点数: {len(pts)}")
    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=6, c='red')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Ammann–Beenker octagonal quasicrystal (R={R})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig('quasicrystal_plot.svg', transparent=True)
    plt.show()

    # 导出为DXF文件，选择图案：'circle' 或 'square'，并指定半径
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
    lumerical_script_file_simplified = './ammann_beenker_simplified_lumerical_script.lsf'
    with open(lumerical_script_file_simplified, 'w') as f:
        f.write(lumerical_script_simplified)
    print(lumerical_script_simplified)
