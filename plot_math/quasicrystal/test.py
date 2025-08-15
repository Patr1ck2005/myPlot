import math
import itertools
import numpy as np

# ===================== 基础：投影矩阵 =====================

def projection_matrices():
    """返回 (P_par, P_perp) 两个 2x4 投影矩阵。"""
    a = 1.0 / math.sqrt(2.0)
    # 物理投影：列向量为 v0..v3
    P_par = np.array([
        [ 1.0,  a, 0.0, -a],
        [ 0.0,  a, 1.0,  a],
    ], dtype=float)
    # 内空间投影：代数共轭（把 +a 换成 -a）
    P_perp = np.array([
        [ 1.0, -a, 0.0,  a],
        [ 0.0, -a, 1.0, -a],
    ], dtype=float)
    return P_par, P_perp

# ===================== 几何工具：凸包与点在多边形内 =====================

def convex_hull(points):
    """
    Andrew 单调链凸包（二维）。输入 Nx2，返回按逆时针排序的凸包顶点（M x 2）。
    """
    pts = sorted(points.tolist())
    if len(pts) <= 1:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

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
    """
    半平面法：判断点 pt 是否在凸多边形 poly（逆时针）内或边上。
    """
    x, y = pt
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        if (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1) < -eps:
            return False
    return True

# ===================== 构造典范八边形窗 =====================

def canonical_octagon_window(P_perp):
    """
    取 4D 单位超立方体 [-1/2, 1/2]^4 的 16 个顶点投影到内空间，做凸包 -> 正八边形窗。
    """
    corners4d = np.array(list(itertools.product([-0.5, 0.5], repeat=4)), dtype=float)  # (16,4)
    projected = corners4d @ P_perp.T  # (16,2)
    hull = convex_hull(projected)
    # hull 应为 8 个点的正八边形；出于稳健性，允许 >8 的情况（数值重复），再做一次去重
    # 按角度排序
    c = hull.mean(axis=0)
    ang = np.arctan2(hull[:,1] - c[1], hull[:,0] - c[0])
    order = np.argsort(ang)
    poly = hull[order]
    # 去近似重复点
    cleaned = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-10:
            cleaned.append(p)
    if np.linalg.norm(cleaned[0] - cleaned[-1]) < 1e-10 and len(cleaned) > 1:
        cleaned.pop()
    return np.array(cleaned, dtype=float)  # 期望长度为 8

# ===================== 生成格点 =====================

def ammann_beenker_points(R=12.0, max_coeff=None, return_internal=False):
    """
    生成半径 R 内（物理平面欧氏距离）的 Ammann–Beenker 顶点集。
    参数:
        R: 物理空间选择半径（以边长=1 为单位）。
        max_coeff: 可选，限制 4D 系数 n_i 的绝对值上界（默认 ceil(R)）。
        return_internal: 若 True，同步返回对应的内空间点，以便调试/可视化窗。
    返回:
        pts: (M,2) 的 numpy 数组，AB 顶点坐标。
        (可选) pts_perp: (M,2) 内空间坐标。
    """
    P_par, P_perp = projection_matrices()
    window_poly = canonical_octagon_window(P_perp)

    if max_coeff is None:
        max_coeff = int(math.ceil(R))  # 经验上足够覆盖半径 R；要更保险可取 ceil(R)+1

    cols_par = [P_par[:, i] for i in range(4)]
    cols_perp = [P_perp[:, i] for i in range(4)]

    pts = []
    pts_perp = []

    # 穷举范围 |n_i| <= max_coeff；对 R<=12 时规模可控
    rng = range(-max_coeff, max_coeff + 1)
    for n0 in rng:
        for n1 in rng:
            for n2 in rng:
                for n3 in rng:
                    n = np.array([n0, n1, n2, n3], dtype=float)

                    # 先验：先检内空间窗（快很多）
                    p_perp = P_perp @ n
                    if not point_in_convex_polygon(p_perp, window_poly, eps=1e-12):
                        continue

                    # 再检物理半径
                    p = P_par @ n
                    if p[0]*p[0] + p[1]*p[1] <= R*R + 1e-12:
                        pts.append(p)
                        if return_internal:
                            pts_perp.append(p_perp)

    if len(pts) == 0:
        return (np.empty((0,2)), np.empty((0,2))) if return_internal else np.empty((0,2))

    pts = np.vstack(pts)

    # 去重（数值容差）
    rounded = np.round(pts, 12)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    pts = pts[idx]

    if return_internal:
        pts_perp = np.vstack(pts_perp)[idx]
        return pts, pts_perp
    else:
        return pts

# ===================== 示例与绘图（可选） =====================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    R = 12.0
    pts = ammann_beenker_points(R=R)

    print(f"生成点数: {len(pts)}")
    plt.figure(figsize=(6,6))
    plt.scatter(pts[:,0], pts[:,1], s=6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Ammann–Beenker 8重准晶顶点 (R={R})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()
