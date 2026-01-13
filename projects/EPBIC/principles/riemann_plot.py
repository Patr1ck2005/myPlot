import numpy as np
import pyvista as pv
from collections import deque

# =========================
# 0) 用户输入（你替换这里）
# coord_x, coord_y: (ny, nx) 实数网格
# eigenfreq1, eigenfreq2: (ny, nx) 复数
# =========================
# coord_x = ...
# coord_y = ...
# eigenfreq1 = ...
# eigenfreq2 = ...

# load projects/EPBIC/mannual/EP_BIC/EP_BIC_0Pslide-3eigens.pkl
import pickle
with open("D:\DELL\Documents\myPlots\projects\EPBIC\mannual\EP_BIC\EP_BIC_0Pslide-3eigens.pkl", "rb") as f:
    data = pickle.load(f)
coords = data["coords"]
coord_x = coords["spacer (nm)"]
coord_y = coords["h_die_grating (nm)"]*50
coord_x, coord_y = np.meshgrid(coord_x, coord_y, indexing="ij")
data_list = data["data_list"]
eigenfreq1 = data_list[0]["eigenfreq"]*1e5  # 第1个频率
eigenfreq2 = data_list[2]["eigenfreq"]*1e5  # 第2个频率

# 简单绘制散点图
import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
plt.scatter(eigenfreq1.real.ravel(), eigenfreq1.imag.ravel(), s=5, c='blue', alpha=0.5, label='Eigenfreq 1')
plt.scatter(eigenfreq2.real.ravel(), eigenfreq2.imag.ravel(), s=5, c='red', alpha=0.5, label='Eigenfreq 2')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Scatter Plot of Eigenfrequencies')
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 1) 参数（可调）
# =========================
hole_q = 0.005        # EP附近挖洞的分位数阈值：0.005~0.02 常用
edge_q = 0.98        # 删边阈值分位数：0.95~0.995 视噪声调整
use_8nbr = False      # True:8邻域（更稳），False:4邻域
z_mode = "abs"      # "real" / "imag" / "abs"
z_exag_factor = 1  # z方向夸张系数（越大越“立体”）

# =========================
# 2) 工具：选择绘制高度
# =========================
def z_of(freq, mode):
    if mode == "real":
        return freq.real
    if mode == "imag":
        return freq.imag
    if mode == "abs":
        return np.abs(freq)
    raise ValueError("z_mode must be real/imag/abs")

# =========================
# 3) 计算挖洞 mask (EP候选区)
# =========================
def compute_hole_mask(f1, f2, q):
    s = np.abs(f1 - f2)
    tau = np.quantile(s[np.isfinite(s)], q)
    hole = s <= tau
    return hole, s, tau

# =========================
# 4) 计算网格相邻边的连续代价，并得到可连接边 mask
# =========================
def compute_edge_keep_masks(f1, f2, hole, edge_q, use_8nbr=True):
    ny, nx = f1.shape

    # 右邻边 (i,j) -> (i, j+1)
    f1a, f1b = f1[:, :-1], f1[:, 1:]
    f2a, f2b = f2[:, :-1], f2[:, 1:]
    hole_a, hole_b = hole[:, :-1], hole[:, 1:]

    c_n = np.abs(f1a - f1b) + np.abs(f2a - f2b)
    c_s = np.abs(f1a - f2b) + np.abs(f2a - f1b)
    c_right = np.minimum(c_n, c_s)

    valid_right = (~hole_a) & (~hole_b) & np.isfinite(c_right)
    thr_right = np.quantile(c_right[valid_right], edge_q) if np.any(valid_right) else np.inf
    keep_right = valid_right & (c_right <= thr_right)   # shape (ny, nx-1)

    # 下邻边 (i,j) -> (i+1, j)
    f1a, f1b = f1[:-1, :], f1[1:, :]
    f2a, f2b = f2[:-1, :], f2[1:, :]
    hole_a, hole_b = hole[:-1, :], hole[1:, :]

    c_n = np.abs(f1a - f1b) + np.abs(f2a - f2b)
    c_s = np.abs(f1a - f2b) + np.abs(f2a - f1b)
    c_down = np.minimum(c_n, c_s)

    valid_down = (~hole_a) & (~hole_b) & np.isfinite(c_down)
    thr_down = np.quantile(c_down[valid_down], edge_q) if np.any(valid_down) else np.inf
    keep_down = valid_down & (c_down <= thr_down)       # shape (ny-1, nx)

    keep_diag1 = keep_diag2 = None
    thr_diag = None

    if use_8nbr:
        # 右下对角 (i,j)->(i+1,j+1)
        f1a, f1b = f1[:-1, :-1], f1[1:, 1:]
        f2a, f2b = f2[:-1, :-1], f2[1:, 1:]
        hole_a, hole_b = hole[:-1, :-1], hole[1:, 1:]
        c_n = np.abs(f1a - f1b) + np.abs(f2a - f2b)
        c_s = np.abs(f1a - f2b) + np.abs(f2a - f1b)
        c_d1 = np.minimum(c_n, c_s)
        valid_d1 = (~hole_a) & (~hole_b) & np.isfinite(c_d1)

        # 左下对角 (i,j)->(i+1,j-1)
        f1a, f1b = f1[:-1, 1:], f1[1:, :-1]
        f2a, f2b = f2[:-1, 1:], f2[1:, :-1]
        hole_a, hole_b = hole[:-1, 1:], hole[1:, :-1]
        c_n = np.abs(f1a - f1b) + np.abs(f2a - f2b)
        c_s = np.abs(f1a - f2b) + np.abs(f2a - f1b)
        c_d2 = np.minimum(c_n, c_s)
        valid_d2 = (~hole_a) & (~hole_b) & np.isfinite(c_d2)

        all_costs = np.concatenate([c_d1[valid_d1].ravel(), c_d2[valid_d2].ravel()])
        thr_diag = np.quantile(all_costs, edge_q) if all_costs.size else np.inf

        keep_diag1 = valid_d1 & (c_d1 <= thr_diag)  # shape (ny-1,nx-1)
        keep_diag2 = valid_d2 & (c_d2 <= thr_diag)  # shape (ny-1,nx-1)

    return keep_right, keep_down, keep_diag1, keep_diag2

# =========================
# 5) 连通分量（patch）标号：只允许沿 keep 边走
# =========================
def label_components(hole, keep_right, keep_down, keep_diag1=None, keep_diag2=None):
    ny, nx = hole.shape
    comp = -np.ones((ny, nx), dtype=np.int32)
    cid = 0

    def neighbors(i, j):
        # 右
        if j < nx-1 and keep_right[i, j]:
            yield (i, j+1)
        # 左
        if j > 0 and keep_right[i, j-1]:
            yield (i, j-1)
        # 下
        if i < ny-1 and keep_down[i, j]:
            yield (i+1, j)
        # 上
        if i > 0 and keep_down[i-1, j]:
            yield (i-1, j)

        if keep_diag1 is not None and keep_diag2 is not None:
            # 右下
            if i < ny-1 and j < nx-1 and keep_diag1[i, j]:
                yield (i+1, j+1)
            # 左上
            if i > 0 and j > 0 and keep_diag1[i-1, j-1]:
                yield (i-1, j-1)
            # 左下
            if i < ny-1 and j > 0 and keep_diag2[i, j-1]:
                yield (i+1, j-1)
            # 右上
            if i > 0 and j < nx-1 and keep_diag2[i-1, j]:
                yield (i-1, j+1)

    for i in range(ny):
        for j in range(nx):
            if hole[i, j] or comp[i, j] != -1:
                continue
            # BFS
            q = deque([(i, j)])
            comp[i, j] = cid
            while q:
                a, b = q.popleft()
                for u, v in neighbors(a, b):
                    if hole[u, v] or comp[u, v] != -1:
                        continue
                    comp[u, v] = cid
                    q.append((u, v))
            cid += 1

    return comp, cid

# =========================
# 6) 在每个 patch 内做 sheet 跟踪（是否交换），得到 sheetA/sheetB
# =========================
def assign_sheets_in_patch(f1, f2, comp, patch_id, keep_right, keep_down, keep_diag1=None, keep_diag2=None):
    ny, nx = f1.shape
    sheetA = np.full((ny, nx), np.nan + 1j*np.nan, dtype=np.complex128)
    sheetB = np.full((ny, nx), np.nan + 1j*np.nan, dtype=np.complex128)
    visited = np.zeros((ny, nx), dtype=bool)

    # 找 patch 内任一非空点作为种子
    seed = np.argwhere(comp == patch_id)
    if seed.size == 0:
        return None
    si, sj = seed[0]
    sheetA[si, sj] = f1[si, sj]
    sheetB[si, sj] = f2[si, sj]
    visited[si, sj] = True

    def can_step(i, j, u, v):
        # 判断 (i,j) 到 (u,v) 这条边是否允许（对应 keep mask）
        di, dj = u - i, v - j
        if di == 0 and dj == 1:
            return keep_right[i, j]
        if di == 0 and dj == -1:
            return keep_right[i, j-1]
        if di == 1 and dj == 0:
            return keep_down[i, j]
        if di == -1 and dj == 0:
            return keep_down[i-1, j]
        if keep_diag1 is not None and keep_diag2 is not None:
            if di == 1 and dj == 1:
                return keep_diag1[i, j]
            if di == -1 and dj == -1:
                return keep_diag1[i-1, j-1]
            if di == 1 and dj == -1:
                return keep_diag2[i, j-1]
            if di == -1 and dj == 1:
                return keep_diag2[i-1, j]
        return False

    q = deque([(si, sj)])
    while q:
        i, j = q.popleft()

        # 8 邻域候选
        for di, dj in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]:
            u, v = i+di, j+dj
            if u < 0 or u >= ny or v < 0 or v >= nx:
                continue
            if comp[u, v] != patch_id or visited[u, v]:
                continue
            if not can_step(i, j, u, v):
                continue

            # 选择是否交换，使得与当前点最连续
            c_n = abs(f1[u,v] - sheetA[i,j]) + abs(f2[u,v] - sheetB[i,j])
            c_s = abs(f2[u,v] - sheetA[i,j]) + abs(f1[u,v] - sheetB[i,j])
            if c_n <= c_s:
                sheetA[u,v], sheetB[u,v] = f1[u,v], f2[u,v]
            else:
                sheetA[u,v], sheetB[u,v] = f2[u,v], f1[u,v]

            visited[u, v] = True
            q.append((u, v))

    return sheetA, sheetB

# =========================
# 7) PyVista：把“patch”作为 cell_data 写入 StructuredGrid，
#    然后 threshold 每个 patch 单独渲染
# =========================
def render_patches_pyvista(coord_x, coord_y, sheetA, sheetB, comp, hole, z_mode="real", z_exag_factor=0.35):
    ny, nx = coord_x.shape

    ZA = z_of(sheetA, z_mode)
    ZB = z_of(sheetB, z_mode)

    # 主网格（两张 sheet 各建一个 StructuredGrid）
    gridA = pv.StructuredGrid(coord_x, coord_y, ZA)
    gridB = pv.StructuredGrid(coord_x, coord_y, ZB)

    # 把 comp/hole 写成 point_data，再转 cell_data 做 threshold
    comp_f = comp.astype(np.float32)
    hole_f = hole.astype(np.float32)

    for g in (gridA, gridB):
        g.point_data["comp"] = comp_f.ravel(order="F")
        g.point_data["hole"] = hole_f.ravel(order="F")

    cellA = gridA.point_data_to_cell_data(pass_point_data=True)
    cellB = gridB.point_data_to_cell_data(pass_point_data=True)

    # 去掉 hole 附近 cell（hole cell 平均值接近 1）
    cellA = cellA.threshold((0.0, 0.5), scalars="hole")  # keep hole<0.5
    cellB = cellB.threshold((0.0, 0.5), scalars="hole")

    p = pv.Plotter(window_size=(1150, 820))

    # 颜色轮（patch 多时循环）
    blues = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#1d4ed8", "#0ea5e9"]
    reds  = ["#dc2626", "#ef4444", "#f97316", "#fb7185", "#b91c1c", "#f43f5e"]

    # patch 数
    patch_ids = sorted([pid for pid in np.unique(comp) if pid >= 0])

    # 对每个 patch 单独抽出来画（两张sheet各画一次）
    for idx, pid in enumerate(patch_ids):
        pa = cellA.threshold((pid-0.1, pid+0.1), scalars="comp")
        pb = cellB.threshold((pid-0.1, pid+0.1), scalars="comp")
        if pa.n_cells:
            p.add_mesh(pa, color=blues[idx % len(blues)], opacity=0.92,
                       smooth_shading=True, specular=0.15)
        if pb.n_cells:
            p.add_mesh(pb, color=reds[idx % len(reds)], opacity=0.92,
                       smooth_shading=True, specular=0.15)

    # 坐标轴/边界（单独控制）
    p.show_bounds(
        grid="front", location="outer", all_edges=True,
        xtitle="coord_x", ytitle="coord_y", ztitle=f"{z_mode}(eigenfreq)",
        ticks="outside"
    )
    p.add_axes(line_width=2)

    # # 比例控制：z 方向夸张
    # xy_span = float(max(np.ptp(coord_x), np.ptp(coord_y)))
    # z_span = float(np.nanmax([np.nanmax(ZA), np.nanmax(ZB)]) - np.nanmin([np.nanmin(ZA), np.nanmin(ZB)]))
    # z_span = max(z_span, 1e-9)
    # z_exag = (z_exag_factor * xy_span) / z_span
    # p.set_scale(1.0, 1.0, z_exag)
    # p.set_scale(1e-1, 1.0, 1e2)
    p.set_scale(1, 1.0, 1)

    # p.camera_position = "iso"
    # p.camera.zoom(1.15)
    p.show()


# ============================================================
# 主流程：从数据自动分 patch + patch 内分 sheet + 渲染
# ============================================================
def plot_riemann_from_grid(coord_x, coord_y, eigenfreq1, eigenfreq2):
    hole, s, tau = compute_hole_mask(eigenfreq1, eigenfreq2, hole_q)

    keep_right, keep_down, keep_d1, keep_d2 = compute_edge_keep_masks(
        eigenfreq1, eigenfreq2, hole, edge_q, use_8nbr=use_8nbr
    )

    comp, ncomp = label_components(hole, keep_right, keep_down, keep_d1, keep_d2)

    # 初始化总的 sheetA/B（全域）
    ny, nx = eigenfreq1.shape
    sheetA = np.full((ny, nx), np.nan + 1j*np.nan, dtype=np.complex128)
    sheetB = np.full((ny, nx), np.nan + 1j*np.nan, dtype=np.complex128)

    # 每个 patch 内跟踪分配
    for pid in range(ncomp):
        out = assign_sheets_in_patch(eigenfreq1, eigenfreq2, comp, pid, keep_right, keep_down, keep_d1, keep_d2)
        if out is None:
            continue
        A, B = out
        mask = (comp == pid)
        sheetA[mask] = A[mask]
        sheetB[mask] = B[mask]

    render_patches_pyvista(coord_x, coord_y, sheetA, sheetB, comp, hole, z_mode=z_mode, z_exag_factor=z_exag_factor)

# ---- 运行 ----
plot_riemann_from_grid(coord_x, coord_y, eigenfreq1, eigenfreq2)
