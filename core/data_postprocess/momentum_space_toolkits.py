# momentum_space_toolkits.py
# -*- coding: utf-8 -*-
import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 基础与对称小工具（复用）
# =========================
def _wrap_angle_pi(phi):
    """把 φ 规约到 (0, π]（线偏振角模 π）。"""
    return (phi) % np.pi

def _axis_coords(arr):
    """坐标轴镜像：不重复 0 且保持升序。"""
    arr = np.asarray(arr)
    left = -arr[1:][::-1] if np.isclose(arr[0], 0) else -arr[::-1]
    return np.concatenate([left, arr], 0)

def _angle_mirror(phi):
    """轴镜像：φ' = -φ（mod π），用 wrap 收回到 (-π/2, π/2]。"""
    return _wrap_angle_pi(-_wrap_angle_pi(phi))


def _mirror_xy_data(Z):
    """对二维字段做 x、y 轴几何镜像（不含物理变换），得到全平面。"""
    Zx = np.concatenate([np.flip(Z[1: ], 0), Z], 0)
    Zy = np.concatenate([np.flip(Zx[:, 1:], 1), Zx], 1)
    return Zy

# =========================
# 仅几何补全（标量/复数场）
# =========================
def geom_complete(coords, Z, mode='xy'):
    """
    仅几何镜像/旋转，不做物理变换。
    mode: 'xy' | 'x' | 'y' | 'C4'（先 xy，再与 R90 平均）
    返回 (coords_out, Z_full)
    """
    keys = list(coords.keys())
    xk, yk = ('kx','ky') if {'kx','ky'}.issubset(coords) else (keys[0], keys[1])
    x, y, Z = np.asarray(coords[xk]), np.asarray(coords[yk]), np.asarray(Z)

    if mode == 'xy':
        x_full, y_full, Z_full = _axis_coords(x), _axis_coords(y), _mirror_xy_data(Z)
    elif mode == 'x':
        x_full, y_full = _axis_coords(x), y
        Z_full = np.concatenate([np.flip(Z[1:], 0), Z], 0)
    elif mode == 'y':
        x_full, y_full = x, _axis_coords(y)
        Z_full = np.concatenate([np.flip(Z[:,1:], 1), Z], 1)
    elif mode == 'C4':
        x_full, y_full, Z_full = _axis_coords(x), _axis_coords(y), _mirror_xy_data(Z)
        # if Z_full.shape[0] == Z_full.shape[1]:
        #     Z_full = 0.5*(Z_full + np.rot90(Z_full, 1))
    else:
        raise ValueError(f"未知 mode: {mode}")

    out = dict(coords)
    out[xk], out[yk] = x_full, y_full
    return out, Z_full

# =========================
# 偏振补全（φ, χ）
# =========================

def _mirror_phi_x(phi):
    """ky→-ky 镜像：φ 用 2φ→π-2φ；χ 取反；列翻转对齐。"""
    return _angle_mirror(np.flip(phi[:,1:], 1))

def _mirror_phi_y(phi):
    return _angle_mirror(np.flip(phi[1:], 0))

def _mirror_even_chi_x(chi):
    return np.flip(chi[:,1:], 1)

def _mirror_odd_chi_x(chi):
    return -np.flip(chi[:,1:], 1)

def _mirror_even_chi_y(chi):
    return np.flip(chi[1:], 0)

def _mirror_odd_chi_y(chi):
    return -np.flip(chi[1:], 0)

def _mirror_phi_chi_x(phi, chi):
    """ky→-ky 镜像：φ 用 2φ→π-2φ；χ 取反；列翻转对齐。"""
    return _mirror_phi_x(phi), -np.flip(chi[:,1:], 1)

def _mirror_phi_chi_y(phiR, chiR):
    """kx→-kx 镜像：φ 用 2φ→π-2φ；χ 取反；行翻转对齐。"""
    return _mirror_phi_y(phiR), -np.flip(chiR[1:], 0)

def complete_C4_polarization(coords, phi_Q1, chi_Q1):
    """
    由第一象限 (phi, chi) 补全至全平面：
    """
    keys = list(coords.keys())
    xk, yk = ('kx','ky') if {'kx','ky'}.issubset(coords) else (keys[0], keys[1])
    kx, ky = coords[xk], coords[yk]
    phi, chi = _wrap_angle_pi(np.asarray(phi_Q1)), np.asarray(chi_Q1)

    assert kx.ndim==ky.ndim==1 and np.all(np.diff(kx)>0) and np.all(np.diff(ky)>0)
    assert np.isclose(kx[0],0) and np.isclose(ky[0],0)
    assert phi.shape[:2]==(kx.size,ky.size) and chi.shape[:2]==(kx.size,ky.size)

    kx_full, ky_full = _axis_coords(kx), _axis_coords(ky)

    # 右半平面
    phi_RB, chi_RB = _mirror_phi_chi_x(phi, chi)
    phi_R = np.concatenate([phi_RB, phi], 1)
    chi_R = np.concatenate([chi_RB, chi], 1)
    # 左半平面
    phi_L, chi_L = _mirror_phi_chi_y(phi_R, chi_R)
    phi_full = np.concatenate([phi_L, phi_R], 0)
    chi_full = np.concatenate([chi_L, chi_R], 0)

    full_coords = {'m1': kx_full, 'm2': ky_full}
    return full_coords, _wrap_angle_pi(phi_full), chi_full

def complete_C6_polarization(coords, phi_Q1, chi_Q1):
    """
    由第一象限 (phi, chi) 补全至全平面：
    """
    keys = list(coords.keys())
    xk, yk = ('kx','ky') if {'kx','ky'}.issubset(coords) else (keys[0], keys[1])
    kx, ky = coords[xk], coords[yk]
    phi, chi = _wrap_angle_pi(np.asarray(phi_Q1)), np.asarray(chi_Q1)

    assert kx.ndim==ky.ndim==1 and np.all(np.diff(kx)>0) and np.all(np.diff(ky)>0)
    assert np.isclose(kx[0],0) and np.isclose(ky[0],0)
    assert phi.shape[:2]==(kx.size,ky.size) and chi.shape[:2]==(kx.size,ky.size)

    kx_full, ky_full = _axis_coords(kx), _axis_coords(ky)

    # 右半平面
    phi_RB = _mirror_phi_x(phi)
    chi_RB = _mirror_even_chi_x(chi)
    phi_R = np.concatenate([phi_RB, phi], 1)
    chi_R = np.concatenate([chi_RB, chi], 1)
    # 左半平面
    phi_L = _mirror_phi_y(phi_R)
    chi_L = _mirror_odd_chi_y(chi_R)
    phi_full = np.concatenate([phi_L, phi_R], 0)
    chi_full = np.concatenate([chi_L, chi_R], 0)

    full_coords = {'m1': kx_full, 'm2': ky_full}
    return full_coords, _wrap_angle_pi(phi_full), chi_full

def complete_C2_polarization(coords, phi_Q1, chi_Q1):
    """
    由第一/四象限 (phi, chi) 补全至全平面（对称轴 y 轴）：
    """
    keys = list(coords.keys())
    xk, yk = ('kx', 'ky') if {'kx', 'ky'}.issubset(coords) else (keys[0], keys[1])
    kx, ky = coords[xk], coords[yk]
    phi, chi = _wrap_angle_pi(np.asarray(phi_Q1)), np.asarray(chi_Q1)

    assert kx.ndim == ky.ndim == 1 and np.all(np.diff(kx) > 0) and np.all(np.diff(ky) > 0)
    assert np.isclose(kx[0], 0) or np.isclose(ky[0], 0)
    assert phi.shape[:2] == (kx.size, ky.size) and chi.shape[:2] == (kx.size, ky.size)

    kx_full, ky_full = _axis_coords(kx), ky

    # 对称 y 轴
    phi_L, chi_L = _mirror_phi_chi_y(phi, chi)
    phi_full = np.concatenate([phi_L, phi], 0)
    chi_full = np.concatenate([chi_L, chi], 0)

    full_coords = {'m1': kx_full, 'm2': ky_full}
    return full_coords, _wrap_angle_pi(phi_full), chi_full

import numpy as np

import numpy as np

def complete_C4_spectrum(coords, complex_amplitude, eps=1e-9):
    """
    简单规则：Q1 旋转 90° 后相位整体 +π（乘 -1）；
             180° => (+1)，270° => (-1)。
    返回: {xk: ax, yk: ay}, AF (shape=(len(ay), len(ax)))
    """
    keys = list(coords.keys())
    xk, yk = ('kx','ky') if {'kx','ky'}.issubset(coords) else (keys[0], keys[1])
    kx, ky = np.asarray(coords[xk]), np.asarray(coords[yk])
    A = np.asarray(complex_amplitude)

    # 统一为 2D 网格
    if kx.ndim==1 and ky.ndim==1:
        KX, KY = np.meshgrid(kx, ky, indexing='xy')
        if A.shape != KX.shape: A = A.reshape(KX.shape)
    elif kx.ndim==2 and ky.ndim==2:
        KX, KY = kx, ky
        if A.shape != KX.shape: A = A.reshape(KX.shape)
    else:
        raise ValueError("kx, ky 必须同为 1D 或同为 2D。")

    # Q1 数据并钳零，避免轴丢失
    m = (KX >= -1e-12) & (KY >= -1e-12)
    X, Y, V = KX[m].ravel(), KY[m].ravel(), A[m].ravel()
    X = np.where(np.abs(X) < eps, 0.0, X)
    Y = np.where(np.abs(Y) < eps, 0.0, Y)

    # 生成四次旋转后的坐标与相位（90°:+pi => 乘 -1）
    R0 = ( X,  Y,  V)
    R1 = (-Y,  X, -V)
    R2 = (-X, -Y,  V)
    R3 = ( Y, -X, -V)

    # 收集所有旋转后的坐标以构建 1D 轴（并强制包含 0）
    def clamp0(arr):
        arr = np.asarray(arr); arr[np.abs(arr) < eps] = 0.0; return np.round(arr, 12)
    all_x = np.concatenate([clamp0(r[0]) for r in (R0,R1,R2,R3)])
    all_y = np.concatenate([clamp0(r[1]) for r in (R0,R1,R2,R3)])
    ax = np.unique(all_x); ay = np.unique(all_y)
    if ax.size==0 or ax[0] > 0: ax = np.insert(ax, 0, 0.0)
    if ay.size==0 or ay[0] > 0: ay = np.insert(ay, 0, 0.0)
    ax.sort(); ay.sort()

    AF = np.full((ay.size, ax.size), np.nan+1j*np.nan, dtype=np.complex128)
    filled = np.zeros_like(AF, dtype=bool)

    # 容差：步长的 1/4
    dx = np.min(np.diff(ax)) if ax.size>1 else 1e-9
    dy = np.min(np.diff(ay)) if ay.size>1 else 1e-9
    tx, ty = 0.25*dx, 0.25*dy

    def place(x, y, v):
        x = clamp0(x); y = clamp0(y); v = np.asarray(v).ravel()
        ix = np.clip(np.searchsorted(ax, x), 0, ax.size-1).astype(int)
        iy = np.clip(np.searchsorted(ay, y), 0, ay.size-1).astype(int)
        for xi, yi, val, xv, yv in zip(ix, iy, v, x, y):
            xv = float(xv); yv = float(yv)
            if abs(float(ax[xi]) - xv) <= tx and abs(float(ay[yi]) - yv) <= ty:
                AF[yi, xi] = 0.5*(AF[yi, xi] + val) if filled[yi, xi] else val
                filled[yi, xi] = True

    # 填充四象限（R1/R3 已经整体 +π）
    place(*R0); place(*R1); place(*R2); place(*R3)

    # 保障轴上有值：对 x=0 列与 y=0 行做单侧填补
    j0 = int(np.where(np.isclose(ax, 0.0, atol=eps))[0][0])
    i0 = int(np.where(np.isclose(ay, 0.0, atol=eps))[0][0])
    # x=0 列
    for i in range(AF.shape[0]):
        if np.isnan(AF[i, j0]):
            # 优先取正侧
            right = next((j for j in range(j0+1, AF.shape[1]) if not np.isnan(AF[i, j])), None)
            left  = next((j for j in range(j0-1, -1, -1)        if not np.isnan(AF[i, j])), None)
            AF[i, j0] = AF[i, right] if right is not None else (AF[i, left] if left is not None else AF[i, j0])
    # y=0 行
    for j in range(AF.shape[1]):
        if np.isnan(AF[i0, j]):
            up   = next((i for i in range(i0+1, AF.shape[0]) if not np.isnan(AF[i, j])), None)
            down = next((i for i in range(i0-1, -1, -1)      if not np.isnan(AF[i, j])), None)
            AF[i0, j] = AF[up, j] if up is not None else (AF[down, j] if down is not None else AF[i0, j])

    return {xk: ax, yk: ay}, AF


# =========================
# 序列化
# =========================
def load_bundle(pkl_path):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"未找到数据文件：{pkl_path}")
    with open(pkl_path, 'rb') as f:
        bundle = pickle.load(f)
    # 基本校验
    need = ['m1_full','m2_full','Z_full','phi_full','tanchi_full','Qlog_full']
    for k in need:
        if k not in bundle:
            raise KeyError(f"pkl 中缺少键：{k}")
    return bundle

# ------------------------------
# 栅格插值（含角度变量）
# ------------------------------
def _find_cell_edges(grid):
    """由单调一维网格推算每个点的 cell 左边界坐标，用于双线性插值的索引定位。"""
    g = np.asarray(grid)
    d = np.diff(g)
    # 边界：中点
    lefts = np.empty_like(g)
    lefts[1:] = (g[:-1] + g[1:]) / 2
    # 最左边 extrapolate
    lefts[0] = g[0] - d[0]/2
    return lefts

def _bilinear_interpolate(xgrid, ygrid, F, xq, yq):
    """
    在规则网格上对标量场 F 做双线性插值。
    xgrid,ygrid 升序一维；F.shape=(len(xgrid),len(ygrid))；xq,yq 可为一维同长。
    """
    xg = np.asarray(xgrid); yg = np.asarray(ygrid)
    F = np.asarray(F)
    xq = np.asarray(xq); yq = np.asarray(yq)
    assert F.shape[:2] == (xg.size, yg.size)

    # cell 边界
    xl = _find_cell_edges(xg); yl = _find_cell_edges(yg)

    # 每个查询点找到所在 cell 的左上角索引 i,j
    i = np.clip(np.searchsorted(xl, xq, side='right') - 1, 0, xg.size-2)
    j = np.clip(np.searchsorted(yl, yq, side='right') - 1, 0, yg.size-2)

    x0 = xg[i]; x1 = xg[i+1]
    y0 = yg[j]; y1 = yg[j+1]
    # 权重
    tx = (xq - x0) / (x1 - x0 + 1e-12)
    ty = (yq - y0) / (y1 - y0 + 1e-12)

    f00 = F[i, j]
    f10 = F[i+1, j]
    f01 = F[i, j+1]
    f11 = F[i+1, j+1]
    return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11

def _bilinear_interpolate_angle_pi(xgrid, ygrid, PHI, xq, yq):
    """对模 π 的角度 φ 做双线性插值：先映射到 (cos2φ, sin2φ) 再插值，最后反算 φ/2。"""
    PHI = _wrap_angle_pi(PHI)
    c2 = np.cos(2*PHI); s2 = np.sin(2*PHI)
    c2q = _bilinear_interpolate(xgrid, ygrid, c2, xq, yq)
    s2q = _bilinear_interpolate(xgrid, ygrid, s2, xq, yq)
    return 0.5*np.arctan2(s2q, c2q)

# ------------------------------
# 等频线提取 & 采样
# ------------------------------
def extract_isofreq_paths(xgrid, ygrid, FREQ, level):
    """
    用临时 2D Axes 提取等频线；兼容不同 Matplotlib 版本（使用 cs.allsegs）。
    返回若干 path (N_i, 2) 数组列表（单位在 x,y 坐标）。
    """
    import numpy as np
    import matplotlib.pyplot as plt

    xg = np.asarray(xgrid)
    yg = np.asarray(ygrid)
    F = np.asarray(FREQ, dtype=float)

    if F.ndim != 2 or F.shape != (xg.size, yg.size):
        raise ValueError(f"extract_isofreq_paths: FREQ 形状应为 ({xg.size}, {yg.size})，当前为 {F.shape}")

    # 掩蔽 NaN
    if np.isnan(F).any():
        F = np.ma.array(F, mask=np.isnan(F))

    fmin = float(np.nanmin(F))
    fmax = float(np.nanmax(F))
    if not (fmin <= level <= fmax):
        print(f"[WARN] isofreq={level} 超出可用范围 [{fmin:.6g}, {fmax:.6g}]，不提取等频线。")
        return []

    # 退化面（整个矩阵常量等于 level）时做微扰
    if np.allclose(F, level, atol=0, rtol=0):
        level = float(level) + 1e-12

    # 构造规则网格（保持 indexing='ij' 与外部一致）
    X, Y = np.meshgrid(xg, yg, indexing='ij')

    fig, ax = plt.subplots()  # 临时 2D 轴，避免 3D 轴返回 QuadContourSet3D
    try:
        cs = ax.contour(X, Y, F, levels=[level])
        paths = []
        # 关键：不同版本统一用 allsegs（list[levels] -> list[segments ndarray(N,2)]) 来取线段
        if hasattr(cs, "allsegs") and len(cs.allsegs) > 0:
            for seg in cs.allsegs[0]:  # 我们只画了一个 level
                seg = np.asarray(seg)
                if seg.ndim == 2 and seg.shape[0] >= 2 and seg.shape[1] == 2:
                    paths.append(seg.copy())
        else:
            # 极端兜底（理论上不会触发）
            print("[WARN] Matplotlib ContourSet 未提供 allsegs，未能提取等频线。")
            paths = []
    finally:
        plt.close(fig)

    return paths

def resample_path_uniform(path_xy, step=None, npts=400):
    """
    将路径按弧长重采样为等距点列；若提供 step（物理步长），优先用 step；否则用 npts。
    返回 (xy_s, s)，xy_s shape=(M,2)，s 为累计弧长（从 0 开始）。
    """
    P = np.asarray(path_xy)
    d = np.sqrt(np.sum(np.diff(P, axis=0)**2, axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    if step is not None:
        total = s[-1]
        M = max(2, int(math.ceil(total/step))+1)
        s_new = np.linspace(0, total, M)
    else:
        s_new = np.linspace(0, s[-1], npts)

    # 线性插值坐标
    x = np.interp(s_new, s, P[:,0])
    y = np.interp(s_new, s, P[:,1])
    xy = np.stack([x,y], axis=1)
    return xy, s_new

def sample_fields_along_path(xgrid, ygrid, fields, path_xy, step=None, npts=400):
    """
    在 path 上双线性插值采样：fields 是 dict，可包含 'phi', 'chi', 'Q', 'freq' 等。
    φ 用 _bilinear_interpolate_angle_pi，其他用普通双线性。
    返回 samples: dict + 统一的 s 弧长。
    """
    xy_s, s = resample_path_uniform(path_xy, step=step, npts=npts)
    xq, yq = xy_s[:,0], xy_s[:,1]

    out = {'s': s, 'x': xq, 'y': yq}
    for name, arr in fields.items():
        if name.lower() in ['phi','phi_full','angle','polar_angle']:
            out[name] = _bilinear_interpolate_angle_pi(xgrid, ygrid, arr, xq, yq)
        else:
            out[name] = _bilinear_interpolate(xgrid, ygrid, arr, xq, yq)
    return out

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

def plot_isofreq_contours2D(ax, xgrid, ygrid, FREQ, levels,
                            colors=None, linewidths=1, linestyles='-',
                            return_paths=True, **contour_kwargs):
    """
    在给定 2D Axes 上绘制等频线，并（可选）返回由各 level 提取的 paths。
    - ax: matplotlib.axes.Axes（2D）
    - xgrid, ygrid: 一维升序；与模块保持 indexing='ij'
    - FREQ: 2D 数组形状 (len(xgrid), len(ygrid))
    - levels: float 或 list[float]
    - colors/linewidths/linestyles: 与 matplotlib.contour 兼容
    - return_paths: True -> 同时返回 {level: [np.ndarray(N_i,2), ...], ...}
    - 其余参数直接透传给 ax.contour
    返回：
      ax
    """
    import numpy as np

    xg = np.asarray(xgrid); yg = np.asarray(ygrid)
    F = np.asarray(FREQ, dtype=float)
    assert F.shape == (xg.size, yg.size), f"FREQ shape {F.shape} != ({xg.size},{yg.size})"
    if np.isnan(F).any():
        F = np.ma.array(F, mask=np.isnan(F))

    X, Y = np.meshgrid(xg, yg, indexing='ij')
    cs = ax.contour(X, Y, F, levels=np.atleast_1d(levels),
                    colors=colors, linewidths=linewidths,
                    linestyles=linestyles, **contour_kwargs)

    if not return_paths:
        return cs

    # 统一用 allsegs 提取路径，兼容不同 Matplotlib 版本
    paths_dict = {}
    for ilev, lev in enumerate(np.atleast_1d(levels)):
        segs = []
        if hasattr(cs, "allsegs") and ilev < len(cs.allsegs):
            for seg in cs.allsegs[ilev]:
                seg = np.asarray(seg)
                if seg.ndim == 2 and seg.shape[1] == 2 and seg.shape[0] >= 2:
                    segs.append(seg.copy())
        paths_dict[float(lev)] = segs
    return ax

def plot_paths(ax, paths, colors='C0', linewidth=2.0, alpha=1.0, labels=None, zorder=3):
    """
    将已给定的路径列表绘制到 ax 上。
    - paths: list[np.ndarray(N,2)] 或 dict[level -> list[np.ndarray]]
    - colors: 单色或列表，长度与路径数量匹配时逐条使用
    - labels: 可选，与路径数量一致
    返回：Line2D 对象列表
    """
    import numpy as np
    lines = []
    # 统一为 list[list[np.ndarray]]
    if isinstance(paths, dict):
        all_paths = []
        for lev, plist in paths.items():
            all_paths.extend(plist)
    else:
        all_paths = list(paths)

    # 颜色准备
    if isinstance(colors, (list, tuple, np.ndarray)):
        color_list = list(colors)
    else:
        color_list = None

    for i, p in enumerate(all_paths):
        p = np.asarray(p)
        c = color_list[i] if color_list and i < len(color_list) else colors
        lab = labels[i] if (labels and i < len(labels)) else None
        ln, = ax.plot(p[:,0], p[:,1], color=c, lw=linewidth, alpha=alpha, label=lab, zorder=zorder)
        lines.append(ln)
    return lines

def plot_polarization_ellipses(ax, xgrid, ygrid, phi, chi,
                               step=(2, 2), scale=None,
                               color_by='chi', cmap=None,
                               clim=None, edgecolor='k', lw=2, alpha=0.9,
                               zorder=2):
    """
    在给定 2D Axes 上用椭圆绘制偏振场（逐点贴片）。
    - phi: 模 π 的线偏振角（rad），主轴方向
    - chi: 圆偏振度 S3/S0 ∈ [-1,1]，与椭圆倾角 ε 满足 sin(2ε)=chi，轴比 b/a=|tan ε|
    - step: (sx, sy) 栅格抽样步长，避免贴片过密
    - scale: 每个贴片的“直径”标度（数据坐标）；默认 0.8*min(dx,dy)
    - color_by: 'chi' | 'phi' | None  （决定贴片面颜色）
    - cmap: 默认 'RdBu' 对 chi、'twilight' 对 phi
    - clim: (vmin, vmax) 手动颜色范围
    返回：PatchCollection
    """
    import numpy as np
    from matplotlib import cm

    xg = np.asarray(xgrid); yg = np.asarray(ygrid)
    PHI = _wrap_angle_pi(np.asarray(phi))
    CHI = np.asarray(chi)
    assert PHI.shape == (xg.size, yg.size)
    assert CHI.shape == (xg.size, yg.size)

    # 栅格间距与 scale
    dx = np.min(np.diff(xg)) if xg.size > 1 else 1.0
    dy = np.min(np.diff(yg)) if yg.size > 1 else 1.0
    if scale is None:
        scale = 0.8*min(dx, dy)

    sx, sy = step
    xs = range(0, xg.size, max(1, int(sx)))
    ys = range(0, yg.size, max(1, int(sy)))

    patches = []
    colors = []

    for i in xs:
        for j in ys:
            phi_ij = PHI[i, j]
            chi_ij = np.clip(CHI[i, j], -1.0, 1.0)

            # 椭圆几何：ε 为椭圆倾角（ellipticity angle）
            eps = 0.5*np.arcsin(chi_ij)            # ε ∈ [-π/4, π/4]
            a = scale                               # 设定主轴直径
            b = scale * abs(np.tan(eps))            # 副轴直径（|tan ε|）
            b = max(b, 1e-3*scale)                  # 下限，避免完全退化

            angle_deg = np.degrees(phi_ij)          # Ellipse 的旋转角（度）
            e = Ellipse((xg[i], yg[j]), width=a, height=b, angle=angle_deg)
            patches.append(e)

            if color_by == 'chi':
                colors.append(chi_ij)
            elif color_by == 'phi':
                colors.append((phi_ij + np.pi/2) % np.pi)  # [0,π)
            else:
                colors.append(0.0)

    pc = PatchCollection(patches, facecolor='none', edgecolor=edgecolor, linewidths=lw, alpha=alpha, zorder=zorder)

    if color_by in ('chi', 'phi'):
        if cmap is None:
            cmap = 'RdBu' if color_by == 'chi' else 'twilight'
        cmap_obj = cm.get_cmap(cmap)
        arr = np.asarray(colors, dtype=float)
        if clim is None:
            if color_by == 'chi':
                vmin, vmax = -1.0, 1.0
            else:
                vmin, vmax = 0.0, np.pi
        else:
            vmin, vmax = clim
        # 归一化并设置面颜色
        normed = (arr - vmin) / (vmax - vmin + 1e-12)
        facecolors = cmap_obj(np.clip(normed, 0, 1))
        pc.set_facecolor(facecolors)
        pc.set_edgecolor(facecolors)  # 用填充色更直观；如需边界改为 edgecolor
    ax.add_collection(pc)
    ax.set_aspect('equal', adjustable='box')
    return pc

