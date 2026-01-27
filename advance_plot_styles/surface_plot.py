from typing import Optional, Dict, Any, Tuple, Sequence, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize


def _auto_norm(data: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> Normalize:
    """根据数据与可选的 vmin/vmax 生成 Normalize，自动忽略 NaN/inf。"""
    if vmin is None or vmax is None:
        finite = np.isfinite(data)
        if not finite.any():
            # 兜底，避免全 NaN 导致异常
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax
        else:
            if vmin is None:
                vmin = float(np.nanmin(data[finite]))
            if vmax is None:
                vmax = float(np.nanmax(data[finite]))
            if vmin == vmax:
                # 扩一点范围，避免除零
                eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-9
                vmin -= eps
                vmax += eps
    return Normalize(vmin=vmin, vmax=vmax, clip=True)

def plot_advanced_surface(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z1: np.ndarray,  # 高度
    z2: np.ndarray,  # 颜色值
    z3: Optional[np.ndarray] = None,  # 透明度值(可选；未给则全不透明)
    rgba: Optional[np.ndarray] = None,  # 直接给出颜色映射
    *,
    mapping: Dict[str, Any],
    elev: float = 30,
    azim: float = 25,
    x_key: str = '',
    y_key: str = '',
    z_label: str = '',
    rstride: int = 1,
    cstride: int = 1,
    box_aspect: list = [1, 1, 1],
    **kwargs
) -> Tuple[plt.Axes, ScalarMappable]:
    """
    在同一张 3D 图上绘制一个带面：z1 控制高度，z2 控制颜色，z3 控制 alpha。
    参数
    ----
    mapping: 形如 {
        'cmap': 'hot',
        'z2': {'vmin': a, 'vmax': b},   # 可选；未给则自动取数据范围
        'z3': {'vmin': c, 'vmax': d},   # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
    }
    返回
    ----
    (ax, mappable_for_colorbar)
    """

    # 网格，按 (i,j) 对齐
    Mx, My = np.meshgrid(x, y, indexing='ij')
    assert z1.shape == z2.shape == Mx.shape, "z1/z2 与网格尺寸不一致"
    if z3 is not None:
        assert z3.shape == z1.shape, "z3 与网格尺寸不一致"

    # colormap & normalize
    cmap_name = mapping.get('cmap', 'hot')
    cmap = get_cmap(cmap_name)

    z2_cfg = mapping.get('z2', {})
    norm_z2 = _auto_norm(z2, z2_cfg.get('vmin'), z2_cfg.get('vmax'))

    if z3 is not None:
        z3_cfg = mapping.get('z3', {})
        norm_z3 = _auto_norm(z3, z3_cfg.get('vmin'), z3_cfg.get('vmax'))
        alphas = norm_z3(z3)  # 0~1
    elif 'alpha' in kwargs:
        alpha_val = kwargs.pop('alpha')
        alphas = np.full_like(z2, fill_value=alpha_val, dtype=float)
    else:
        alphas = np.ones_like(z2, dtype=float)


    if rgba is None:
        # 生成 RGBA Facecolors
        colors = cmap(norm_z2(z2))
        # 写入 alpha 通道（保留每个面的透明度）
        colors = np.array(colors, copy=True)
        colors[..., 3] = np.clip(alphas, 0.0, 1.0)

        # 对无效数据设为全透明，避免出现杂乱三角面
        invalid = ~np.isfinite(z1) | ~np.isfinite(z2) | (~np.isfinite(z3) if z3 is not None else False)
        if invalid.any():
            colors[invalid] = (0, 0, 0, 0)
    else:
        colors = rgba

    # 绘制
    ax.plot_surface(Mx, My, z1, rstride=rstride, cstride=cstride, facecolors=colors, **kwargs)

    # 轴与视角
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_zlabel(z_label)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(box_aspect)

    # 为 colorbar 构造可复用的 mappable（只用于 z2 的颜色映射）
    mappable = ScalarMappable(norm=norm_z2, cmap=cmap)
    mappable.set_array([])  # 与 colorbar API 兼容

    return ax, mappable


import numpy as np
import matplotlib.pyplot as plt
import s3dlib.surface as s3d


def s3d_build_planar_surface_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    rez,
    basetype,
    cmap,
    alpha,
    shade,
    hilite,
    cname: str = "Value",
    geom_scale: float = 1.0,
    z_offset: float = 0.0,
) -> s3d.PlanarSurface:

    if z1.shape != (x.size, y.size):
        raise ValueError(f"z1 shape {z1.shape} must be (len(mx), len(my))={(x.size, y.size)}")
    if z2.shape != z1.shape:
        raise ValueError(f"z2 shape {z2.shape} must match z1 shape {z1.shape}")

    # fill invalid for stable mapping
    z1_f = np.array(z1, copy=True)
    z2_f = np.array(z2, copy=True)
    bad = ~np.isfinite(z1_f) | ~np.isfinite(z2_f)
    if bad.any():
        z1_f[bad] = np.nanmean(z1_f[np.isfinite(z1_f)]) if np.isfinite(z1_f).any() else 0.0
        z2_f[bad] = np.nanmean(z2_f[np.isfinite(z2_f)]) if np.isfinite(z2_f).any() else 0.0

    surface = s3d.PlanarSurface(rez, basetype=basetype, cmap=cmap)
    surface.cname = cname

    surface.map_cmap_from_datagrid(z2_f)

    surface.map_geom_from_datagrid(z1_f, scale=float(geom_scale))

    surface.transform(translate=[0.0, 0.0, float(z_offset)])

    surface.set_alpha(float(alpha))

    if shade:
        surface.shade().hilite(hilite)

    return surface


def export_obj(
    filename: str,
    vertices: np.ndarray,   # (N, 3)
    faces: np.ndarray       # (M, 3) index from 0
):
    with open(filename, "w") as f:
        f.write("# exported mesh\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # OBJ 是 1-based
            i, j, k = face + 1
            f.write(f"f {i} {j} {k}\n")

def grid_to_tri_mesh(x, y, z):
    z = np.asarray(z)

    # --- 关键：处理 x, y 是 1D 的情况 ---
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y

    ny, nx = z.shape

    vertices = np.column_stack([
        X.ravel(),
        Y.ravel(),
        z.ravel()
    ])

    faces = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            i0 = iy * nx + ix
            i1 = i0 + 1
            i2 = i0 + nx
            i3 = i2 + 1

            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])

    return np.asarray(vertices), np.asarray(faces)


def s3d_plot_multi_surfaces_combined(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z1_list: Sequence[np.ndarray],
    z2_list: Optional[Sequence[np.ndarray]] = None,
    *,
    rez: int = 3,
    basetype: str = "oct1",
    cmap: str = "hot",
    vmin=None,
    vmax=None,
    elev: float = 30,
    azim: float = 25,
    shade: bool = False,
    hilite: float = 0,
    alpha_default: float = 1.0,
    z_span_plot: float = 1.0,
) -> Tuple[plt.Axes, Any, ScalarMappable]:

    if z2_list is None:
        z2_list = z1_list

    if len(z1_list) == 0:
        raise ValueError("z1_list is empty")
    if len(z2_list) != len(z1_list):
        raise ValueError("z2_list length must match z1_list")

    # ---- colorbar norm（vmin/vmax 只影响颜色解释）----
    all_z2 = np.concatenate([np.asarray(z).ravel() for z in z2_list])
    all_z2 = all_z2[np.isfinite(all_z2)]
    norm_z2 = _auto_norm(all_z2, vmin=vmin, vmax=vmax)
    # 应用到各个 z2
    z2_list = [norm_z2(z2) for z2 in z2_list]

    # ---- 全局 z 范围（决定统一坐标系）----
    zmins = []
    zmaxs = []
    for z1 in z1_list:
        a = np.asarray(z1).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            raise ValueError("Some z1 is all non-finite.")
        zmins.append(float(a.min()))
        zmaxs.append(float(a.max()))

    zmin_all = float(np.min(zmins))
    zmax_all = float(np.max(zmaxs))
    zrange_all = max(zmax_all - zmin_all, 1e-12)

    combined = None

    all_vertices = []
    all_faces = []
    v_offset = 0

    for z1, z2, zmin_i, zmax_i in zip(z1_list, z2_list, zmins, zmaxs):
        zrange_i = max(zmax_i - zmin_i, 1e-12)

        geom_scale_i = float(z_span_plot) * (zrange_i / zrange_all)

        z_offset_i = float(z_span_plot) * ((zmin_i - zmin_all) / zrange_all)

        surf = s3d_build_planar_surface_from_arrays(
            x, y, z1, z2,
            rez=rez, basetype=basetype, cmap=cmap,
            alpha=float(alpha_default),
            shade=shade, hilite=hilite,
            geom_scale=geom_scale_i,
            z_offset=z_offset_i,
        )
        combined = surf if combined is None else (combined + surf)

        if True:
            z_plot = (
                    z_offset_i +
                    geom_scale_i * (z1 - zmin_i) / zrange_i
            )

            verts, faces = grid_to_tri_mesh(x, y, z_plot)

            all_vertices.append(verts)
            all_faces.append(faces + v_offset)
            v_offset += verts.shape[0]

    ax.add_collection3d(combined)
    ax.view_init(elev=elev, azim=azim)

    # ---- colorbar mappable（数值解释权威）----
    mappable = ScalarMappable(norm=norm_z2, cmap=get_cmap(cmap))
    mappable.set_array([])

    if True:
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)

        export_obj("surface.obj", vertices, faces)
    return ax, combined, mappable
