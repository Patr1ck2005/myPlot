from typing import Optional, Dict, Any, Tuple, Sequence, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
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
    assert z1.shape == z2.shape == Mx.shape, "z1/z2 与网格尺寸不一致: {} vs {}".format(z1.shape, Mx.shape)
    if z3 is not None:
        assert z3.shape == z1.shape, "z3 与网格尺寸不一致"

    # colormap & normalize
    cmap_name = mapping.get('cmap', 'hot')
    cmap = mpl.colormaps.get_cmap(cmap_name)

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


import s3dlib.surface as s3d
import s3dlib.cmap_utilities as s3dcmap


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
        norm_z2: Optional[Normalize] = None,
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

    cmap_func = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap

    dmin = np.nanmin(z2_f)
    dmax = np.nanmax(z2_f)
    dspan = dmax - dmin

    cmap_i = s3dcmap.op_cmap(
        lambda t: cmap_func(norm_z2(dmin + t * dspan))[:, :3].T,
        rgb=True,
        name=None,
    )
    surface.map_cmap_from_datagrid(z2_f, cmap=cmap_i)
    surface._facecolor3d[:, 3] = alpha

    surface.map_geom_from_datagrid(z1_f, scale=geom_scale)
    # ---- xy: map from native [-1, 1] to physical [xmin, xmax], [ymin, ymax] ----
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_halfspan = 0.5 * (x_max - x_min)
    y_halfspan = 0.5 * (y_max - y_min)
    surface.transform(scale=[x_halfspan, y_halfspan, 1.0])
    surface.transform(translate=[x_center, y_center, 0.0])

    surface.transform(translate=[0.0, 0.0, z_offset])

    if shade:
        surface.shade().hilite(hilite)

    return surface


def export_obj(
        filename: str,
        vertices: np.ndarray,  # (N, 3)
        faces: np.ndarray  # (M, 3) index from 0
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
        rez: int = 4,
        basetype: str = "oct1",
        cmap: str = "hot",
        vmin=None,
        vmax=None,
        elev: float = 30,
        azim: float = 25,
        shade: bool = False,
        hilite: float = 0,
        alpha_default: float = 1.0,
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

    combined = None

    all_vertices = []
    all_faces = []
    v_offset = 0

    for z1, z2, zmin_i, zmax_i in zip(z1_list, z2_list, zmins, zmaxs):
        zrange_i = zmax_i - zmin_i
        geom_scale_i = zrange_i
        z_offset_i = zmin_i
        surf = s3d_build_planar_surface_from_arrays(
            x, y, z1, z2,
            rez=rez, basetype=basetype, cmap=cmap,
            alpha=alpha_default,
            shade=shade, hilite=hilite,
            geom_scale=geom_scale_i,
            z_offset=z_offset_i,
            norm_z2=norm_z2,
        )
        combined = surf if combined is None else (combined + surf)
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

    # combined.set_edgecolor('none')
    # combined.set_edgecolor('face')
    if alpha_default < 1.0:
        combined._edgecolor3d[:, 3] = 0

    # ---- colorbar mappable（数值解释权威）----
    mappable = ScalarMappable(norm=norm_z2, cmap=mpl.colormaps.get_cmap(cmap))
    mappable.set_array([])

    if True:
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)

        export_obj("surface.obj", vertices, faces)
    return ax, combined, mappable


if __name__ == '__main__':
    def make_demo_bands(kx: np.ndarray, ky: np.ndarray):
        """
        生成 3 条“合理但合成”的能带数据：
        - z1: 频率实部（随 k 变化的平滑曲面）
        - z2: Q 因子（给出跨 1e3~1e5 的变化，适合做颜色）
        返回: (z1_list, z2_list)
        """
        # 注意：这里 z 的 shape 需要是 (len(kx), len(ky))
        KX, KY = np.meshgrid(kx, ky, indexing="ij")

        # --- 三条能带的频率实部（z1） ---
        # band 1: 近似余弦色散
        f1 = 1.00 + 0.18 * np.cos(KX) + 0.10 * np.cos(KY) + 0.03 * np.cos(KX + KY)
        # band 2: 近似鞍点/交叉形色散
        f2 = 1.55 + 0.12 * np.sin(KX) * np.cos(KY) + 0.05 * np.cos(2 * KY)
        # band 3: 更高频的“二倍谐波”特征
        f3 = 2.05 + 0.10 * np.cos(2 * KX) + 0.08 * np.sin(2 * KY) + 0.02 * np.cos(KX - 2 * KY)

        # --- 三条能带的 Q 因子（z2） ---
        # 做一个“中心高Q、边缘低Q”的模式，并叠加一些各向异性，让图更像真数据
        def q_field(cx, cy, s, base=2e3, peak=9e4):
            r2 = (KX - cx) ** 2 + (KY - cy) ** 2
            core = np.exp(-r2 / (2 * s * s))
            ripple = 0.15 * np.cos(2 * KX) * np.cos(KY)  # 小幅纹理
            Q = base + peak * np.clip(core * (1.0 + ripple), 0.0, None)
            return Q

        Q1 = q_field(cx=0.3, cy=-0.2, s=1.3, base=1e3, peak=1e3)
        Q2 = q_field(cx=-0.8, cy=0.4, s=1.0, base=1e3, peak=7e4)
        Q3 = q_field(cx=0.5, cy=1.0, s=0.9, base=3e3, peak=1.0e5)

        z1_list = [f1, f2, f3]
        z2_list = [Q1, Q2, Q3]
        return z1_list, z2_list


    def demo_multisurface_bandstructure():
        # k-space 网格
        kx = np.linspace(-np.pi, np.pi, 70)
        ky = np.linspace(-np.pi, np.pi, 70)

        z1_list, z2_list = make_demo_bands(kx, ky)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # 画多表面能带
        ax, combined, mappable = s3d_plot_multi_surfaces_combined(
            ax=ax,
            x=kx,
            y=ky,
            z1_list=z1_list,  # 高度：频率实部
            z2_list=z2_list,  # 颜色：Q
            rez=4,
            basetype="oct1",
            cmap="hot",  # 也可用 "viridis" 等
            # vmin/vmax 不传则自动取 Q 的全局 min/max；也可手动指定：
            vmin=1e2, vmax=1e5,
            elev=28,
            azim=35,
            shade=False,
            hilite=0.6,
            alpha_default=0.5,
        )

        # 轴标签
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.set_zlabel(r"$\Re(\omega)$ (arb.)")
        ax.set_title("3-band surface (height=Re(freq), color=Q)")

        # Q 的颜色条（解释的是原始 Q 的数值语义）
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label("Mode Q factor")

        plt.tight_layout()
        plt.show()

        print("Done. Note: surface.obj has been exported in current directory (by your function).")


    demo_multisurface_bandstructure()
