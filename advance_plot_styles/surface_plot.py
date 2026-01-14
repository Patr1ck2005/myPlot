from typing import Optional, Dict, Any, Tuple

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
    mx: np.ndarray,
    my: np.ndarray,
    z1: np.ndarray,  # 高度
    z2: np.ndarray,  # 颜色值
    z3: Optional[np.ndarray] = None,  # 透明度值(可选；未给则全不透明)
    rbga: Optional[np.ndarray] = None,  # 直接给出颜色映射
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
    Mx, My = np.meshgrid(mx, my, indexing='ij')
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


    if rbga is None:
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
        colors = rbga

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
