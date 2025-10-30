from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable

GridCoords = Dict[str, np.ndarray]
ZArray = np.ndarray  # object 数组（你的 Z / Z_target 结构）

# -----------------------------
# 数据抽取 / 整理
# -----------------------------

def extract_basic_analysis_fields(
    additional_Z_grouped: ZArray,
    band_index: int,
    z_keys: Sequence[str],
    *,
    freq_key: Optional[str] = None,
    q_key: str = '品质因子 (1)',
    tanchi_key: str = 'tanchi (1)',
    phi_key: str = 'phi (rad)'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 additional_Z_grouped 中提取：phi, tanchi, log10(Q), freq(real)
    - 不再假设固定顺序，而是根据 z_keys 动态定位各列索引。
    - freq_key 缺省时会优先使用 '特征频率 (THz)'，若不存在则退回 '频率 (Hz)'.
    结构：additional_Z_grouped[i][j][band][k]，其中 k 对应 z_keys 的索引。
    """
    # 建立 key->idx 映射
    key_to_idx = {k: i for i, k in enumerate(z_keys)}

    # 确定频率字段
    if freq_key is None:
        if '特征频率 (THz)' in key_to_idx:
            freq_key = '特征频率 (THz)'
        elif '频率 (Hz)' in key_to_idx:
            freq_key = '频率 (Hz)'
        else:
            raise KeyError("z_keys 中未找到可用的频率字段：'特征频率 (THz)' 或 '频率 (Hz)'")

    required = [freq_key, q_key, tanchi_key, phi_key]
    for rk in required:
        if rk not in key_to_idx:
            raise KeyError(f"z_keys 缺少必要字段：{rk}")

    freq_idx = key_to_idx[freq_key]
    q_idx = key_to_idx[q_key]
    tanchi_idx = key_to_idx[tanchi_key]
    phi_idx = key_to_idx[phi_key]

    H, W = additional_Z_grouped.shape[:2]
    phi = np.zeros((H, W))
    tanchi = np.zeros((H, W))
    qlog = np.zeros((H, W))
    freq_real = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            band = additional_Z_grouped[i, j][band_index]
            freq = band[freq_idx]
            Q = band[q_idx]
            t = band[tanchi_idx]
            p = band[phi_idx]
            phi[i, j] = float(p)
            tanchi[i, j] = float(t)
            qlog[i, j] = float(np.log10(Q)) if Q > 0 else 0.0
            freq_real[i, j] = float(freq.real if isinstance(freq, complex) else freq)
    return phi, tanchi, qlog, freq_real

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
    *,
    mapping: Dict[str, Any],
    elev: float = 30,
    azim: float = 25,
    font_size: int = 9,
    x_key: str = 'm1',
    y_key: str = 'm2',
    z_label: str = 'Frequency (normalized)',
    rstride: int = 1,
    cstride: int = 1,
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
    plt.rcParams.update({'font.size': font_size})

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
    else:
        alphas = np.ones_like(z2, dtype=float)

    # 生成 RGBA Facecolors
    colors = cmap(norm_z2(z2))
    # 写入 alpha 通道（保留每个面的透明度）
    colors = np.array(colors, copy=True)
    colors[..., 3] = np.clip(alphas, 0.0, 1.0)

    # 对无效数据设为全透明，避免出现杂乱三角面
    invalid = ~np.isfinite(z1) | ~np.isfinite(z2) | (~np.isfinite(z3) if z3 is not None else False)
    if invalid.any():
        colors[invalid] = (0, 0, 0, 0)

    # 绘制
    ax.plot_surface(Mx, My, z1, rstride=rstride, cstride=cstride, facecolors=colors)

    # 轴与视角
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_zlabel(z_label)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])

    # 为 colorbar 构造可复用的 mappable（只用于 z2 的颜色映射）
    mappable = ScalarMappable(norm=norm_z2, cmap=cmap)
    mappable.set_array([])  # 与 colorbar API 兼容

    return ax, mappable
