import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def plot_scatter_advanced(ax, x_vals, z1, z2=None, z3=None, index=0, **kwargs):
    """
    高级散点图函数，支持四种样式：基础、动态颜色 (A)、动态大小 (B)、动态大小+颜色 (C)。

    参数:
    - ax: Matplotlib 的 Axes 对象 (e.g., fig, ax = plt.subplots() 或 ax = plt.gca())。
    - x_vals: 数组，横轴值。
    - z1: 数组，决定点的位置 (y 值，通常实部)。
    - z2: 数组或 None，决定点的大小 (通常虚部绝对值)。
    - z3: 数组或 None，决定点颜色 (通常虚部原始值，用于 cmap 映射)。
    - index: 绘制的当前散点的次序. 为了自动标记颜色等。
    - **kwargs: 绘图属性字典。
        - enable_size_variation: bool, 是否启用 z2 控制点大小 (默认 False)。
        - enable_dynamic_color: bool, 是否启用 z3 控制动态颜色 (默认 False)。
        - scale: float, 大小缩放因子 (默认 1.0)。
        - alpha: float, 点透明度 (默认 0.8)。
        - cmap: str 或 Colormap, 自定义颜色表 (默认 'RdBu' for 动态颜色, 'Blues' for 纯色)。
        - default_color: str, 默认颜色 (默认 'blue')。
        - s_base: float, 基础点大小 (默认 50)。
        - edge_color: str, 边缘颜色 (默认 'black')。
        - add_colorbar: bool, 是否添加颜色条 (默认 False)。
        - global_color_vmax: float, 全局颜色最大值 (默认 None, 自动)。
        - global_color_vmin: float, 全局颜色最小值 (默认 None, 自动)。
        - linewidth: float, 边缘线宽 (默认 0.5)。

    返回: 绘图后的 ax 对象。
    """
    # 默认参数
    enable_size_variation = kwargs.get('enable_size_variation', False)
    enable_dynamic_color = kwargs.get('enable_dynamic_color', False)
    scale = kwargs.get('scale', 1.0)
    alpha = kwargs.get('alpha', 0.8)
    cmap = kwargs.get('cmap', 'RdBu' if enable_dynamic_color else 'Blues')
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    default_color = kwargs.get('default_color', 'blue')
    s_base = kwargs.get('s_base', 50)
    edge_color = kwargs.get('edge_color', 'black')
    add_colorbar = kwargs.get('add_colorbar', False)
    linewidth = kwargs.get('linewidth', 0.5)
    global_color_vmax = kwargs.get('global_color_vmax', None)
    global_color_vmin = kwargs.get('global_color_vmin', None)

    # 验证数组长度
    n = len(x_vals)
    if len(z1) != n:
        raise ValueError("z1 长度必须与 x_vals 一致")
    if z2 is not None and len(z2) != n:
        raise ValueError("z2 长度必须与 x_vals 一致")
    if z3 is not None and len(z3) != n:
        raise ValueError("z3 长度必须与 x_vals 一致")

    # 处理 z2 (大小)
    scatter_sizes = np.full(n, s_base)  # 默认固定大小
    if z2 is not None and enable_size_variation:
        sizes_abs = np.abs(z2)
        # 可选归一化：sizes_norm = (sizes_abs - np.min(sizes_abs)) / (np.max(sizes_abs) - np.min(sizes_abs) + 1e-8)
        scatter_sizes = s_base + scale * sizes_abs  # 直接缩放，避免归一化以保持原始比例

    # 处理 z3 (颜色)
    color_vals = z3 if z3 is not None else np.full(n, 0)
    if global_color_vmax is not None and global_color_vmin is not None:
        norm_color = Normalize(vmin=global_color_vmin, vmax=global_color_vmax)
    else:
        norm_color = Normalize(vmin=np.min(color_vals), vmax=np.max(color_vals)) if z3 is not None else None

    # 绘制散点（所有模式都包含）
    scatter_kwargs = {
        's': scatter_sizes,
        'alpha': alpha,
        'edgecolors': edge_color,
        'linewidth': linewidth,
        **{k: v for k, v in kwargs.items() if k in ['marker', 'linestyle', 'zorder']}  # 传递 marker 等
    }

    if enable_dynamic_color:
        # 样式 A/C：动态颜色
        if z3 is None:
            raise ValueError("启用 dynamic_color 需要 z3")
        sc = ax.scatter(x_vals, z1, c=color_vals, cmap=cmap, norm=norm_color, **scatter_kwargs)
    else:
        # 基础样式或纯大小模式的纯色
        if 'default_color' not in kwargs:  # 如果未指定，基于 index 自动颜色
            default_color = plt.cm.tab10(index % 10)
        sc = ax.scatter(x_vals, z1, c=default_color, **scatter_kwargs, label='Scatter Points')

    # 添加颜色条
    if add_colorbar and enable_dynamic_color and z3 is not None:
        sm = cm.ScalarMappable(norm=norm_color, cmap=cmap)
        sm.set_array(color_vals)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('z3 (controls point color)')

    # 设置轴限（考虑大小变异稍扩展 ylim 以显示大点）
    ax.set_xlim(x_vals.min(), x_vals.max())
    y_min, y_max = z1.min(), z1.max()
    if enable_size_variation:
        y_extra = np.max(scatter_sizes) / 100  # 粗略扩展，基于点大小
        y_min -= y_extra
        y_max += y_extra
    ax.set_ylim(y_min, y_max)

    return ax


if __name__ == '__main__':
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    z1 = np.sin(x)  # y 值
    z2 = np.random.rand(100) * 10  # 大小控制 (虚部绝对值)
    z3 = np.cos(x)  # 颜色控制 (虚部原始值)
    ax = plot_scatter_advanced(ax, x, z1, z2, z3, enable_size_variation=True, enable_dynamic_color=True,
                               add_colorbar=True)
    plt.show()
