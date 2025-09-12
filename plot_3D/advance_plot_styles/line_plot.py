import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.cm as cm

def plot_line_advanced(ax, x_vals, z1, z2=None, z3=None, **kwargs):
    """
    高级复数绘图函数，支持四种样式：基础、动态颜色填充 (A)、纯色填充 (B)、动态颜色线条 (C)。

    参数:
    - ax: Matplotlib 的 Axes 对象 (e.g., fig, ax = plt.subplots() 或 ax = plt.gca())。
    - x_vals: 数组，横轴值。
    - z1: 数组，决定线的位置 (y 值，通常实部)。
    - z2: 数组或 None，决定填充宽度 (通常虚部绝对值)。
    - z3: 数组或 None，决定填充或线条颜色 (通常虚部原始值)。
    - **kwargs: 绘图属性字典。
        - enable_fill: bool, 是否启用 z2 控制填充宽度 (默认 False)。
        - enable_dynamic_color: bool, 是否启用 z3 控制动态颜色 (默认 False)。
        - scale: float, 宽度缩放因子 (默认 0.5)。
        - alpha_line: float, 线透明度 (默认 0.8)。
        - alpha_fill: float, 填充透明度 (默认 0.3)。
        - cmap: str 或 Colormap, 自定义颜色表 (默认 'RdBu' for 动态颜色, 'Blues' for 纯色)。
        - default_color: str, 默认颜色 (默认 'blue')。
        - add_colorbar: bool, 是否添加颜色条 (默认 False)。
        - linewidth_base: float, 基础线宽 (默认 1)。

    返回: 绘图后的 ax 对象。
    """
    # 默认参数
    enable_fill = kwargs.get('enable_fill', False)
    enable_dynamic_color = kwargs.get('enable_dynamic_color', False)
    scale = kwargs.get('scale', 0.5)
    alpha_line = kwargs.get('alpha_line', 0.8)
    alpha_fill = kwargs.get('alpha_fill', 0.3)
    cmap = kwargs.get('cmap', 'RdBu' if enable_dynamic_color else 'Blues')
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    default_color = kwargs.get('default_color', 'blue')
    add_colorbar = kwargs.get('add_colorbar', False)
    linewidth_base = kwargs.get('linewidth_base', 1)

    # 验证数组长度
    n = len(x_vals)
    if len(z1) != n:
        raise ValueError("z1 长度必须与 x_vals 一致")
    if z2 is not None and len(z2) != n:
        raise ValueError("z2 长度必须与 x_vals 一致")
    if z3 is not None and len(z3) != n:
        raise ValueError("z3 长度必须与 x_vals 一致")

    # 处理 z2 (宽度)
    y_upper, y_lower = z1, z1  # 默认无填充
    if z2 is not None and enable_fill:
        widths = np.abs(z2)  # 绝对值用于宽度
        # widths_norm = (widths - np.min(widths)) / (np.max(widths) - np.min(widths) + 1e-8)  # 归一化 [0,1]
        widths_norm = widths
        y_upper = z1 + scale * widths_norm
        y_lower = z1 - scale * widths_norm

    # 处理 z3 (颜色)
    color_vals = z3 if z3 is not None else np.zeros(n)
    norm_color = plt.Normalize(vmin=np.min(color_vals), vmax=np.max(color_vals)) if z3 is not None else None

    # 绘制基础线条（所有模式都包含）
    if enable_dynamic_color and not enable_fill:
        # 样式 C：动态颜色线条 (LineCollection)
        if z3 is None:
            raise ValueError("启用 dynamic_color 且不填充需要 z3")
        points = np.array([x_vals, z1]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, array=color_vals, cmap=cmap, norm=norm_color, linewidth=linewidth_base, alpha=alpha_line)
        ax.add_collection(lc)
    else:
        # 基础样式或填充模式的细线
        ax.plot(x_vals, z1, color=default_color, linewidth=linewidth_base, alpha=alpha_line, label='Base Line')

    # 填充模式（样式 A 或 B）
    if enable_fill:
        if z2 is None:
            raise ValueError("启用 fill 需要 z2")
        if enable_dynamic_color:
            # 样式 A：动态颜色填充 (PolyCollection)
            if z3 is None:
                raise ValueError("启用 dynamic_color 填充需要 z3")
            verts = []
            colors = []
            for i in range(n - 1):
                verts.append([
                    (x_vals[i], y_lower[i]),
                    (x_vals[i], y_upper[i]),
                    (x_vals[i + 1], y_upper[i + 1]),
                    (x_vals[i + 1], y_lower[i + 1])
                ])
                colors.append((color_vals[i] + color_vals[i + 1]) / 2)  # 平滑过渡
            poly = PolyCollection(verts, array=colors, cmap=cmap, norm=norm_color, alpha=alpha_fill)
            ax.add_collection(poly)
        else:
            # 样式 B：纯色填充 (fill_between)
            ax.fill_between(x_vals, y_lower, y_upper, color=default_color, alpha=alpha_fill, label='Fill Width')

    # 添加颜色条
    if add_colorbar:
        if enable_dynamic_color and z3 is not None:
            # 颜色条 for 动态颜色 (z3)
            sm = cm.ScalarMappable(norm=norm_color, cmap=cmap)
            sm.set_array(color_vals)
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('z3 (controls color)')

    # 设置轴限
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(np.min(y_lower) if enable_fill else z1.min(), np.max(y_upper) if enable_fill else z1.max())

    return ax
