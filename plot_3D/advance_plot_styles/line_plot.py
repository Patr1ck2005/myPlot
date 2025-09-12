import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.cm as cm

def plot_line_advanced(ax, x_vals, z1, z2=None, z3=None, **kwargs):
    """
    高级复数绘图函数，支持五种样式：基础、动态颜色填充 (A)、纯色填充 (B)、动态颜色线条 (C)、渐变填充基于 imshow (D)。

    参数:
    - ax: Matplotlib 的 Axes 对象 (e.g., fig, ax = plt.subplots() 或 ax = plt.gca())。
    - x_vals: 数组，横轴值。
    - z1: 数组，决定线的位置 (y 值，通常实部)。
    - z2: 数组或 None，决定填充宽度 (通常虚部绝对值)。
    - z3: 数组或 None，决定填充或线条颜色 (通常虚部原始值)。
    - **kwargs: 绘图属性字典。
        - enable_fill: bool, 是否启用 z2 控制填充宽度 (默认 False)。
        - enable_dynamic_color: bool, 是否启用 z3 控制动态颜色 (默认 False)。
        - gradient_fill: bool, 是否使用 imshow 渐变填充 (默认 False，优先于样式 A/B)。
        - gradient_direction: str, 渐变方向 ('horizontal', 'vertical', 'z3') (默认 'horizontal')。
        - scale: float, 宽度缩放因子 (默认 0.5)。
        - alpha_line: float, 线透明度 (默认 0.8)。
        - alpha_fill: float, 填充透明度 (默认 0.3)。
        - cmap: str 或 Colormap, 自定义颜色表 (默认 'RdBu' for 动态颜色, 'Blues' for 纯色/渐变)。
        - default_color: str, 默认颜色 (默认 'blue')。
        - add_colorbar: bool, 是否添加颜色条 (默认 False)。
        - linewidth_base: float, 基础线宽 (默认 1)。
        - global_color_vmax: float, ...
        - global_color_vmin: float, ...

    返回: 绘图后的 ax 对象。
    """
    # 默认参数
    enable_fill = kwargs.get('enable_fill', False)
    enable_dynamic_color = kwargs.get('enable_dynamic_color', False)
    gradient_fill = kwargs.get('gradient_fill', False)
    gradient_direction = kwargs.get('gradient_direction', 'horizontal')  # 新增：渐变方向
    scale = kwargs.get('scale', 0.5)
    alpha_line = kwargs.get('alpha_line', 0.8)
    alpha_fill = kwargs.get('alpha_fill', 0.3)
    cmap = kwargs.get('cmap', 'RdBu' if enable_dynamic_color else 'Blues')
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    default_color = kwargs.get('default_color', 'blue')
    edge_color = kwargs.get('edge_color', 'gray')
    add_colorbar = kwargs.get('add_colorbar', False)
    linewidth_base = kwargs.get('linewidth_base', 1)
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

    # 处理 z2 (宽度)
    y_upper, y_lower = z1, z1  # 默认无填充
    if z2 is not None and enable_fill:
        widths = np.abs(z2)  # 绝对值用于宽度
        widths_norm = widths  # 可选归一化：(widths - np.min(widths)) / (np.max(widths) - np.min(widths) + 1e-8)
        y_upper = z1 + scale * widths_norm
        y_lower = z1 - scale * widths_norm

    # 处理 z3 (颜色)
    color_vals = z3 if z3 is not None else np.zeros(n)
    if global_color_vmax is not None and global_color_vmin is not None:
        norm_color = plt.Normalize(vmin=global_color_vmin, vmax=global_color_vmax)
    else:
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

    # 填充模式
    if enable_fill:
        if z2 is None:
            raise ValueError("启用 fill 需要 z2")
        if gradient_fill:
            # 样式 D：渐变填充基于 imshow + 剪裁路径
            # 创建填充路径（多边形：下边界 + 反转上边界 + 闭合）
            verts = np.vstack([np.column_stack([x_vals, y_lower]),
                               np.column_stack([x_vals[::-1], y_upper[::-1]])])
            path = Path(verts)
            patch = PathPatch(path, facecolor='none', edgecolor=edge_color)  # 无色patch，用于剪裁
            ax.add_patch(patch)

            # 修复：总是用当前数据范围设置 extent，而不是 ax.get_（避免初次默认 (0,1) 导致空白）
            xlim = (x_vals.min(), x_vals.max())
            ylim = (np.min(y_lower), np.max(y_upper))

            # 创建渐变图像
            if gradient_direction == 'horizontal':
                # 水平渐变（基于 x_vals）
                gradient_data = np.linspace(0, 1, n).reshape(1, -1)  # 1行n列，值从0到1
            elif gradient_direction == 'vertical':
                # 垂直渐变（基于 y 值）
                y_range = np.linspace(ylim[0], ylim[1], 100)  # 用当前 ylim 范围，确保匹配
                gradient_data = np.linspace(0, 1, 100).reshape(-1, 1)  # 100行1列
            elif gradient_direction == 'z3' and z3 is not None:
                # 基于 z3 的渐变
                gradient_data = norm_color(color_vals).reshape(1, -1)  # 使用归一化的 z3
            else:
                raise ValueError("gradient_direction 必须是 'horizontal', 'vertical' 或 'z3'")

            # 绘制渐变图像
            im = ax.imshow(
                gradient_data,
                cmap=cmap,
                aspect='auto',
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],  # 用当前数据范围，确保覆盖路径无空白
                alpha=alpha_fill,
                clip_path=patch,
                clip_on=True,
                vmin=0,
                vmax=1
            )
        else:
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
        if (enable_dynamic_color or gradient_fill) and z3 is not None:
            # 颜色条 for 动态颜色或渐变 (z3)
            sm = cm.ScalarMappable(norm=norm_color, cmap=cmap)
            sm.set_array(color_vals)
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('z3 (controls color)')

    # 设置轴限（连续调用时，最后一次覆盖为当前范围；若需全局，函数外手动调整）
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(np.min(y_lower) if enable_fill else z1.min(), np.max(y_upper) if enable_fill else z1.max())

    return ax


if __name__ == '__main__':
    # 示例数据
    x_vals = np.linspace(0, 10, 100)
    z1 = np.sin(x_vals)  # 中心线
    z2 = 0.5 * np.ones_like(x_vals)  # 固定宽度
    z3 = np.cos(x_vals)  # 控制颜色

    fig, ax = plt.subplots()
    ax = plot_line_advanced(
        ax, x_vals, z1, z2, z3,
        enable_fill=True,
        gradient_fill=True,  # 启用 imshow 渐变填充
        gradient_direction='z3',  # 指定渐变
        cmap='magma',
        alpha_fill=0.5,
        default_color='gray',
        add_colorbar=False,  # 先不加颜色条
    )
    # ax = plot_line_advanced(
    #     ax, x_vals, z1*2, z2, z3,
    #     enable_fill=True,
    #     gradient_fill=True,  # 启用 imshow 渐变填充
    #     gradient_direction='z3',  # 指定渐变
    #     cmap='magma',
    #     alpha_fill=0.5,
    #     default_color='gray',
    #     add_colorbar=True  # 只在最后加颜色条
    # )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    # 可选：手动调整全局 ylim（如果最后调用范围不足以覆盖所有）
    # all_ymin = min(np.sin(x_vals).min() - 0.5, (2*np.sin(x_vals)).min() - 0.5)
    # all_ymax = max(np.sin(x_vals).max() + 0.5, (2*np.sin(x_vals)).max() + 0.5)
    # ax.set_ylim(all_ymin, all_ymax)

    plt.show()

