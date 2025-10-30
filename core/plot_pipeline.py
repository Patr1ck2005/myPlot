import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


def draw_shapes(ax, x, y, plot_type='line', color='blue', marker=None, linestyle='-', alpha=1.0, label=None,
                hide_default_ticks=False, hide_default_ticklabels=False):
    """
    简化核心绘图函数：本质上用普通的plot等方法，只绘制美术形状，可选隐藏默认坐标标注。

    参数:
    - ax: matplotlib的Axes对象
    - x: x数据 (array-like, 单曲线)
    - y: y数据 (array-like, 单曲线)
    - plot_type: 类型 ('line', 'scatter', 'bar', 'hist')
    - color: 颜色 (str)
    - marker: 标记 (str or None, 如 'o')
    - linestyle: 线型 (str, 如 '-')
    - alpha: 透明度 (float, 0-1)
    - label: 形状标签 (str or None，用于潜在图例)
    - hide_default_ticks: 是否隐藏默认轴线/刻度/标签 (bool)
    - hide_default_ticklabels: 是否隐藏默认轴线/刻度/标签 (bool)

    返回: 加工后的ax
    """
    # 简单检查（仅长度匹配，非hist时）
    if plot_type != 'hist' and len(x) != len(y):
        raise ValueError("x 和 y 长度不匹配")

    # 绘制形状（本质上普通的ax方法）
    if plot_type == 'line':
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, alpha=alpha, label=label)
    elif plot_type == 'scatter':
        ax.scatter(x, y, color=color, marker=marker or 'o', alpha=alpha, label=label)
    elif plot_type == 'bar':
        ax.bar(x, y, color=color, alpha=alpha, label=label)
    elif plot_type == 'hist':
        ax.hist(y, bins='auto', color=color, alpha=alpha, label=label)
    else:
        raise ValueError(f"不支持的plot_type: {plot_type}")

    # 可选隐藏默认坐标标注
    if hide_default_ticks:
        # 隐藏刻度与标签
        ax.set_xticks([])
        ax.set_yticks([])
    if hide_default_ticklabels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return ax


def add_annotations(ax, title=None, xlabel=None, ylabel=None,
                    xlim=None, ylim=None,
                    show_legend=False, legend_loc='best',
                    add_grid=False, grid_style='-', grid_alpha=0.5,
                    # X轴 ticks 定制化参数
                    xtick_mode='auto', xtick_count=None, xticks=None, xtick_labels=None,
                    # Y轴 ticks 定制化参数
                    ytick_mode='auto', ytick_count=None, yticks=None, ytick_labels=None):
    """
    可选标注函数：添加标题、标签、图例、轴控制和定制化 ticks。

    参数:
        ax (matplotlib.axes.Axes): 当前的Axes对象。
        title (str, optional): 图表标题。
        xlabel (str, optional): X轴标签。
        ylabel (str, optional): Y轴标签。
        xlim (tuple, optional): X轴显示范围 (min, max)。
        ylim (tuple, optional): Y轴显示范围 (min, max)。
        show_legend (bool): 是否显示图例。
        legend_loc (str): 图例位置，例如 'best', 'upper left'。
        add_grid (bool): 是否添加网格。
        grid_style (str): 网格线样式，例如 '--'。
        grid_alpha (float): 网格线透明度。

        # X轴 ticks 定制化参数
        xtick_mode (str): X轴 ticks 的模式。可选值：
                          'auto' (默认): Matplotlib 自动确定。
                          'approx_count': 尝试指定 ticks 的大致数量 (通过 xtick_count)。
                          'manual': 完全手动指定 ticks 位置 (通过 xticks)。
        xtick_count (int, optional): 当 xtick_mode='approx_count' 时，指定 X轴 ticks 的大致数量。
        xticks (list or numpy.array, optional): 当 xtick_mode='manual' 时，X轴 ticks 的具体位置。
        xtick_labels (list of str, optional): 当 xtick_mode='manual' 时，X轴 ticks 对应的标签。

        # Y轴 ticks 定制化参数
        ytick_mode (str): Y轴 ticks 的模式。可选值：
                          'auto' (默认): Matplotlib 自动确定。
                          'approx_count': 尝试指定 ticks 的大致数量 (通过 ytick_count)。
                          'manual': 完全手动指定 ticks 位置 (通过 yticks)。
        ytick_count (int, optional): 当 ytick_mode='approx_count' 时，指定 Y轴 ticks 的大致数量。
        yticks (list or numpy.array, optional): 当 ytick_mode='manual' 时，Y轴 ticks 的具体位置。
        ytick_labels (list of str, optional): 当 ytick_mode='manual' 时，Y轴 ticks 对应的标签。

    返回: 加工后的ax
    """
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    # 先设置轴范围，这会影响 ticks 的自动计算
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    # 处理 X 轴 ticks
    if xtick_mode == 'auto':
        # Matplotlib 默认行为，无需额外设置
        pass
    elif xtick_mode == 'approx_count':
        if xtick_count is not None:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=xtick_count))
    elif xtick_mode == 'manual':
        if xticks is not None:
            ax.set_xticks(xticks)
            if xtick_labels is not None:
                ax.set_xticklabels(xtick_labels)
        else:
            print("Warning: xtick_mode is 'manual' but xticks are not provided. X-axis ticks will be auto-set.")

    # 处理 Y 轴 ticks
    if ytick_mode == 'auto':
        # Matplotlib 默认行为，无需额外设置
        pass
    elif ytick_mode == 'approx_count':
        if ytick_count is not None:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ytick_count))
    elif ytick_mode == 'manual':
        if yticks is not None:
            ax.set_yticks(yticks)
            if ytick_labels is not None:
                ax.set_yticklabels(ytick_labels)
        else:
            print("Warning: ytick_mode is 'manual' but yticks are not provided. Y-axis ticks will be auto-set.")

    if show_legend: ax.legend(loc=legend_loc)
    if add_grid: ax.grid(True, linestyle=grid_style, alpha=grid_alpha)

    return ax

