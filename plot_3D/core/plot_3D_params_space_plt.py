import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
import re
import os
from .utils import safe_str
from plot_3D.advance_plot_styles.line_plot import plot_line_advanced

# fontsize
fs = 12
plt.rcParams.update({'font.size': fs})

def add_annotations(ax, plot_params):
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
    title = plot_params.get('title', None)
    xlabel = plot_params.get('xlabel', 'X')
    ylabel = plot_params.get('ylabel', "Y")
    zlabel = plot_params.get('zlabel', "Z")
    xlim = plot_params.get('xlim', None)
    ylim = plot_params.get('ylim', None)
    zlim = plot_params.get('zlim', None)
    show_legend = plot_params.get('show_legend', False)
    legend_loc = plot_params.get('legend_loc', 'best')
    add_grid = plot_params.get('add_grid', False)
    grid_style = plot_params.get('grid_style', '--')
    grid_alpha = plot_params.get('grid_alpha', 0.3)

    # X轴 ticks 定制化参数
    xtick_mode = plot_params.get('xtick_mode', 'auto')
    xtick_count = plot_params.get('xtick_count', None)
    xticks = plot_params.get('xticks', None)
    xtick_labels = plot_params.get('xtick_labels', None)

    # Y轴 ticks 定制化参数
    ytick_mode = plot_params.get('ytick_mode', 'auto')
    ytick_count = plot_params.get('ytick_count', None)
    yticks = plot_params.get('yticks', None)
    ytick_labels = plot_params.get('ytick_labels', None)


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

    return ax.get_figure(), ax


# 一维多曲线
def plot_1d_lines(ax, x_vals, y_vals_list, plot_params):
    default_color_list = plot_params.get('default_color_list', None)
    enable_line_fill = plot_params.get('enable_line_fill', True)
    scale = plot_params.get('scale', 0.5)
    log_scale = plot_params.get('log_scale', False)

    y_mins, y_maxs = [], []
    for i, y_vals in enumerate(y_vals_list):
        if default_color_list is not None:
            plot_params['default_color'] = default_color_list[i % len(default_color_list)]
        ax = plot_line_advanced(ax, x_vals, z1=y_vals.real, z2=y_vals.imag, z3=y_vals.imag, **plot_params)

        if enable_line_fill:
            widths = np.abs(y_vals.imag)
            y_mins.append(np.min(y_vals.real - scale * widths))
            y_maxs.append(np.max(y_vals.real + scale * widths))
        else:
            y_mins.append(np.min(y_vals.real))
            y_maxs.append(np.max(y_vals.real))

    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(np.nanmin(y_mins) * 0.98, np.nanmax(y_maxs) * 1.02)
    if log_scale:
        ax.set_yscale('log')

    return ax.get_figure(), ax


# 二维热图
def plot_2d_heatmap(ax, x_vals, y_vals, Z, plot_params):
    cmap1_name = plot_params.get('cmap', 'viridis')
    cmap2_name = plot_params.get('cmap2', 'plasma')
    alpha_val = plot_params.get('alpha', 1.0)
    log_scale = plot_params.get('log_scale', False)

    imshow_aspect = plot_params.get('imshow_aspect', 'auto')
    plot_imaginary = plot_params.get('imag', False)
    add_colorbar = plot_params.get('add_colorbar', False)

    Z_real_plot = Z.real
    Z_imag_plot = Z.imag
    if log_scale:
        Z_real_plot = np.log10(np.abs(Z_real_plot))
        Z_imag_plot = np.log10(np.abs(Z_imag_plot))

    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    surf1 = ax.imshow(
        Z_real_plot.T,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        aspect=imshow_aspect,
        cmap=cmap1_name,
        alpha=alpha_val,
        interpolation='none'
    )

    if Z.dtype == np.complex128 and plot_imaginary:
        ax.pcolormesh(X, Y, Z_imag_plot, cmap=cmap2_name, alpha=alpha_val)

    if add_colorbar:
        ax.get_figure().colorbar(surf1, ax=ax)

    return ax.get_figure(), ax


# 新绘图函数：二维多曲线
def plot_2d_multiline(ax, x_vals, y_vals, Z, plot_params):
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    cmap_name = plot_params.get('cmap', 'viridis')
    y_space = plot_params.get('y_space', 1)
    default_color = plot_params.get('default_color', None)
    default_color_list = plot_params.get('default_color_list', None)
    alpha_val = plot_params.get('alpha', 1.0)
    log_scale = plot_params.get('log_scale', False)
    plot_imaginary = plot_params.get('imag', False)
    enable_legend = plot_params.get('legend', False)
    title = plot_params.get('title', '')

    # down-sampling y_vals
    y_vals = y_vals[::y_space]
    Z = Z[:, ::y_space]

    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=y_vals.min(), vmax=y_vals.max())
    ny = len(y_vals)

    Z_real = Z.real
    Z_imag = Z.imag if np.iscomplexobj(Z) else np.zeros_like(Z.real)
    if log_scale:
        Z_real = np.log10(np.abs(Z_real))
        Z_imag = np.log10(np.abs(Z_imag))

    y_mins, y_maxs = [], []
    for j in range(ny):
        if default_color is not None:
            color = default_color
        elif default_color_list is not None:
            color = default_color_list[j % len(default_color_list)]
        else:
            color = cmap(norm(y_vals[j]))

        ax.plot(x_vals, Z_real[:, j], label=f'Real (y={y_vals[j]:.2f})', color=color, alpha=alpha_val)
        y_mins.append(np.min(Z_real[:, j]))
        y_maxs.append(np.max(Z_real[:, j]))

        if np.iscomplexobj(Z) and plot_imaginary and np.abs(Z_imag[:, j]).max() > 1e-8:
            ax.plot(x_vals, Z_imag[:, j], label=f'Imag (y={y_vals[j]:.2f})',
                    color=color, linestyle='--', alpha=alpha_val)
            y_mins.append(np.min(Z_imag[:, j]))
            y_maxs.append(np.max(Z_imag[:, j]))

    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(np.nanmin(y_mins) * 0.98, np.nanmax(y_maxs) * 1.02)
    if log_scale:
        ax.set_yscale('log')
    if title:
        ax.set_title(title)
    if enable_legend:
        ax.legend()

    return ax.get_figure(), ax


# 修改后的 plot_Z_2D
def plot_Z_2D(subs, x_vals, x_key, y_vals=None, y_key=None, plot_params=None, fixed_params=None, is_1d=True):
    plot_params = plot_params or {}
    fixed_params = fixed_params or {}
    figsize = plot_params.get('figsize', (8, 6))
    save_raw_data = plot_params.get('save_raw_data', True)
    advanced_process = plot_params.get('advanced_process', False)

    save_dir = plot_params.get('save_dir', './rsl/2D/')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join(save_dir, f"raw_data_{timestamp}")
    if save_raw_data:
        os.makedirs(data_dir, exist_ok=True)

    if advanced_process == 'y_mirror':
        x_vals = np.concatenate([-x_vals[::-1], x_vals])
        subs = [np.concatenate([sub[::-1], sub]) for sub in subs]

    fig, ax = plt.subplots(figsize=figsize)

    if is_1d:
        y_vals_list = []
        for sub in subs:
            mask = np.isnan(sub)
            if np.any(mask):
                print(f"Warning: 存在非法数据，已移除。")
                y_vals = sub[~mask]
                temp_x_vals = x_vals[~mask]
            else:
                y_vals = sub
                temp_x_vals = x_vals
            y_vals_list.append(y_vals)

        if save_raw_data:
            for i, y_vals in enumerate(y_vals_list):
                raw_data_dict = {
                    'x': temp_x_vals,
                    'y_real': y_vals.real,
                    'y_imag': y_vals.imag if np.iscomplexobj(y_vals) else np.zeros_like(y_vals.real)
                }
                param_items = [f"{k}-{safe_str(v)}" for k, v in sorted(fixed_params.items())]
                param_items.append(f"curve-{i + 1}")
                filename = f"data_{'_'.join(param_items)}.csv"
                if len(filename) > 200:
                    filename = filename[:200] + ".csv"
                file_path = os.path.join(data_dir, filename)
                df = pd.DataFrame(raw_data_dict)
                df.to_csv(file_path, index=False)
                print(f"原始数据已保存为：{file_path}")

        fig, ax = plot_1d_lines(ax, temp_x_vals, y_vals_list, plot_params)
    else:
        # if len(subs) > 1:
        #     print("Warning: 多 Z 在二维模式下仅使用第一个 Z。")
        y_mins, y_maxs = [], []
        for i, sub in enumerate(subs):
            Z = sub
            y_mins.append(np.min(Z))
            y_maxs.append(np.max(Z))
            if save_raw_data:
                data_dict = {
                    'x_vals': x_vals,
                    'y_vals': y_vals,
                    'Z': Z
                }
                param_items = [f"{k}-{safe_str(v)}" for k, v in sorted(fixed_params.items())]
                param_items.append(f"x-{safe_str(x_key)}_y-{safe_str(y_key)}")
                filename = f"data_heatmap_{'_'.join(param_items)}.pkl"
                if len(filename) > 200:
                    filename = filename[:200] + ".pkl"
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'wb') as f:
                    pickle.dump(data_dict, f)
                print(f"原始数据已保存为：{file_path}")

            plot_type = plot_params.get('plot_type', 'heatmap')
            if plot_type == 'multiline':
                fig, ax = plot_2d_multiline(ax, x_vals, y_vals, Z, plot_params)
                ax.set_ylim(min(y_mins), max(y_maxs))
            else:
                fig, ax = plot_2d_heatmap(ax, x_vals, y_vals, Z, plot_params)
                break
    if plot_params.get('tight_layout', False):
        plt.tight_layout()

    return fig, ax


# 其他函数保持不变
def plot_Z_3D(subs, x_vals, x_key, y_vals, y_key, plot_params=None, fixed_params=None):
    plot_params = plot_params or {}
    fixed_params = fixed_params or {}
    save_raw_data = plot_params.get('save_raw_data', True)

    save_dir = plot_params.get('save_dir', './rsl/2D/')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join(save_dir, f"raw_data_{timestamp}")
    if save_raw_data:
        os.makedirs(data_dir, exist_ok=True)

    if len(subs) > 1:
        print("Warning: 多 Z 在三维模式下仅使用第一个 Z。")
    Z = subs[0]

    if save_raw_data:
        data_dict = {
            'x_vals': x_vals,
            'y_vals': y_vals,
            'Z': Z
        }
        param_items = [f"{k}-{safe_str(v)}" for k, v in sorted(fixed_params.items())]
        param_items.append(f"x-{safe_str(x_key)}_y-{safe_str(y_key)}")
        filename = f"data_surface_{'_'.join(param_items)}.pkl"
        if len(filename) > 200:
            filename = filename[:200] + ".pkl"
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"原始数据已保存为：{file_path}")

    fig = plt.figure(figsize=plot_params.get('figsize', (10, 8)))
    ax = fig.add_subplot(111, projection='3d')
    # fig, ax = plot_3d_surface(ax, x_vals, y_vals, Z, plot_params)  # 目前不开发3D绘图

    return fig, ax


def plot_Z(new_coords, Z_list, x_key, plot_params=None, y_key=None, fixed_params=None, show=False):
    plot_params = plot_params or {}
    fixed_params = fixed_params or {}
    if not isinstance(Z_list, list):
        Z_list = [Z_list]

    keys = list(new_coords.keys())
    assert x_key in keys, f"{x_key} 不在 new_coords 中"
    if y_key:
        assert y_key in keys and y_key != x_key, f"{y_key} 不在 new_coords 中或与 {x_key} 重复"
    for k in fixed_params:
        assert k in keys and k not in (x_key, y_key or x_key), f"固定参数 {k} 无效"

    slicer = []
    for k in keys:
        if k == x_key or k == y_key:
            slicer.append(slice(None))
        else:
            val = fixed_params.get(k)
            assert val is not None, f"参数 {k} 未在 fixed_params 中指定"
            idx = np.where(new_coords[k] == val)[0]
            assert idx.size == 1, f"{k} 中未找到值 {val}"
            slicer.append(idx[0])
    slicer = tuple(slicer)

    subs = [Z[slicer] for Z in Z_list]
    x_vals = new_coords[x_key]
    y_vals = new_coords.get(y_key, None) if y_key else None

    if y_vals is None or plot_params.get('dim', 2) == 2:
        fig, ax = plot_Z_2D(subs, x_vals, x_key, y_vals, y_key, plot_params, fixed_params, is_1d=(y_key is None))
    else:
        fig, ax = plot_Z_3D(subs, x_vals, x_key, y_vals, y_key, plot_params, fixed_params)

    fig, ax = add_annotations(ax, plot_params)

    save_dir = plot_params.get('save_dir', './rsl/2D/')
    full_params = {**fixed_params, **plot_params}
    param_items = [f"{k}-{safe_str(v)}" for k, v in sorted(full_params.items())]
    filename = "plot_" + "_".join(param_items) + ".svg"
    if len(filename) > 200:
        filename = filename[:200] + ".svg"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + filename, dpi=300, bbox_inches="tight", transparent=True)
    print(f"图像已保存为：{save_dir + filename}")

    if show:
        plt.show()
    return fig, ax
