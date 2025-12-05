import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker
from matplotlib.projections import PolarAxes
from mpl_toolkits.mplot3d import Axes3D
import re
import os
from .utils import safe_str
from advance_plot_styles.line_plot import plot_line_advanced
from advance_plot_styles.scatter_plot import plot_scatter_advanced

# # fontsize
# fs = 12
# plt.rcParams.update({'font.size': fs})


def add_annotations(ax, plot_params):
    """
    可选标注函数：添加标题、标签、图例、轴控制和定制化 ticks。
    默认隐藏所有轴标签 (xlabel, ylabel, title) 和 tick labels，只有手动启用时显示。
    完善了对极坐标 (PolarAxes) 的支持，包括 theta/r 映射、ticks、labels 隐藏等。

    参数:
        ax (matplotlib.axes.Axes): 当前的Axes对象。

        # 通用标注参数
        title (str, optional): 图表标题。只有 show_axis_labels=True 时显示。
        xlabel (str, optional): X轴/theta轴标签。只有 show_axis_labels=True 时显示。默认 None。
        ylabel (str, optional): Y轴/r轴标签。只有 show_axis_labels=True 时显示。默认 None。
        zlabel (str, optional): Z轴标签（2D 未用）。默认 "Z"。
        xlim (tuple, optional): X/theta轴显示范围 (min, max)。
        ylim (tuple, optional): Y/r轴显示范围 (min, max)。
        zlim (tuple, optional): Z轴显示范围 (min, max)。默认 None。
        x_log_scale (bool, optional): X/theta轴对数尺度。默认 False（极坐标 theta 不常用）。
        y_log_scale (bool, optional): Y/r轴对数尺度。默认 False。
        show_legend (bool, optional): 是否显示图例。默认 False。
        legend_loc (str, optional): 图例位置，例如 'best', 'upper left'。默认 'best'。
        add_grid (bool, optional): 是否添加网格。默认 False。
        grid_style (str, optional): 网格线样式，例如 '--'。默认 '--'。
        grid_alpha (float, optional): 网格线透明度。默认 0.3。

        # 显示控制参数（新添加）
        show_axis_labels (bool, optional): 是否显示轴标签 (title, xlabel, ylabel)。默认 False（隐藏）。
        show_tick_labels (bool, optional): 是否显示 tick labels（所有轴的刻度标签）。默认 False（隐藏）。
        show_ticks (bool, optional): 是否显示 ticks（所有轴的刻度）。默认 True（显示）。

        # 极坐标专用参数（新添加/扩展）
        rlabel_position (int or float, optional): r轴标签显示位置（度，-1 隐藏）。仅极坐标有效，默认 -1（隐藏）。

        # X轴/theta轴 ticks 定制化参数
        xtick_mode (str, optional): X/theta轴 ticks 的模式。可选值：
                                   'auto' (默认): Matplotlib 自动确定。
                                   'approx_count': 尝试指定 ticks 的大致数量 (通过 xtick_count)。
                                   'manual': 完全手动指定 ticks 位置 (通过 xticks)。
        xtick_count (int, optional): 当 xtick_mode='approx_count' 时，指定 X/theta轴 ticks 的大致数量。默认 None。
        xticks (list or numpy.array, optional): 当 xtick_mode='manual' 时，X/theta轴 ticks 的具体位置。默认 None。
        xtick_labels (list of str, optional): 当 xtick_mode='manual' 时，X/theta轴 ticks 对应的标签。默认 None。

        # Y轴/r轴 ticks 定制化参数
        ytick_mode (str, optional): Y/r轴 ticks 的模式。可选值：
                                    'auto' (默认): Matplotlib 自动确定。
                                    'approx_count': 尝试指定 ticks 的大致数量 (通过 ytick_count)。
                                    'manual': 完全手动指定 ticks 位置 (通过 yticks)。
        ytick_count (int, optional): 当 ytick_mode='approx_count' 时，指定 Y/r轴 ticks 的大致数量。默认 None。
        yticks (list or numpy.array, optional): 当 ytick_mode='manual' 时，Y/r轴 ticks 的具体位置。默认 None。
        ytick_labels (list of str, optional): 当 ytick_mode='manual' 时，Y/r轴 ticks 对应的标签（r轴暂不支持自定义标签）。默认 None。

        # 布局参数
        enable_tight_layout (bool, optional): 是否启用紧凑布局。默认 False。

    返回: (fig, ax) - 加工后的 figure 和 ax。
    """
    plt.savefig('temp_default_ticks.png', dpi=300, bbox_inches='tight')
    print("Temp figure with default ticks saved as 'temp_default_ticks.png'.")
    # 从 plot_params 获取参数，默认值调整为隐藏
    title = plot_params.get('title', None)
    xlabel = plot_params.get('xlabel', None)  # 默认 None，不设置
    ylabel = plot_params.get('ylabel', None)  # 默认 None，不设置
    zlabel = plot_params.get('zlabel', None)  # 默认 None
    xlim = plot_params.get('xlim', None)
    ylim = plot_params.get('ylim', None)
    zlim = plot_params.get('zlim', None)
    x_log_scale = plot_params.get('x_log_scale', False)
    y_log_scale = plot_params.get('y_log_scale', False)
    show_legend = plot_params.get('show_legend', False)
    legend_loc = plot_params.get('legend_loc', 'best')
    add_grid = plot_params.get('add_grid', False)
    grid_style = plot_params.get('grid_style', '-')
    grid_alpha = plot_params.get('grid_alpha', 0.3)

    # 新参数：显示控制
    show_axis_labels = plot_params.get('show_axis_labels', False)
    show_tick_labels = plot_params.get('show_tick_labels', False)
    show_ticks = plot_params.get('show_ticks', True)

    # 极坐标专用
    rlabel_position = plot_params.get('rlabel_position', -1)  # -1 隐藏 r 标签

    # Ticks 参数
    xtick_mode = plot_params.get('xtick_mode', 'auto')
    xtick_count = plot_params.get('xtick_count', None)
    xticks = plot_params.get('xticks', None)
    xtick_labels = plot_params.get('xtick_labels', None)

    ytick_mode = plot_params.get('ytick_mode', 'auto')
    ytick_count = plot_params.get('ytick_count', None)
    yticks = plot_params.get('yticks', None)
    ytick_labels = plot_params.get('ytick_labels', None)  # r 轴暂不支持自定义标签

    enable_tight_layout = plot_params.get('enable_tight_layout', False)

    # 判断是否为极坐标
    is_polar = isinstance(ax, PolarAxes)

    # 1. 设置标题（仅当 show_axis_labels=True 且 title 提供时）
    if show_axis_labels and title:
        ax.set_title(title)

    # 2. 设置轴标签（仅当 show_axis_labels=True 且 提供时）
    if show_axis_labels:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if zlabel is not None and not is_polar:  # 2D 无 z
            ax.set_zlabel(zlabel)
    else:
        # 默认隐藏：设置为空（视觉上隐藏）
        ax.set_xlabel('')
        ax.set_ylabel('')
        # if not is_polar:
        #     ax.set_zlabel('')

    # 3. 处理极坐标范围（修正为 set_thetalim/set_rlim）
    if is_polar:
        if xlim:
            ax.set_thetalim(xlim[0], xlim[1])  # theta 范围（弧度）
        if ylim:
            ax.set_rlim(ylim[0], ylim[1])  # r 范围
        # 极坐标 r 标签位置
        ax.set_rlabel_position(rlabel_position)
    elif isinstance(ax, Axes3D):
        print("Warning: Z 轴范围设置在此函数中未实现。请在绘图时单独设置。")
    else:
        # 笛卡尔范围
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if zlim:
            ax.set_zlim(zlim)

    # 4. 对数尺度
    if not is_polar:
        if x_log_scale:
            ax.set_xscale('log')
        if y_log_scale:
            ax.set_yscale('log')
    else:
        # 极坐标：仅 r 支持 log（y_log -> r_log），theta 不常用
        if y_log_scale:
            ax.set_rscale('log')

    # 5. 处理 X/theta 轴 ticks
    if xtick_mode == 'auto':
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
            print("Warning: xtick_mode is 'manual' but xticks are not provided. X/theta-axis ticks will be auto-set.")

    # 6. 处理 Y/r 轴 ticks（极坐标映射）
    if ytick_mode == 'auto':
        pass
    elif ytick_mode == 'approx_count':
        if ytick_count is not None:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ytick_count))
    elif ytick_mode == 'manual':
        if yticks is not None:
            if is_polar:
                ax.set_rticks(yticks)  # r ticks
                # 注意：r 轴无 set_rticklabels，标签自动从值生成；ytick_labels 暂忽略
                if ytick_labels is not None:
                    print(
                        "Warning: ytick_labels for r-axis in polar not directly supported (use formatter for custom).")
            else:
                ax.set_yticks(yticks)
                if ytick_labels is not None:
                    ax.set_yticklabels(ytick_labels)
        else:
            print("Warning: ytick_mode is 'manual' but yticks are not provided. Y/r-axis ticks will be auto-set.")

    # 7. 默认隐藏 tick labels（除非 show_tick_labels=True）
    if not show_tick_labels:
        # 通用：清空标签
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # 极坐标额外：隐藏 r/theata 视觉标签
        if is_polar:
            ax.tick_params(axis='x', labelbottom=False)  # theta labels
            ax.tick_params(axis='y', labelleft=False)  # r labels
        else:
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelleft=False)
    # 如果 show_tick_labels=True，标签已通过 manual/approx 设置显示
    # 额外：控制 ticks 显示与隐藏
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    # 8. 极坐标网格扩展（如果提供 ticks，设置 theta/r grids）
    if is_polar and add_grid:
        if xticks:  # theta grids
            ax.set_thetagrids(np.rad2deg(xticks))  # 转换为度
        if yticks:  # r grids
            ax.set_rgrids(yticks)

    # 9. 图例和网格
    if show_legend:
        ax.legend(loc=legend_loc)
    if add_grid:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)

    # 10. 紧凑布局
    if enable_tight_layout:
        plt.tight_layout()
    return ax.figure, ax  # 返回 fig, ax（修正为 ax.figure）


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

    imshow_aspect = plot_params.get('imshow_aspect', 'auto')
    plot_imaginary = plot_params.get('imag', False)
    add_colorbar = plot_params.get('add_colorbar', False)
    title_colorbar = plot_params.get('title_colorbar', '')

    global_color_vmax = plot_params.get('global_color_vmax', None)
    global_color_vmin = plot_params.get('global_color_vmin', None)

    Z_real_plot = Z.real
    Z_imag_plot = Z.imag



    # X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    # if global_color_vmax is not None and global_color_vmin is not None:
    surf1 = ax.imshow(
        Z_real_plot.T,
        # extent=[X.min(), X.max(), Y.min(), Y.max()],
        # 应该使用index来标定
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        origin='lower',
        aspect=imshow_aspect,
        cmap=cmap1_name,
        alpha=alpha_val,
        interpolation='none',
        vmin=global_color_vmin,
        vmax=global_color_vmax
    )

    if add_colorbar:
        ax.get_figure().colorbar(surf1, ax=ax, label=title_colorbar)

    return ax.get_figure(), ax


# 新绘图函数：二维多曲线
def plot_2d_multiline(ax, x_vals, y_vals, Z, plot_params):
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    cmap_name = plot_params.get('cmap', 'viridis')
    default_color = plot_params.get('default_color', None)
    default_color_list = plot_params.get('default_color_list', None)
    alpha_val = plot_params.get('alpha', 1.0)
    plot_imaginary = plot_params.get('imag', False)
    add_colorbar = plot_params.get('add_colorbar', False)
    global_color_vmin = plot_params.get('global_color_vmin', None)
    global_color_vmax = plot_params.get('global_color_vmax', None)


    cmap = cm.get_cmap(cmap_name)
    if global_color_vmin is None:
        global_color_vmin = y_vals.min()
    if global_color_vmax is None:
        global_color_vmax = y_vals.max()
    # norm = colors.Normalize(vmin=y_vals.min(), vmax=y_vals.max())
    norm = colors.Normalize(vmin=global_color_vmin, vmax=global_color_vmax)
    ny = len(y_vals)

    Z_real = Z.real
    Z_imag = Z.imag if np.iscomplexobj(Z) else np.zeros_like(Z.real)

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

    if add_colorbar and default_color is None and default_color_list is None:
        # 添加颜色条
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(y_vals)  # 设置颜色条的数据
        cbar = ax.get_figure().colorbar(sm, ax=ax)

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


def generate_save_name(save_dir, full_params):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(save_dir, f"{timestamp}")
    os.makedirs(plot_dir, exist_ok=True)

    param_items = [f"{k}-{safe_str(v)}" for k, v in sorted(full_params.items())]
    filename = "plot_" + "_".join(param_items) + ".svg"
    if len(filename) > 200:
        filename = filename[:200] + ".svg"
    image_path = os.path.join(plot_dir, filename)
    return image_path