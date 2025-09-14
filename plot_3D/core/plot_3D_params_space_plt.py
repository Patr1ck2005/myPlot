import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import re
import os
from .utils import safe_str


import plotly.graph_objects as go

from plot_3D.advance_plot_styles.line_plot import plot_line_advanced

# fontsize
fs = 12
plt.rcParams.update({'font.size': fs})

# 注意: 20250914移除plotly的绘图模块

def plot_Z_2D(subs, x_vals, x_key, y_vals=None, y_key=None, plot_params=None, fixed_params=None, is_1d=True):
    """
    二维绘图函数：一维多曲线或二维热图。
    """
    plot_params = plot_params or {}
    fixed_params = fixed_params or {}
    figsize = plot_params.get('figsize', (8, 6))
    cmap1_name = plot_params.get('cmap', 'viridis')
    cmap2_name = plot_params.get('cmap2', 'plasma')
    default_color_list = plot_params.get('default_color_list', None)
    log_scale = plot_params.get('log_scale', False)
    xlabel = plot_params.get('xlabel', x_key)
    zlabel = plot_params.get('zlabel', "Δ")
    ylabel_title = plot_params.get('ylabel', "Δ")
    xlim = plot_params.get('xlim', None)
    ylim = plot_params.get('ylim', None)
    imshow_aspect = plot_params.get('imshow_aspect', 'auto')
    alpha_val = plot_params.get('alpha', 1.0)
    plot_imaginary = plot_params.get('imag', False)
    enable_line_fill = plot_params.get('enable_line_fill', True)
    title = plot_params.get('title', '')
    add_colorbar = plot_params.get('add_colorbar', False)
    enable_legend = plot_params.get('legend', False)
    advanced_process = plot_params.get('advanced_process', False)
    save_raw_data = plot_params.get('save_raw_data', True)

    # 创建保存目录
    save_dir = plot_params.get('save_dir', './rsl/2D/')
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 添加时间戳确保唯一性
    data_dir = os.path.join(save_dir, f"raw_data_{timestamp}")
    if save_raw_data:
        os.makedirs(data_dir, exist_ok=True)

    if advanced_process == 'y_mirror' and is_1d:
        # 将曲线关于y轴镜像对称复制处理（仅适用于一维）
        x_vals = np.concatenate([-x_vals[::-1], x_vals])
        subs = [np.concatenate([sub[::-1], sub]) for sub in subs]

    fig, ax = plt.subplots(figsize=figsize)
    if is_1d:
        y_mins, y_maxs = [], []
        for i, sub in enumerate(subs):
            y_vals = sub
            mask = np.isnan(y_vals)
            if np.any(mask):
                print(f"Warning: 存在非法数据，已移除。")
                y_vals = y_vals[~mask]
                temp_x_vals = x_vals[~mask]
            else:
                temp_x_vals = x_vals

            if default_color_list is not None:
                plot_params['default_color'] = default_color_list[i % len(default_color_list)]

            ax = plot_line_advanced(ax, temp_x_vals, z1=y_vals.real, z2=y_vals.imag, z3=y_vals.imag, **plot_params)

            # 保存原始数据（一维）
            if save_raw_data:
                raw_data_dict = {
                    'x': temp_x_vals,
                    'y_real': y_vals.real,
                    'y_imag': y_vals.imag if np.iscomplexobj(y_vals) else np.zeros_like(y_vals.real)
                }
                param_items = [f"{k}-{safe_str(v)}" for k, v in sorted(fixed_params.items())]
                param_items.append(f"curve-{i+1}")  # 区分多条曲线
                filename = f"data_{'_'.join(param_items)}.csv"
                if len(filename) > 200:
                    filename = filename[:200] + ".csv"
                file_path = os.path.join(data_dir, filename)
                df = pd.DataFrame(raw_data_dict)
                df.to_csv(file_path, index=False)
                print(f"原始数据已保存为：{file_path}")

            # 简化轴限计算
            if enable_line_fill:
                widths = np.abs(y_vals.imag)
                scale = plot_params.get('scale', 0.5)
                y_mins.append(np.min(y_vals.real - scale * widths))
                y_maxs.append(np.max(y_vals.real + scale * widths))
            else:
                y_mins.append(np.min(y_vals.real))
                y_maxs.append(np.max(y_vals.real))

        # 设置轴限、标签等
        ax.set_xlim(x_vals.min(), x_vals.max())
        ax.set_ylim(np.nanmin(y_mins)*0.98, np.nanmax(y_maxs)*1.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(zlabel)
        # ax.grid(True)
    else:
        # 二维热图：暂取第一个 sub（不支持多 Z）
        if len(subs) > 1:
            print("Warning: 多 Z 在二维模式下仅使用第一个 Z。")
        sub = subs[0]
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')  # 注意顺序
        Z_plot = sub  # 转置匹配 shape

        Z_real_plot = Z_plot.real
        Z_imag_plot = Z_plot.imag

        # 绘制热图
        surf1 = ax.imshow(
            Z_real_plot.T,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect=imshow_aspect,
            cmap=cmap1_name, alpha=alpha_val,
            interpolation='none'
        )
        if sub.dtype == np.complex128 and plot_imaginary:
            surf2 = ax.pcolormesh(X, Y, Z_imag_plot, cmap=cmap2_name, alpha=alpha_val)
            # fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        # fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=20, pad=0.0, label=zlabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel_title)

        # 保存原始数据（二维，使用 .pkl 格式）
        if save_raw_data:
            # 构建字典
            data_dict = {
                'x_vals': x_vals,
                'y_vals': y_vals,
                'Z': Z_plot  # 直接保存二维数组
            }
            # 生成规范化文件名
            param_items = [f"{k}-{safe_str(v)}" for k, v in sorted(fixed_params.items())]
            param_items.append(f"x-{safe_str(x_key)}_y-{safe_str(y_key)}")
            filename = f"data_heatmap_{'_'.join(param_items)}.pkl"
            if len(filename) > 200:
                filename = filename[:200] + ".pkl"
            file_path = os.path.join(data_dir, filename)
            # 保存为 .pkl
            with open(file_path, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"原始数据已保存为：{file_path}")

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(f"{xlabel}({x_key}) (and {ylabel_title}({y_key})) vs {zlabel} @ {fixed_params}" + (" (Multiple Datas)" if len(subs) > 1 else ""))
    if enable_legend:
        ax.legend()
    if log_scale:
        ax.set_yscale('log')
    if add_colorbar:
        fig.colorbar(surf1, ax=ax, label=zlabel)
    return fig, ax


# plot_Z_3D 函数（扩展，非重点；类似原）
def plot_Z_3D(subs, x_vals, x_key, y_vals, y_key, plot_params=None, fixed_params=None):
    # 类似 plot_Z_2D 二维，但用 plot_surface
    # 暂取第一个 sub
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 示例：ax.plot_surface(...)  # 完整实现类似历史
    return fig, ax

def plot_Z(new_coords, Z_list, x_key, plot_params=None, y_key=None, fixed_params=None, show=False):
    """
    主绘图函数，支持多个 Z_list（list of ndarray 或单个 ndarray）。

    参数：
    - Z_list: np.ndarray 或 List[np.ndarray]，多个数据数组。
    - 其余同原。
    """
    plot_params = plot_params or {}
    fixed_params = fixed_params or {}
    if not isinstance(Z_list, list):
        Z_list = [Z_list]  # 单个转为列表

    # 参数检查（同原，扩展到每个 Z）
    keys = list(new_coords.keys())
    assert x_key in keys, f"{x_key} 不在 new_coords 中"
    if y_key:
        assert y_key in keys and y_key != x_key, f"{y_key} 不在 new_coords 中或与 {x_key} 重复"
    for k in fixed_params:
        assert k in keys and k not in (x_key, y_key or x_key), f"固定参数 {k} 无效"

    # 构建 slicer（同原）
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

    # 提取 subs
    subs = [Z[slicer] for Z in Z_list]
    # 验证每个 sub 维度一致（示例：假设所有 Z 维度相同）

    x_vals = new_coords[x_key]
    y_vals = new_coords.get(y_key, None) if y_key else None

    # 调用子函数
    if y_vals is None or plot_params.get('dim', 2) == 2:  # 默认二维
        fig, ax = plot_Z_2D(subs, x_vals, x_key, y_vals, y_key, plot_params, fixed_params, is_1d=(y_key is None))
    else:
        fig, ax = plot_Z_3D(subs, x_vals, x_key, y_vals, y_key, plot_params, fixed_params)

    plt.tight_layout()

    # 保存文件（同原）
    save_dir = plot_params.get('save_dir', './rsl/2D/')
    def safe_str(val):
        return re.sub(r'[^\w.-]', '', str(val))
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
