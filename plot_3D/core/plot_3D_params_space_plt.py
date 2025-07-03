import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import os


import plotly.graph_objects as go

def plot_Z_diff_plotly(
        new_coords,
        Z,
        x_key,
        plot_params,
        y_key=None,
        fixed_params=None,
):
    """
    根据 new_coords 和 Z，指定一个或两个 key 作为坐标轴，其它 key 固定，
    使用 Plotly 绘制 Z 的曲线或曲面，并保存绘图文件（文件名中包含绘图参数）。

    参数：
        new_coords: dict，键为参数名，值为该参数的坐标数组（已排序）。
        Z: np.ndarray，多维数组，维度与 new_coords 顺序一致，存放差值数据。
        x_key: str，要作为横坐标的参数名。
        plot_params: dict，绘图时的一些参数，例如：
            - 'cmap1': 实部使用的 colorscale 名称（Plotly 内置 colorscale 名称，如 'Viridis'）
            - 'cmap2': 虚部使用的 colorscale 名称（例如 'Plasma'）
            - 'log_scale': 是否对数显示（True/False）
            - 'zlabel': z 轴标签文本
            - 'ylabel': 用于标题中的描述文字
            - 'alpha': 曲面透明度（0~1之间，一般取 0.5-0.7）
        y_key: str or None，要作为纵坐标的参数名；若为 None，则绘制一维折线图。
        fixed_params: dict，固定其它参数的取值，例如 {"a": 0.0085}。
                      固定参数的键应为 new_coords 中除 x_key, y_key 外的参数名。

    绘图文件将以 "plot_<参数信息>.png" 命名，并保存在当前工作目录。
    例如：
        # 一维折线图
        plot_Z_diff_plotly(new_coords, Z,
                           x_key="w1 (nm)",
                           plot_params={
                               'zlabel': "Δ频率 (kHz)",
                               'ylabel': "Δ频率实部 (kHz)",
                               'alpha': 1.0
                           },
                           fixed_params={"buffer (nm)": 1345.0, "h_grating (nm)": 113.5, "a": 0.0085})

        # 二维曲面
        plot_Z_diff_plotly(new_coords, Z,
                           x_key="w1 (nm)",
                           y_key="buffer (nm)",
                           plot_params={
                               'cmap1': 'Viridis',
                               'cmap2': 'Plasma',
                               'log_scale': False,
                               'zlabel': "Δ频率 (kHz)",
                               'alpha': 0.6
                           },
                           fixed_params={"h_grating (nm)": 113.5, "a": 0.0085})
    """
    # 1. 参数检查
    keys = list(new_coords.keys())
    if x_key not in keys:
        raise ValueError(f"{x_key} 不在 new_coords 中")
    if y_key:
        if (y_key not in keys) or (y_key == x_key):
            raise ValueError(f"{y_key} 不在 new_coords 中或与 {x_key} 重复")
    fixed_params = fixed_params or {}
    for k in fixed_params:
        if k not in keys or k in (x_key, y_key):
            raise ValueError(f"固定参数 {k} 无效")

    # 2. 构建切片索引：对每个维度，x_key 与 y_key 设 slice(None)，其它根据 fixed_params指定索引
    slicer = []
    for k in keys:
        if k == x_key or k == y_key:
            slicer.append(slice(None))
        else:
            val = fixed_params.get(k)
            if val is None:
                raise ValueError(f"参数 {k} 未在 fixed_params 中指定")
            idx = np.where(new_coords[k] == val)[0]
            if idx.size != 1:
                raise ValueError(f"{k} 中未找到唯一值 {val}")
            slicer.append(idx[0])
    slicer = tuple(slicer)

    # 3. 取出子数组及 x 轴数据
    sub = Z[slicer]
    x_vals = new_coords[x_key]

    # 提取绘图参数
    cmap1_name = plot_params.get('cmap1', 'Viridis')
    cmap2_name = plot_params.get('cmap2', 'Plasma')
    log_scale = plot_params.get('log_scale', False)
    zlabel = plot_params.get('zlabel', "Δ")
    ylabel_title = plot_params.get('ylabel', "Δ")
    alpha_val = plot_params.get('alpha', 1.0)

    # 数据缩放（根据需求调整，这里与 matplotlib 版本保持一致）
    scale = 1 / 0.001

    # 4. 根据是否有 y_key 分情况绘图
    if y_key is None:
        # 一维折线图
        y_vals = sub * scale  # sub shape 为 (len(x_vals),)
        fig = go.Figure()
        # 实部
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals.real, mode='lines',
                                 name='real', line=dict(color='blue')))
        # 如果是复数，则也绘制虚部
        if np.iscomplexobj(sub):
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals.imag, mode='lines',
                                     name='imag', line=dict(color='red', dash='dash')))
        fig.update_layout(
            title=f"{x_key} vs {ylabel_title} @ {fixed_params}",
            xaxis_title=x_key,
            yaxis_title=zlabel,
            template='plotly_white'
        )
    else:
        # 二维曲面图
        y_vals = new_coords[y_key]
        # 构造网格数据，注意保持和原来一致，sub 的 shape 应为 (len(x_vals), len(y_vals))
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        Z = sub.T * scale  # 这里注意转置以匹配网格

        # 若对数显示，则计算对数值（注意对于零值或负值可能需要额外处理，这里假设数据取绝对值后再对数）
        if log_scale:
            Z_real_plot = np.log10(np.abs(Z.real))
            Z_imag_plot = np.log10(np.abs(Z.imag))
        else:
            Z_real_plot = Z.real
            Z_imag_plot = Z.imag

        # 创建 3D 图形
        fig = go.Figure()

        # 实部曲面
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_real_plot, colorscale=cmap1_name,
                                 opacity=alpha_val, name='real', showscale=True,
                                 colorbar=dict(title=zlabel)),
        )
        # 如果数据为复数，添加虚部曲面
        if np.iscomplexobj(sub):
            fig.add_trace(go.Surface(x=X, y=Y, z=Z_imag_plot, colorscale=cmap2_name,
                                     opacity=alpha_val, name='imag', showscale=True,
                                     colorbar=dict(title=zlabel)))
        fig.update_layout(
            title=f"{x_key} vs {y_key} @ { {k: v for k, v in fixed_params.items()} }",
            scene=dict(
                xaxis_title=x_key,
                yaxis_title=y_key,
                zaxis_title=zlabel,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),  # 调整 x, y, z 的比例，决定视角
                    # 投影类型：'perspective'（透视）或 'orthographic'（正交）
                    projection=dict(type='orthographic')
                )
            ),
            template='plotly_white'
        )

    # 5. 自动生成文件名，并保存图像
    def safe_str(val):
        return re.sub(r'[^\w.-]', '', str(val))

    param_items = [f"{k}-{safe_str(v)}" for k, v in sorted(plot_params.items())]
    filename = "plot_" + "_".join(param_items) + ".png"
    if len(filename) > 200:
        filename = filename[:200] + ".png"
    # 保存图像为 PNG（需要安装 kaleido 库：pip install -U kaleido）
    fig.write_image(filename, scale=2)
    print(f"图像已保存为：{filename}")

    # 同时显示交互图形
    fig.show()


def plot_Z_diff_plt(
        new_coords,
        Z,
        x_key,
        plot_params,
        y_key=None,
        fixed_params=None,
        show=False,
):
    """
    根据 new_coords 和 Z，指定一个或两个 key 作为坐标轴，其它 key 固定，绘制 Z 的曲线或曲面，
    并保存绘图文件（文件名中包含绘图参数）。

    参数：
        new_coords: dict，键为参数名，值为该参数的坐标数组（已排序）。
        Z: np.ndarray，多维数组，维度与 new_coords 顺序一致，存放差值数据。
        x_key: str，要作为横坐标的参数名。
        plot_params: dict，绘图时的一些参数，例如：
            - 'cmap1': 实部使用的 colormap 名称
            - 'cmap2': 虚部使用的 colormap 名称
            - 'log_scale': 是否对数显示（True/False）
            - 'zlabel': z 轴标签文本
            - 'ylabel': 用于标题中的描述文字
            - 'alpha': 曲面透明度（0~1之间，一般取 0.5-0.7）
        y_key: str or None，要作为纵坐标的参数名；若为 None，则绘制一维曲线图。
        fixed_params: dict，固定其它参数的取值，例如 {"a": 0.0085}。
                      固定参数的键应为 new_coords 中除 x_key, y_key 外的参数名。
        show: bool, 决定是否显示交互式绘图窗口

    绘图文件将以 "plot_<参数信息>.png" 命名，并保存在当前工作目录。
    例如：
        # 一维曲线
        plot_Z_diff(new_coords, Z,
                    x_key="w1 (nm)",
                    plot_params={
                        'zlabel': "Δ频率 (kHz)",
                        'ylabel': "Δ频率实部 (kHz)",
                        'alpha': 1.0
                    },
                    fixed_params={"buffer (nm)": 1345.0, "h_grating (nm)": 113.5, "a": 0.0085})

        # 二维曲面
        plot_Z_diff(new_coords, Z,
                    x_key="w1 (nm)",
                    y_key="buffer (nm)",
                    plot_params={
                        'cmap1': 'viridis',
                        'cmap2': 'plasma',
                        'log_scale': False,
                        'zlabel': "Δ频率 (kHz)",
                        'alpha': 0.6
                    },
                    fixed_params={"h_grating (nm)": 113.5, "a": 0.0085})
    """
    # 1. 参数检查
    keys = list(new_coords.keys())
    assert x_key in keys, f"{x_key} 不在 new_coords 中"
    if y_key:
        assert y_key in keys and y_key != x_key, f"{y_key} 不在 new_coords 中或与 {x_key} 重复"
    fixed_params = fixed_params or {}
    for k in fixed_params:
        assert k in keys and k not in (x_key, y_key), f"固定参数 {k} 无效"

    # 2. 构建切片索引：对每个维度确定要选哪一组索引
    #    对 x_key, y_key 设为 slice(None)，其它根据 fixed_params 找到具体索引
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

    # 3. 取出子数组
    sub = Z[slicer]
    x_vals = new_coords[x_key]

    # 提取绘图参数
    cmap1_name = plot_params.get('cmap1', 'viridis')
    cmap2_name = plot_params.get('cmap2', 'plasma')
    log_scale = plot_params.get('log_scale', False)
    zlabel = plot_params.get('zlabel', "Δ")
    ylabel_title = plot_params.get('ylabel', "Δ")
    alpha_val = plot_params.get('alpha', 1.0)  # 半透明参数
    plot_imaginary = plot_params.get('imag', True)  # 半透明参数

    # 4. 绘图
    if y_key is None:
        # 一维曲线图
        y_vals = sub/0.001  # 对数据进行缩放（根据实际需要调整）
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals.real, label='real', color='blue')
        if sub.dtype == np.dtype(complex) and plot_imaginary:
            plt.plot(x_vals, y_vals.imag, label='imag', color='red', linestyle='--')
        plt.xlabel(x_key)
        plt.ylabel(zlabel)
        plt.title(f"{x_key} vs {ylabel_title} @ {fixed_params}")
        plt.grid(True)
        plt.legend()
        if log_scale:
            plt.yscale('log')
        fig = plt.gcf()
    elif True:
        # 二维曲面
        y_vals = new_coords[y_key]
        X, Y = np.meshgrid(y_vals, x_vals, indexing='ij')
        # 注意：sub 的 shape 应为 (len(x_vals), len(y_vals))
        Z = sub.T / 0.001  # 将数据进行缩放
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # 根据 log_scale 选择绘图数据
        if log_scale:
            Z_real_plot = np.log10(np.abs(Z.real))
            Z_imag_plot = np.log10(np.abs(Z.imag))
        else:
            Z_real_plot = Z.real
            Z_imag_plot = Z.imag

        # 绘制实部曲面
        surf1 = ax.pcolormesh(X, Y, Z_real_plot, cmap=cmap1_name,
                                alpha=alpha_val)
        # 绘制虚部曲面；为了保证交互性，先绘制一部分，再次绘制交叠区域
        if sub.dtype == np.dtype(complex) and plot_imaginary:
            surf2 = ax.pcolormesh(X, Y, Z_imag_plot, cmap=cmap2_name,
                                    alpha=alpha_val)
            fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(f"{x_key} vs {y_key} @ { {k: v for k, v in fixed_params.items()} }")
        fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=20, pad=0.0, label=zlabel)
    else:
        # 二维曲面
        y_vals = new_coords[y_key]
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        # 注意：sub 的 shape 应为 (len(x_vals), len(y_vals))
        Z = sub.T/0.001  # 将数据进行缩放
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 根据 log_scale 选择绘图数据
        if log_scale:
            Z_real_plot = np.log10(np.abs(Z.real))
            Z_imag_plot = np.log10(np.abs(Z.imag))
        else:
            Z_real_plot = Z.real
            Z_imag_plot = Z.imag

        # 绘制实部曲面
        surf1 = ax.plot_surface(X, Y, Z_real_plot, cmap=cmap1_name,
                                alpha=alpha_val, edgecolor='none')
        # 绘制虚部曲面；为了保证交互性，先绘制一部分，再次绘制交叠区域
        if sub.dtype == np.dtype(complex) and plot_imaginary:
            surf2 = ax.plot_surface(X, Y, Z_imag_plot, cmap=cmap2_name,
                                    alpha=alpha_val, edgecolor='none')
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_zlabel(zlabel)
        ax.set_title(f"{x_key} vs {y_key} @ { {k: v for k, v in fixed_params.items()} }")
        fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=20, pad=0.0)
        fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    plt.tight_layout()

    # 5. 自动生成文件名，并保存图像
    # 文件名将基于 plot_params 中的键值（为了简单，只使用部分关键信息）
    def safe_str(val):
        return re.sub(r'[^\w.-]', '', str(val))
    full_params = {**fixed_params, **plot_params}
    param_items = [f"{k}-{safe_str(v)}" for k, v in sorted(full_params.items())]
    filename = "plot_" + "_".join(param_items) + ".png"
    # 如果文件名过长，可以截断或采用 hash 值，这里仅作简单处理
    if len(filename) > 200:
        filename = filename[:200] + ".png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"图像已保存为：{filename}")
    if show:
        plt.show()
