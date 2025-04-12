import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 绘图支持

def direct_plot_3D(
        twoD_data: np.ndarray,
        save_name: str,
        plot_paras: dict = None,
        show: bool = False,
        **kwargs
):
    """
    根据二维数组数据绘制3D柱状高度图。

    参数：
        twoD_data (np.ndarray): 二维数组数据，每个元素代表对应位置柱状图的高度。
        save_name (str): 保存图像的基础文件名（不包含扩展名）。
        plot_paras (dict, optional): 自定义图像外观的参数字典，可包括：
            - 'font_size'     : 全局字体大小，默认12。
            - 'colormap'      : 颜色映射，如 'viridis'、'jet'，默认 'viridis'。
            - 'xlabel', 'ylabel', 'zlabel': 各坐标轴标签。
            - 'xticks', 'yticks', 'zticks': 各坐标轴刻度列表。
            - 'xtickslabels', 'ytickslabels', 'ztickslabels': 各坐标轴刻度标签。
            - 'xlim', 'ylim', 'zlim'        : 各坐标轴显示范围。
            - 'view_angle'    : 视角设置，如 (elev, azim)。
            - 'box_aspect'    : 三维图形的长宽高比，例如 (1, 1, 0.5)。
            - 'raw_yindex'    : 针对原始二维数组在 y 方向选区，格式为 (y_start, y_end)。
            - 'log_scale'     : 是否对高度数据进行对数缩放（布尔值），默认 False。
            - 'vmin', 'vmax'  : 用于颜色归一化的最小和最大值，默认为数据自身的极值。
            - 其它可用于自定义的参数。
        show (bool, optional): 是否显示图像。默认 False。
        **kwargs: 传递给 Axes3D.bar3d 的其他参数。

    效果：
        - 选取二维数据部分（通过 'raw_yindex' 参数，可选）。
        - 如设置 'log_scale'=True，则对数据做对数缩放。
        - 根据传入参数为柱状图着色、设置坐标轴及视角。
        - 同时将结果以 PNG 和 SVG 格式保存至指定目录（文件名前缀为 './rsl/3D-'）。
    """
    # 设置全局字体大小
    if plot_paras:
        plt.rcParams['font.size'] = plot_paras.get('font_size', 12)
    else:
        plt.rcParams['font.size'] = 12

    # 获取二维数据的形状
    height, width = twoD_data.shape

    # 设置默认 colormap
    cmap_name = plot_paras.get('colormap', 'viridis') if plot_paras else 'viridis'
    cmap = plt.get_cmap(cmap_name)

    # 设置 figure 尺寸，固定 dpi 值
    dpi = 100
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 柱子宽度和深度
    dx = 0.8
    dy = 0.8

    selected_data = twoD_data

    nrows, ncols = selected_data.shape
    # 构造柱状图的网格坐标，并将柱子中心调整
    xx, yy = np.meshgrid(np.arange(ncols), np.arange(nrows))
    x_coords = xx.flatten() - dx / 2
    y_coords = yy.flatten() - dy / 2
    z_coords = np.zeros_like(x_coords)

    # 获取柱子高度数据
    dz = selected_data.flatten()
    if plot_paras and plot_paras.get('log_scale', False):
        # 避免零值问题，乘以常数后取对数
        dz = np.log10(dz * 100 + 1e-8)

    # 颜色归一化与映射
    vmin = plot_paras.get('vmin', dz.min()) if plot_paras else dz.min()
    vmax = plot_paras.get('vmax', dz.max()) if plot_paras else dz.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(dz))

    # 绘制3D柱状图
    ax.bar3d(x_coords, y_coords, z_coords, dx, dy, dz,
             color=colors, shade=True, **kwargs)

    # 坐标轴与视角设置
    if plot_paras:
        if 'xlabel' in plot_paras:
            ax.set_xlabel(plot_paras['xlabel'])
        if 'ylabel' in plot_paras:
            ax.set_ylabel(plot_paras['ylabel'])
        if 'zlabel' in plot_paras:
            ax.set_zlabel(plot_paras['zlabel'])
        if 'xticks' in plot_paras:
            ax.set_xticks(plot_paras['xticks'])
        if 'yticks' in plot_paras:
            ax.set_yticks(plot_paras['yticks'])
        if 'zticks' in plot_paras:
            ax.set_zticks(plot_paras['zticks'])
        if 'xtickslabels' in plot_paras:
            ax.set_xticklabels(plot_paras['xtickslabels'])
        if 'ytickslabels' in plot_paras:
            ax.set_yticklabels(plot_paras['ytickslabels'])
        if 'ztickslabels' in plot_paras:
            ax.set_zticklabels(plot_paras['ztickslabels'])
        if 'xlim' in plot_paras:
            ax.set_xlim(plot_paras['xlim'])
        if 'ylim' in plot_paras:
            ax.set_ylim(plot_paras['ylim'])
        if 'zlim' in plot_paras:
            ax.set_zlim(plot_paras['zlim'])
        if 'view_angle' in plot_paras:
            elev, azim = plot_paras['view_angle']
            ax.view_init(elev=elev, azim=azim)
        if 'z-a.u' in plot_paras:
            ax.set_zticks([dz.min(), dz.max()])
            ax.set_zticklabels([0, 1])
        if 'box_aspect' in plot_paras:
            ax.set_box_aspect(plot_paras['box_aspect'])

    # 隐藏背景网格
    ax.grid(False)

    # 构造保存文件名，将参数字典转为字符串（去除特殊字符）
    dict_str = json.dumps(plot_paras, separators=(',', ':')) if plot_paras else ""
    if dict_str:
        dict_str = dict_str.replace('{', '').replace('}', '').replace('"', '').replace(' ', '_').replace(':', '_')
        save_path = f'./rsl/3D-{save_name}+{dict_str}'
    else:
        save_path = f'./rsl/{save_name}'
    plt.savefig(save_path+'.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.savefig(save_path+'.svg', bbox_inches='tight', pad_inches=0, transparent=True)

    if show:
        plt.show()
    plt.close()

def direct_plot_aggregated_bar(
        twoD_data: np.ndarray,
        save_name: str,
        aggregate_axis: str = 'x',
        plot_paras: dict = None,
        show: bool = False
):
    """
    根据二维数组数据，在 x 或 y 方向聚合计算柱子的高度总和比例，并绘制二维柱状图。

    参数：
        twoD_data (np.ndarray): 二维数组数据。
        save_name (str): 保存图像的基础文件名（不包含扩展名）。
        aggregate_axis (str, optional): 聚合方向，'x' 表示沿列求和（各列的贡献比例），
                                        'y' 表示沿行求和（各行的贡献比例）。默认 'x'。
        plot_paras (dict, optional): 自定义图像的参数字典，可包括：
            - 'font_size'    : 字体大小。
            - 'xlabel'       : x 轴标签。
            - 'ylabel'       : y 轴标签。
            - 'title'        : 图形标题。
            - 其它 matplotlib 的常用参数。
        show (bool, optional): 是否显示图像，默认为 False。

    效果：
        - 沿选定方向聚合二维数据，计算每个条目的总和占所有数据和的比例（单位为 1）。
        - 绘制一个二维柱状图展示各聚合条目的比例，并保存为 PNG 和 SVG 格式。
    """
    # 设置全局字体大小
    if plot_paras:
        plt.rcParams['font.size'] = plot_paras.get('font_size', 12)
    else:
        plt.rcParams['font.size'] = 12

    # 根据聚合方向计算沿 x 或 y 方向的和
    if aggregate_axis.lower() == 'x':
        # 沿列求和，返回每一列的总和
        aggregated = np.sum(twoD_data, axis=0)
        x_ticks = np.arange(twoD_data.shape[1])[::-1]
        xlabel = plot_paras.get('xlabel', None)
    elif aggregate_axis.lower() == 'y':
        # 沿行求和，返回每一行的总和
        aggregated = np.sum(twoD_data, axis=1)
        x_ticks = np.arange(twoD_data.shape[0])[::-1]
        xlabel = plot_paras.get('xlabel', None)
    else:
        raise ValueError("aggregate_axis 必须为 'x' 或 'y'")

    total_sum = np.sum(aggregated)
    # 计算比例，单位为1
    proportions = aggregated / total_sum

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x_ticks, proportions, color=plot_paras.get('bar_color', 'skyblue') if plot_paras else 'skyblue')

    # 设置坐标轴标签与标题
    ax.set_xlabel(xlabel)
    ax.set_ylabel(plot_paras.get('ylabel', None))
    if plot_paras and 'title' in plot_paras:
        ax.set_title(plot_paras['title'])

    # 设定 x 轴刻度
    ax.set_xticks(x_ticks[::2])
    ax.set_ylim(plot_paras.get('zlim', [0, 1]))

    if plot_paras.get('mark_value', None):
        # 将比例标记在柱子上
        for i, prop in enumerate(proportions):
            ax.text(i, prop + 0.005, f'{prop:.2f}', ha='center', va='bottom')

    ax.set_box_aspect(plot_paras.get('box_aspect', 1))

    # 保存图像
    dict_str = json.dumps(plot_paras, separators=(',', ':')) if plot_paras else ""
    if dict_str:
        dict_str = dict_str.replace('{', '').replace('}', '').replace('"', '').replace(' ', '_').replace(':', '_')
        save_path = f'./rsl/2D-{save_name}+{dict_str}'
    else:
        save_path = f'./rsl/{save_name}'
    plt.savefig(save_path+'.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.savefig(save_path+'.svg', bbox_inches='tight', pad_inches=0, transparent=True)

    if show:
        plt.show()
    plt.close()
