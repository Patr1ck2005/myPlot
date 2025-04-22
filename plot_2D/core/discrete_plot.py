import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
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

    log_dz = np.log10(dz)
    if plot_paras.get('log_color', False):
        color_dz = log_dz
    else:
        color_dz = dz
    # 颜色归一化与映射
    vmin = plot_paras.get('vmin', color_dz.min()) if plot_paras else color_dz.min()
    vmax = plot_paras.get('vmax', color_dz.max()) if plot_paras else color_dz.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(color_dz))

    # 绘制3D柱状图
    ax.bar3d(x_coords, y_coords, z_coords, dx, dy, dz,
             color=colors, shade=True, **kwargs)

    ax.grid(plot_paras.get('grid', False))

    # 坐标轴与视角设置
    if plot_paras:
        if 'z-a.u' in plot_paras:
            z_ticks = np.linspace(dz.min(), dz.max(), 3, endpoint=True)
            z_ticklabels = np.linspace(0, 1, 3, endpoint=True)
            ax.set_zticks(z_ticks)
            ax.set_zticklabels(z_ticklabels)
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
        if 'box_aspect' in plot_paras:
            ax.set_box_aspect(plot_paras['box_aspect'])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # 构造保存文件名，将参数字典转为字符串（去除特殊字符）
    dict_str = json.dumps(plot_paras, separators=(',', ':')) if plot_paras else ""
    if dict_str:
        dict_str = dict_str.replace('{', '').replace('}', '').replace('"', '').replace(' ', '_').replace(':', '_')
        save_path = f'./rsl/3D-{save_name}+{dict_str}'
    else:
        save_path = f'./rsl/{save_name}'
    plt.savefig(save_path[:100]+'.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.savefig(save_path[:100]+'.svg', bbox_inches='tight', pad_inches=0, transparent=True)

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
    根据二维数组数据，在 x 或 y 方向聚合计算柱子的高度总和比例，并绘制带渐变色效果的二维柱状图。

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
            - 'bar_color'    : 边框颜色（可选）。
            - 'colormap'         : 渐变使用的颜色映射，默认 'viridis'。
            - 'mark_value'   : 是否在柱子上标记数值（布尔）。
            - 'num_xticks'   : 当未提供 xticks 时，自动均分生成的个数。
            - 'zlim'         : y 轴显示范围，默认 [0,1]（比例值）。
            - 'box_aspect'   : 图形的长宽比例。
            - 其它 matplotlib 的常用参数。
        show (bool, optional): 是否显示图像，默认为 False。

    效果：
        - 沿选定方向聚合二维数据，计算各条目比例（归一化后单位为 1）。
        - 对每个柱使用 imshow 绘制渐变色（颜色由 cmap 指定，按照柱高度进行渐变）。
        - 保存图像为 PNG 和 SVG 格式。
    """
    # 设置全局字体大小
    if plot_paras:
        plt.rcParams['font.size'] = plot_paras.get('font_size', 12)
    else:
        plt.rcParams['font.size'] = 12

    # 根据聚合方向计算和，并确定 x 轴刻度顺序（这里采用倒序显示，可根据需求调整）
    if aggregate_axis.lower() == 'x':
        aggregated = np.sum(twoD_data, axis=0)
        x_ticks = np.arange(twoD_data.shape[1])[::-1]
        xlabel = plot_paras.get('xlabel', 'X Axis (columns)')
    elif aggregate_axis.lower() == 'y':
        aggregated = np.sum(twoD_data, axis=1)
        x_ticks = np.arange(twoD_data.shape[0])[::-1]
        xlabel = plot_paras.get('xlabel', 'Y Axis (rows)')
    else:
        raise ValueError("aggregate_axis 必须为 'x' 或 'y'")

    total_sum = np.sum(aggregated)
    # 计算比例，确保范围在 [0,1]
    proportions = aggregated / total_sum

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # 取渐变 cmap，默认为 'viridis'
    cmap_name = plot_paras.get('colormap', 'viridis') if plot_paras else 'viridis'
    cmap = plt.get_cmap(cmap_name)

    # 柱宽设定（此处设为 0.8，可根据需要调整）
    bar_width = 0.8
    print(max(proportions))
    # 对于每个柱，使用 imshow 绘制垂直方向的渐变
    for i, prop in enumerate(proportions):
        # 每个柱的中心位置
        x_center = x_ticks[i]
        # x轴范围
        left = x_center - bar_width / 2
        right = x_center + bar_width / 2
        # y轴下沿设为 0，上沿为 prop
        bottom = 0
        top = prop

        # 生成 256×1 的渐变数据，值从 0 到 prop
        gradient = np.linspace(0, prop, 256).reshape(256, 1)
        # 使用 imshow 绘制渐变，并使其填充该柱的区域
        ax.imshow(gradient,
                  aspect='auto',
                  cmap=cmap,
                  extent=[left, right, bottom, top],
                  origin='lower',
                  vmin=plot_paras.get('vmin'),
                  vmax=plot_paras.get('vmax'),
                  )

        # 如果需要，也可添加边框矩形
        if plot_paras.get('bulk_bar', None):
            rect = patches.Rectangle((left, bottom), bar_width, prop,
                                     fill=False,
                                     edgecolor=plot_paras.get('bar_color', 'k'),
                                     linewidth=1)
            ax.add_patch(rect)

        # 如果需要标记数值
        if plot_paras.get('mark_value', False):
            ax.text(x_center, top + 0.005, f'{prop:.2f}',
                    ha='center', va='bottom')

    # 设置坐标轴标签与标题
    ax.set_xlabel(xlabel)
    ax.set_ylabel(plot_paras.get('ylabel', 'Proportion') if plot_paras else 'Proportion')
    if plot_paras and 'title' in plot_paras:
        ax.set_title(plot_paras['title'])
    else:
        ax.set_title(f'Aggregated Proportions along {aggregate_axis.upper()} axis')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(plot_paras.get('xticklabels', []))
    ax.set_yticklabels(plot_paras.get('xyticklabels', []))

    # y轴范围默认为 [0,1]或 plot_paras 中指定
    ax.set_ylim(plot_paras.get('zlim', [0, 1]))

    ax.set_box_aspect(plot_paras.get('box_aspect', 1))

    # 构造保存文件名，将参数字典转为字符串
    dict_str = json.dumps(plot_paras, separators=(',', ':')) if plot_paras else ""
    if dict_str:
        dict_str = dict_str.replace('{', '').replace('}', '').replace('"', '').replace(' ', '_').replace(':', '_')
        save_path = f'./rsl/2D-{save_name}+{dict_str}'
    else:
        save_path = f'./rsl/{save_name}'
    plt.savefig(save_path[:100] + '.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.savefig(save_path[:100] + '.svg', bbox_inches='tight', pad_inches=0, transparent=True)

    if show:
        plt.show()
    plt.close()
