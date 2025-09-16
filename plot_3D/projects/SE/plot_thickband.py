import pickle

from matplotlib import pyplot as plt

from plot_3D.core.plot_3D_params_space_plt import *


def main(data_path):
    # 2. 手动读取处理（展开代码）
    with open(data_path, 'rb') as f:
        plot_data = pickle.load(f)
    # ... 这里手动处理：打印、验证、修改 ...
    x_vals = plot_data['x_vals']
    subs = plot_data['subs']

    x_key = plot_data['metadata']['x_key']
    figsize = (4, 5)
    save_dir = './rsl/delta_space'
    fixed_params = {}
    show = True

    x_vals_list = []
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
        x_vals_list.append(temp_x_vals)
        y_vals_list.append(y_vals)

    # # 另一种加载.csv数据的方法
    # df = load_csv_data('./rsl/delta_space/20240320_172338.csv')
    # df.columns = ['x_vals', 'y_vals']
    # x_vals = df['x_vals'].values
    # y_vals = df['y_vals'].values
    #
    # x_vals_list = [x_vals]
    # y_vals_list = [y_vals]


    """
    阶段3: 从已加载数据集成绘制图像。
    """

    # Step 1: 调用现有绘图核心 (从历史plot_Z复制)
    fig, ax = plt.subplots(figsize=figsize)

    plot_params = {
        'add_colorbar': True, 'cmap': 'magma',
        'title': False,
    }

    plot_params = {
        'enable_fill': True, 'gradient_fill': True, 'gradient_direction': 'z3', 'cmap': 'magma', 'add_colorbar': False,
        "global_color_vmin": 0, "global_color_vmax": 5e-3, "default_color": 'gray', 'legend': False, 'alpha_fill': 1,
        'edge_color': 'none', 'title': False, 'scale': 1,
    }

    fig, ax = plot_1d_lines(ax, x_vals, y_vals_list, plot_params)

    default_color_list = plot_params.get('default_color_list', None)
    enable_line_fill = plot_params.get('enable_line_fill', True)
    scale = 1

    y_mins, y_maxs = [], []
    for i, (x_vals, y_vals) in enumerate(zip(x_vals_list, y_vals_list)):
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

    # Step 2: 添加注解 (直接调用现有)
    annotations = {
        'xlabel': r"", 'ylabel': "f (c/P)",
        # 'xlim': (0.430, 0.440), 'ylim': (0, 1.15e11),
    }

    fig, ax = add_annotations(ax, annotations)

    plt.tight_layout()

    # Step 3: 保存图像 (从历史复制)
    full_params = {**plot_params}
    image_path = generate_save_name(save_dir, full_params)
    plt.savefig(image_path, dpi=300, bbox_inches="tight", transparent=True)
    print(f"图像已保存为：{image_path} 🎨")

    if show:
        plt.show()


if __name__ == '__main__':
    main(
        r'D:\DELL\Documents\myPlots\plot_3D\projects\SE/rsl/eigensolution\20250916_175843\plot_data__x-w_delta_factor_1d.pkl')
