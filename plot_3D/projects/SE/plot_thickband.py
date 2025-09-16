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
    x_vals = temp_x_vals  # 更新x_vals

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

    # Step 2: 添加注解 (直接调用现有)
    annotations = {
        'xlabel': r"k", 'ylabel': "f (c/P)",
        'y_log_scale': True,
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
