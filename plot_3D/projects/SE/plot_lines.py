import pickle

from matplotlib import pyplot as plt

from plot_3D.advance_plot_styles.polar_plot import plot_polar_line
from plot_3D.core.plot_3D_params_space_plt import *
from plot_3D.core.utils import *


def main():
    # 2. 手动读取处理（展开代码）
    ref_data = load_lumerical_jsondata(r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\sweep_NA\PL_Analysis_Ref.json')
    target_data = load_lumerical_jsondata(r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\sweep_NA\PL_Analysis.json')

    # NOTE: data structure:
    # {
    #   'key1': {
    #       '_complex': bool,
    #       '_data': list,
    #       '_size': list,
    #       '_type': str,  # could be 'matrix'
    #   },
    #   'key2': {...}
    # }

    # data.items, keys
    print(f"ref_data.keys(): {ref_data.keys()}")
    print(f"target_data.keys(): {target_data.keys()}")

    freq = structure_lumerical_jsondata(ref_data, 'freq')
    ref_NA_list = structure_lumerical_jsondata(ref_data, 'NA_list')
    # ref_k_list = structure_lumerical_jsondata(ref_data, 'k_list')
    ref_farfield_power_from_trans = structure_lumerical_jsondata(ref_data, 'ref_farfield_power_from_trans')
    ref_integrate_farfield_power = structure_lumerical_jsondata(ref_data, 'ref_integrate_farfield_power')
    ref_purcell_factors = structure_lumerical_jsondata(ref_data, 'ref_purcell_factors')
    ref_theta = structure_lumerical_jsondata(ref_data, 'ref_theta')
    ref_total_power = structure_lumerical_jsondata(ref_data, 'ref_total_power')

    target_k_list = structure_lumerical_jsondata(target_data, 'k_list')
    target_farfield_power_from_trans = structure_lumerical_jsondata(target_data, 'farfield_power_from_trans')
    target_integrate_farfield_power = structure_lumerical_jsondata(target_data, 'integrate_farfield_power')
    target_purcell_factors = structure_lumerical_jsondata(target_data, 'purcell_factors')
    theta = structure_lumerical_jsondata(target_data, 'theta')
    target_total_power = structure_lumerical_jsondata(target_data, 'total_power')
    target_E2_hyperdata = structure_lumerical_jsondata(target_data, 'ref_E2_hyperdata')

    target_integrate_PL_factor = target_integrate_farfield_power/ref_integrate_farfield_power/3.5/3.5


    figsize = (4, 5)
    save_dir = './rsl'
    show = True

    paras_1 = target_k_list.ravel().tolist()
    paras_2 = ref_NA_list.ravel().tolist()
    paras_3 = freq.ravel().tolist()[::1]
    x_vals_list = []
    y_vals_list = []
    y_max_val_list = []

    global_plot_params = {
        ''
    }

    # dataset_shape: target_purcell_factors(freq, k_list, NA_list)  note: NA_list could be 1 for some data

    # MODE0 直接绘制
    fig, ax = plt.subplots(figsize=figsize)
    plot_params = {
        'add_colorbar': True, 'cmap': 'magma',
    }
    ax = plot_line_advanced(ax, freq.ravel(), z1=-target_farfield_power_from_trans[:, 0, 0], z2=None, z3=None, **plot_params)

    # MODE1 绘制不同k的线图重叠
    for i, para in enumerate(paras_1):
        # # plot group2
        x_vals = np.linspace(freq.ravel()[0], freq.ravel()[-1], 1000)
        y_vals = target_purcell_factors[:, i, 0].ravel()
        x_vals_list.append(x_vals)
        y_vals_list.append(y_vals)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (x_vals, y_vals) in enumerate(zip(x_vals_list, y_vals_list)):
        plot_params = {
            'add_colorbar': True, 'cmap': 'magma',
        }
        ax = plot_line_advanced(ax, x_vals, z1=y_vals, z2=None, z3=None, **plot_params, index=i)


    # MODE2 绘制不同频率的角分辨图
    for i, para in enumerate(paras_3):
        # # plot group3
        x_vals = theta.ravel()
        y_vals = target_E2_hyperdata[:, i, 0, 0].ravel()

        x_vals_list.append(x_vals)
        y_vals_list.append(y_vals)

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': 'polar'})
    for i, (x_vals, y_vals) in enumerate(zip(x_vals_list, y_vals_list)):
        plot_params = {
            'add_colorbar': True, 'cmap': 'magma',
        }
        ax = plot_polar_line(ax, x_vals, y_vals/y_vals.max(), **plot_params)
        ax.set_theta_zero_location('N')  # 0度在上方
        ax.set_theta_direction(-1)  # 顺时针
        ax.set_thetalim(np.deg2rad(-60), np.deg2rad(60))  # 限制显示范围

    # MODE3 绘制不同k的PL最大值线图
    for i, para in enumerate(paras_1):
        x_vals = freq.ravel()
        y_vals = -2*target_farfield_power_from_trans[:, i, 0].ravel()
        x_vals_list.append(x_vals)
        y_vals_list.append(y_vals)
        # 降维算法  (消去维度1, 最终绘制 paras vs y_max_val 的曲线图)
        y_max_val = np.max(y_vals)
        y_max_val_list.append(y_max_val)

    fig, ax = plt.subplots(figsize=figsize)
    plot_params = {
        'add_colorbar': True, 'cmap': 'magma',
    }
    ax = plot_line_advanced(ax, np.array(paras_1), z1=np.array(y_max_val_list), z2=None, z3=None, **plot_params, index=i)

    ########################################################
    # Step 2: 添加注解 (直接调用现有)
    annotations = {
        'xlabel': r"", 'ylabel': "",
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
    main()
