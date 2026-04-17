from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.process_multi_dim_params_space import create_data_grid, group_solution

import numpy as np
import pandas as pd

from core.utils import norm_freq, convert_complex

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/PhC-Trap-I-t_slab_space-vary_fill-t_tot-norm_mesh.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    period = 500
    df_sample["特征频率 (THz)"] = (df_sample["特征频率 (THz)"].apply(convert_complex)
                                   .apply(norm_freq, period=period * 1e-9 * 1e12))
    df_sample["频率 (Hz)"] = np.real(df_sample["特征频率 (THz)"])
    param_keys = ["m1", "m2", "t_slab (nm)", "t_tot (nm)", "fill", "trap_asym", "slit_delta", "slit_shift_delta", "substrate (nm)"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "fake_factor (1)", "up_S3 (1)", "U_factor (1)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    X_KEY = 't_slab (nm)'

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            'm1': 0.00,
            'm2': 0.00,
            't_tot (nm)': 150,
            'fill': 0.75,
            # 'trap_asym': 0.15,
            'trap_asym': 0.1,
            'slit_delta': 0.0,
            'slit_shift_delta': 0.0,
            'substrate (nm)': 1500,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            "品质因子 (1)": {"<": 1e9},  # 筛选
            # "特征频率 (THz)": {"<": 0.60, ">": 0},  # 筛选
        }
    )

    deltas = (1e-3,)  # n个维度的网格间距
    # 当沿维度 d 生长时，值差权重矩阵（n×n）
    # 例如：value_weights[d, j] = 在 grow_dir=d 时，对维度 j 的值差权重
    value_weights = np.array([
        [1,],   # 沿维度生长时
    ])
    # 当沿维度 d 生长时，导数不连续权重矩阵（n×n）
    deriv_weights = np.array([
        [1,],
    ])
    # 创建一个新的数组，用于存储更新后的结果
    Z_new = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        Z_new[i] = Z_filtered[i][0]  # 提取每个 lst_ij 的第 b 列

# ###############################################################################################################
#     from matplotlib import pyplot as plt
#     fig, ax = plt.subplots(figsize=(6, 10))
#     # 通过散点的方式绘制出来，看看效果
#     for i in range(Z_new.shape[0]):
#         z_vals = Z_new[i]
#         for val in z_vals:
#             if val is not None:
#                 plt.scatter(new_coords[X_KEY][i], np.real(val), color='blue', s=10)
#     plt.xlabel(X_KEY)
#     plt.ylabel('Re(eigenfreq) (THz)')
#     plt.title('Filtered Eigenfrequencies before Grouping')
#     plt.grid(True)
#     plt.show()
#     ###############################################################################################################
    MAX_NUM = 20
    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=MAX_NUM,
        nan_cost_penalty=1e1,
        auto_split_streams=False
    )

    Z_targets = []
    for freq_index in range(MAX_NUM):
        new_coords, Z_target = group_solution(
            new_coords, Z_grouped,
            freq_index=freq_index  # 第n个频率
        )
        Z_targets.append(Z_target)

    ###################################################################################################################
    from core.plot_workflow import PlotConfig
    from core.prepare_plot import prepare_plot_data
    from core.process_multi_dim_params_space import extract_adjacent_fields

    datasets = []
    for i, Z_target in enumerate(Z_targets):
        dataset = {'eigenfreq_real': Z_target.real, 'eigenfreq_imag': Z_target.imag}
        # z_keys = ["特征频率 (THz)", "品质因子 (1)", "fake_factor (1)", "up_S3", "U_factor"]
        eigenfreq, qfactor, fake_factor, up_s3, u_factor = extract_adjacent_fields(
            additional_Z_grouped,
            z_keys=z_keys,
            band_index=i
        )
        qlog = np.log10(qfactor)
        dataset['qlog'] = qlog.real.ravel()
        dataset['-u_factor'] = np.abs(u_factor.real.ravel())
        dataset['up_s3'] = up_s3.real.ravel()
        print(f"Band {i}: qlog range = [{dataset['qlog'].min()}, {dataset['qlog'].max()}]")
        datasets.append(dataset)

    data_path = prepare_plot_data(
        new_coords, data_class='Eigensolution', dataset_list=datasets, fixed_params={},
        save_dir='./rsl/1_para_space',
    )

    # ============================================================================================================
    from core.plot_workflow import PlotConfig
    from core.plot_cls import OneDimFieldVisualizer

    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={
            'xlabel': '', 'ylabel': '',
            'show_axis_labels': True, 'show_tick_labels': True,
            # 'ylim': (0.30, 0.60),
        },
    )
    config.update(figsize=(2, 2), tick_direction='in')
    plotter = OneDimFieldVisualizer(config=config, data_path=data_path)
    plotter.load_data()
    plotter.new_2d_fig()
    for i in range(MAX_NUM):
        plotter.plot(
            index=i, x_key=X_KEY, z1_key='-u_factor', z3_key='qlog', cmap='nipy_spectral',
            # enable_dynamic_color=True, global_color_vmin=2, global_color_vmax=7, linewidth_base=2
        )
    plotter.adjust_view_2dim_auto()
    plotter.ax.set_yscale('log')
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.re_initialized_plot()
    plotter.new_2d_fig()
    for i in range(MAX_NUM):
        plotter.plot(
            index=i, x_key=X_KEY, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
            # enable_fill=True, default_color='gray', alpha_fill=0.3, scale=1
        )
    # for i in range(MAX_NUM):
    #     plotter.plot(
    #         index=i, x_key=X_KEY, z1_key='eigenfreq_real', z3_key='qlog', cmap='nipy_spectral',
    #         # enable_dynamic_color=True, global_color_vmin=2, global_color_vmax=7, linewidth_base=2
    #     )
    plotter.adjust_view_2dim_auto()
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.re_initialized_plot()
    plotter.new_2d_fig()
    for i in range(MAX_NUM):
        plotter.plot(
            index=i, x_key=X_KEY, z1_key='qlog', z2_key='qlog',
            # enable_fill=True, default_color='gray', alpha_fill=0.3, scale=1
        )
    plotter.adjust_view_2dim_auto()
    plotter.add_annotations()
    plotter.save_and_show()
