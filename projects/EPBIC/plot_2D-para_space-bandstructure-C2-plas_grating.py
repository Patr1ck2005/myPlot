from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.plot_3D_params_space_plt import *
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

from core.utils import norm_freq, convert_complex

if __name__ == '__main__':
    data_path = 'data/plas_BIC-delta_space.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    period = 1500
    df_sample["特征频率 (THz)"] = (df_sample["特征频率 (THz)"].apply(convert_complex)
                                   .apply(norm_freq, period=period * 1e-9 * 1e12))
    df_sample["频率 (Hz)"] = np.real(df_sample["特征频率 (THz)"])
    df_sample["k"] = df_sample["a"]
    df_sample["fake_factor (1)"] = 1 / df_sample["S3 (1)"]
    param_keys = ["k", "spacer (nm)", "h_plas_grating (nm)", "plas_wid_scale (nm)"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "fake_factor (1)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    X_KEY = 'plas_wid_scale (nm)'

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            'k': 0,
            "h_plas_grating (nm)": 400,
            "spacer (nm)": 2000,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
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
    # ============================================================================================================
    fig, ax = plt.subplots(figsize=(6, 10))
    # 通过散点的方式绘制出来，看看效果
    for i in range(Z_new.shape[0]):
        z_vals = Z_new[i]
        for val in z_vals:
            if val is not None:
                plt.scatter(new_coords[X_KEY][i], np.real(val), color='blue', s=10)
    plt.xlabel(X_KEY)
    plt.ylabel('Re(eigenfreq) (THz)')
    plt.title('Filtered Eigenfrequencies before Grouping')
    plt.grid(True)
    plt.show()
    # ============================================================================================================

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=2,
        auto_split_streams=False
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_solution(
        new_coords, Z_grouped,
        freq_index=0  # 第n个频率
    )
    new_coords, Z_target2 = group_solution(
        new_coords, Z_grouped,
        freq_index=1  # 第n个频率
    )

    dataset1 = {'eigenfreq_real': Z_target1.real, 'eigenfreq_imag': Z_target1.imag}
    dataset2 = {'eigenfreq_real': Z_target2.real, 'eigenfreq_imag': Z_target2.imag}

    data_path = prepare_plot_data(
        new_coords, data_class='Eigensolution', dataset_list=[
            dataset1,
            dataset2,
        ], fixed_params={},
        save_dir='./rsl/eigensolution',
    )

    # ============================================================================================================
    from core.plot_workflow import PlotConfig
    from core.plot_cls import OneDimFieldVisualizer

    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={
            'xlabel': '', 'ylabel': '',
            'show_axis_labels': True, 'show_tick_labels': True,
            'ylim': (0.573, 0.576),
            'xlim': (0, 10),
        },
    )
    config.update(figsize=(1., 1), tick_direction='in')
    plotter = OneDimFieldVisualizer(config=config, data_path=data_path)
    plotter.load_data()
    plotter.new_2d_fig()
    plotter.plot(index=0, x_key=X_KEY, z1_key='eigenfreq_real', default_color='green', twinx=False, )
    plotter.plot(index=1, x_key=X_KEY, z1_key='eigenfreq_real', default_color='blue', twinx=False, )
    plotter.add_annotations()
    plotter.plot(index=0, x_key=X_KEY, z1_key='eigenfreq_imag', default_color='green', twinx=True, default_linestyle='--', )
    plotter.plot(index=1, x_key=X_KEY, z1_key='eigenfreq_imag', default_color='blue', twinx=True, default_linestyle='--', )
    config.annotations.update(ylim=(0, 0.0018))
    plotter.add_twinx_annotations()
    plotter.save_and_show()

