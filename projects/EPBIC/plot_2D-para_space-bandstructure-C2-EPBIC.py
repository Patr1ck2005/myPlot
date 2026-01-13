from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/EP_BIC-L_space-diff_t_die.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))

    def norm_freq(freq, period):
        return freq/(c_const/period)

    def recognize_sp(phi_arr, kx_arr, ky_arr):
        # 对于 ky=0 的情况，phi=π/2 为 s 偏振, phi=0 为 p 偏振
        # 对于 ky=kx 的情况，phi=π/4 为 s 偏振，phi=3*π/4 为 p 偏振
        sp_polar = []
        for phi, kx, ky in zip(phi_arr, kx_arr, ky_arr):
            if np.isclose(ky, 0):
                if np.isclose(phi, np.pi/2, atol=1e-1):
                    sp_polar.append(1)
                else:
                    sp_polar.append(0)
            elif np.isclose(ky, kx):
                if np.isclose(phi, np.pi/4, atol=1e-1):
                    sp_polar.append(1)
                else:
                    sp_polar.append(0)
            else:
                sp_polar.append(-1)
        return sp_polar


    period = 1500
    df_sample["特征频率 (THz)"] = (df_sample["特征频率 (THz)"].apply(convert_complex)
                                   .apply(norm_freq, period=period * 1e-9 * 1e12))
    df_sample["频率 (Hz)"] = np.real(df_sample["特征频率 (THz)"])
    df_sample["k"] = df_sample["a"]
    df_sample["fake_factor (1)"] = 1 / df_sample["S3 (1)"]
    param_keys = ["k", "spacer (nm)", "h_die_grating (nm)"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "fake_factor (1)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    X_KEY = 'spacer (nm)'

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            'k': 0,
            # "h_die_grating (nm)": 461.5,
            # "h_die_grating (nm)": 461.4,
            "h_die_grating (nm)": 461.3,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            "特征频率 (THz)": {"<": 0.60, ">": 0},  # 筛选
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

    Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=3,
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
    new_coords, Z_target3 = group_solution(
        new_coords, Z_grouped,
        freq_index=2  # 第n个频率
    )

    print(np.nanmin(np.imag(Z_target1)), np.nanmax(np.imag(Z_target1)))


    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")

    dataset1 = {'eigenfreq_real': Z_target1.real, 'eigenfreq_imag': Z_target1.imag}
    dataset2 = {'eigenfreq_real': Z_target2.real, 'eigenfreq_imag': Z_target2.imag}
    dataset3 = {'eigenfreq_real': Z_target3.real, 'eigenfreq_imag': Z_target3.imag}

    data_path = prepare_plot_data(
        new_coords, data_class='Eigensolution', dataset_list=[
            dataset1,
            dataset2,
            dataset3,
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
            'ylim': (0.572, 0.576),
        },
    )
    config.update(figsize=(1.25, 0.75), tick_direction='in')
    plotter = OneDimFieldVisualizer(config=config, data_path=data_path)
    plotter.load_data()
    plotter.new_2d_fig()
    plotter.plot(index=0, x_key=X_KEY, z1_key='eigenfreq_real', default_color='green', twinx=False, )
    plotter.plot(index=1, x_key=X_KEY, z1_key='eigenfreq_real', default_color='gray', twinx=False, )
    plotter.plot(index=2, x_key=X_KEY, z1_key='eigenfreq_real', default_color='blue', twinx=False, )
    plotter.add_annotations()
    plotter.plot(index=0, x_key=X_KEY, z1_key='eigenfreq_imag', default_color='green', twinx=True, default_linestyle='--', )
    plotter.plot(index=1, x_key=X_KEY, z1_key='eigenfreq_imag', default_color='gray', twinx=True, default_linestyle='--', )
    plotter.plot(index=2, x_key=X_KEY, z1_key='eigenfreq_imag', default_color='blue', twinx=True, default_linestyle='--', )
    config.annotations.update(ylim=(0, 0.002))
    plotter.add_twinx_annotations()
    plotter.save_and_show()

