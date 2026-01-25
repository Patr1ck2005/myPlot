from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.process_multi_dim_params_space import create_data_grid, group_solution

import numpy as np
import pandas as pd

from core.utils import norm_freq, convert_complex

if __name__ == '__main__':
    data_path = 'data/EP_BIC-sliding_space.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    period = 1500
    df_sample["特征频率 (THz)"] = (df_sample["特征频率 (THz)"].apply(convert_complex)
                                   .apply(norm_freq, period=period * 1e-9 * 1e12))
    df_sample["频率 (Hz)"] = np.real(df_sample["特征频率 (THz)"])
    df_sample["k"] = df_sample["a"]
    df_sample["fake_factor (1)"] = 1 / df_sample["S3 (1)"]
    df_sample["sliding"] = (df_sample["layer_shift_top (nm)"]-df_sample["layer_shift_btn (nm)"])/1500
    param_keys = ["k", "spacer (nm)", "h_die_grating (nm)", "sliding"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "fake_factor (1)",
              "Efield_top (kg^2*m^4/(s^6*A^2))", "Efield_btn (kg^2*m^4/(s^6*A^2))", "Hfield_top (A^2)", "Hfield_btn (A^2)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    X_KEY = 'sliding'

    # 已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            'k': 0,
            "h_die_grating (nm)": 461.3,
            "spacer (nm)": 2000,  # 2250
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
    from matplotlib import pyplot as plt
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

    ###################################################################################################################
    from core.process_multi_dim_params_space import extract_basic_analysis_fields, extract_adjacent_fields
    from core.prepare_plot import prepare_plot_data

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        additional_data=Z_filtered,
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

    # 提取 band= 的附加场数据
    eigenfreq1, qfactor1, fake_factor1, Efield_top1, Efield_btn1, Hfield_top1, Hfield_btn1 = extract_adjacent_fields(
        additional_Z_grouped,
        z_keys=z_keys,
        band_index=0
    )
    eigenfreq2, qfactor2, fake_factor2, Efield_top2, Efield_btn2, Hfield_top2, Hfield_btn2 = extract_adjacent_fields(
        additional_Z_grouped,
        z_keys=z_keys,
        band_index=1
    )
    # print(additional_Z_grouped)
    eigenfreq3, qfactor3, fake_factor3, Efield_top3, Efield_btn3, Hfield_top3, Hfield_btn3 = extract_adjacent_fields(
        additional_Z_grouped,
        z_keys=z_keys,
        band_index=2
    )

    dataset1 = {
        'eigenfreq_real': Z_target1.real, 'eigenfreq_imag': Z_target1.imag,
        'field_top': Hfield_top1.real/(Hfield_top1.real+Hfield_btn1.real),
        'field_btn': Hfield_btn1.real/(Hfield_top1.real+Hfield_btn1.real)
    }
    dataset2 = {
        'eigenfreq_real': Z_target2.real, 'eigenfreq_imag': Z_target2.imag,
        'field_top': Hfield_top2.real/(Hfield_top2.real+Hfield_btn2.real),
        'field_btn': Hfield_btn2.real/(Hfield_top2.real+Hfield_btn2.real)
    }
    dataset3 = {
        'eigenfreq_real': Z_target3.real, 'eigenfreq_imag': Z_target3.imag,
        'field_top': Hfield_top3.real/(Hfield_top3.real+Hfield_btn3.real),
        'field_btn': Hfield_btn3.real/(Hfield_top3.real+Hfield_btn3.real)
    }

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
            'ylim': (0.0, 1),
        },
    )
    config.update(figsize=(1.25, 0.75), tick_direction='in')
    plotter = OneDimFieldVisualizer(config=config, data_path=data_path)
    plotter.load_data()
    plotter.new_2d_fig()
    plotter.plot(index=0, x_key=X_KEY, z1_key='field_top', default_color='green', twinx=False, )
    plotter.plot(index=1, x_key=X_KEY, z1_key='field_top', default_color='gray', twinx=False, )
    plotter.plot(index=2, x_key=X_KEY, z1_key='field_top', default_color='blue', twinx=False, )
    # plotter.plot(index=0, x_key=X_KEY, z1_key='eigenfreq_real', default_color='green', twinx=False, )
    # plotter.plot(index=1, x_key=X_KEY, z1_key='eigenfreq_real', default_color='gray', twinx=False, )
    # plotter.plot(index=2, x_key=X_KEY, z1_key='eigenfreq_real', default_color='blue', twinx=False, )
    plotter.add_annotations()
    plotter.plot(index=0, x_key=X_KEY, z1_key='field_btn', default_color='green', twinx=True, default_linestyle='--', )
    plotter.plot(index=1, x_key=X_KEY, z1_key='field_btn', default_color='gray', twinx=True, default_linestyle='--', )
    plotter.plot(index=2, x_key=X_KEY, z1_key='field_btn', default_color='blue', twinx=True, default_linestyle='--', )
    config.annotations.update(ylim=(0, 1))
    plotter.add_twinx_annotations()
    plotter.save_and_show()

