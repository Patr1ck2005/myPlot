from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.process_multi_dim_params_space import create_data_grid, group_solution

import numpy as np
import pandas as pd

from core.utils import norm_freq, convert_complex

c_const = 299792458

if __name__ == '__main__':
    data_path = "data/Hex-1Dms-ultra_mesh-geo_tri_asym-0.1k.csv"
    df_sample = pd.read_csv(data_path, sep='\t')

    period = 400
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq,
                                                                                           period=period * 1e-9 * 1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period * 1e-9)
    df_sample["phi (rad)"] = df_sample["phi (rad)"].apply(lambda x: x % np.pi)
    df_sample["cx (V/m)"] = df_sample["cx (V/m)"].apply(convert_complex)
    df_sample["cy (V/m)"] = df_sample["cy (V/m)"].apply(convert_complex)
    df_sample["k"] = df_sample["m1"]-df_sample["m2"]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = [
        "k", "t_tot (nm)", "r1 (nm)", "r2 (nm)", "substrate (nm)",
        "asym_y_scaling", "tri_factor", "rot_angle (deg)"
    ]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "频率 (Hz)", "cx (V/m)", "cy (V/m)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    X_KEY = 'k'

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            't_tot (nm)': 150,
            'r1 (nm)': 150,
            'r2 (nm)': 0,
            'substrate (nm)': 500,
            'asym_y_scaling': 1.0,
            'tri_factor': 0.125,
            'rot_angle (deg)': 0,
        },  # 固定
        filter_conditions={
            # "fake_factor (1)": {"<": 1},  # 筛选
            "频率 (Hz)": {">": 0.0, "<": 0.5},  # 筛选
        }
    )

    deltas = (1e-3,)  # n个维度的网格间距
    # 当沿维度 d 生长时，值差权重矩阵（n×n）
    # 例如：value_weights[d, j] = 在 grow_dir=d 时，对维度 j 的值差权重
    value_weights = np.array([
        [1, ],  # 沿维度生长时
    ])
    # 当沿维度 d 生长时，导数不连续权重矩阵（n×n）
    deriv_weights = np.array([
        [1, ],
    ])
    # 创建一个新的数组，用于存储更新后的结果
    Z_new = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        Z_new[i] = Z_filtered[i][0]  # 提取每个 lst_ij 的第 b 列

    ###############################################################################################################
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
    ###############################################################################################################

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=15,
        auto_split_streams=False
    )

    Z_targets = []
    for freq_index in range(15):
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
        eigenfreq, qfactor, up_tanchi, up_phi, freq, \
            up_cx, up_cy= extract_adjacent_fields(
            additional_Z_grouped,
            z_keys=z_keys,
            band_index=i
        )
        qlog = np.log10(qfactor)
        dataset['qlog'] = qlog.real.ravel()
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
            'ylim': (0.30, 0.60),
        },
    )
    config.update(figsize=(1.25, 3), tick_direction='in')
    plotter = OneDimFieldVisualizer(config=config, data_path=data_path)
    plotter.load_data()
    plotter.new_2d_fig()
    for i in range(15):
        plotter.plot(
            index=i, x_key=X_KEY, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
            enable_fill=True, default_color='gray', alpha_fill=0.3, scale=1
        )
    for i in range(15):
        plotter.plot(
            index=i, x_key=X_KEY, z1_key='eigenfreq_real', z3_key='qlog', cmap='nipy_spectral',
            enable_dynamic_color=True, global_color_vmin=2, global_color_vmax=7, linewidth_base=2
        )
    plotter.add_annotations()
    plotter.save_and_show()
