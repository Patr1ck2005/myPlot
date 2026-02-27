from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.process_multi_dim_params_space import create_data_grid, group_solution

import numpy as np
import pandas as pd

from core.utils import norm_freq, convert_complex

c_const = 299792458

if __name__ == '__main__':
    # data_path = "data/Hex-ultra_mesh-geo_tri_asym-0.2k.csv"
    # data_path = "data/Hex-ultra_mesh-geo_tri_asym-0.2k-supp1.csv"
    # data_path = "data/Hex-ultra_mesh-geo_tri_asym-0.1k.csv"
    data_path = "data/Hex-ultra_mesh-geo-0.1k.csv"
    df_sample = pd.read_csv(data_path, sep='\t')

    period = 500*np.sqrt(3)/2
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq,
                                                                                           period=period * 1e-9 * 1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period * 1e-9)
    df_sample["phi (rad)"] = df_sample["phi (rad)"].apply(lambda x: x % np.pi)
    # filter m1, m2 to [0, 0.1]
    df_sample = df_sample[(df_sample["m1"] >= 0) & (df_sample["m1"] <= 0.1) & (df_sample["m2"] >= 0) & (df_sample["m2"] <= 0.1)]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = [
        "m1", "m2", "t_tot (nm)", "r1 (nm)", "r2 (nm)", "substrate (nm)",
        "asym_y_scaling", "tri_factor", "rot_angle (deg)"
    ]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "fake_factor (1)", "频率 (Hz)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    X_KEY = 'm1'
    Y_KEY = 'm2'

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            't_tot (nm)': 150,
            'r1 (nm)': 150,
            'r2 (nm)': 0,
            'substrate (nm)': 180,
            'asym_y_scaling': 1.0,
            # 'tri_factor': 0.125,
            'tri_factor': 0,
            'rot_angle (deg)': 0,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            "频率 (Hz)": {">": 0.4, "<": 0.6},  # 筛选
        }
    )

    ###############################################################################################################
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 12))
    xs = []
    ys = []
    zs = []
    colors = []
    for i, m1 in enumerate(new_coords[X_KEY]):
        for j, m2 in enumerate(new_coords[Y_KEY]):
            lst_ij = Z_filtered[i][j]
            for freq in lst_ij[0]:
                xs.append(m1)
                ys.append(m2)
                zs.append(freq.real)
                colors.append(freq.imag)
    sc = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o', alpha=0.8, s=1)
    # set aspect
    ax.set_box_aspect([1, 1, 3])
    # set view angle
    ax.view_init(elev=15, azim=45)
    plt.colorbar(sc, label='Imaginary Part of Frequency (THz)')
    ax.set_xlabel('m1')
    ax.set_ylabel('m2')
    ax.set_zlabel('Frequency (THz)')
    plt.title('3D Scatter Plot of Eigenfrequencies')
    plt.show()
    ###############################################################################################################
    BAND_NUM = 7
    deltas = (1e-3, 1e-3)  # n个维度的网格间距
    # 当沿维度 d 生长时，值差权重矩阵（n×n）
    # 例如：value_weights[d, j] = 在 grow_dir=d 时，对维度 j 的值差权重
    value_weights = np.array([
        [1, 1], [1, 1]  # 沿维度生长时
    ])
    # 当沿维度 d 生长时，导数不连续权重矩阵（n×n）
    deriv_weights = np.array([
        [1, 1], [1, 1]
    ])
    # 创建一个新的数组，用于存储更新后的结果
    Z_new = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        for j in range(Z_filtered.shape[1]):
            Z_new[i, j] = Z_filtered[i][j][0]  # 提取每个 lst_ij 的第 b 列
            # Z_new[i, j] = np.imag(Z_filtered[i][j])[0]

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        nan_cost_penalty=1e1,
        max_m=BAND_NUM,
        auto_split_streams=False
    )

    Z_targets = []
    for band_index in range(BAND_NUM):
        new_coords, Z_target = group_solution(
            new_coords, Z_grouped,
            freq_index=band_index  # 第n个频率
        )
        Z_targets.append(Z_target)

    ###################################################################################################################
    from core.process_multi_dim_params_space import extract_adjacent_fields
    from core.prepare_plot import prepare_plot_data
    from core.data_postprocess.data_package import package_stad_C6_data
    from core.prepare_plot import prepare_plot_data

    raw_dataset_list = []
    for i, Z_target in enumerate(Z_targets):
        dataset = {'eigenfreq_real': Z_target.real, 'eigenfreq_imag': Z_target.imag}
        eigenfreq, qfactor, tanchi, phi, fake_factor, freq = extract_adjacent_fields(
            additional_Z_grouped,
            z_keys=z_keys,
            band_index=i
        )
        qlog = np.log10(qfactor)
        dataset['qlog'] = qlog.real.ravel()
        print(f"Band {i}: qlog range = [{dataset['qlog'].min()}, {dataset['qlog'].max()}]")
        raw_dataset_list.append(dataset)

    band_indices = (0, 1, 2, 3, 4, 5)
    dataset_list = []
    for band_index in band_indices:
        Z_target = Z_targets[band_index]
        full_coords, dataset = package_stad_C6_data(
            new_coords, band_index, Z_target, additional_Z_grouped, z_keys,
            q_key='品质因子 (1)',
            tanchi_key='tanchi (1)',
            phi_key='phi (rad)',
        )
        dataset_list.append(dataset)
    data_path = prepare_plot_data(
        coords=full_coords, data_class='Eigensolution', dataset_list=dataset_list, fixed_params={},
        save_dir='./rsl/2_para_space',
    )

    ####################################################################################################################
    from core.plot_cls import MomentumSpaceEigenVisualizer
    from core.plot_workflow import PlotConfig

    BAND_INDEX = 3  # 2, 3
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.5, 1.5), tick_direction='in')

    plotter = MomentumSpaceEigenVisualizer(config=config, data_path=data_path)
    plotter.load_data()

    # plotter.new_3d_fig(figsize=(3, 3))
    # plotter.plot_3d_surfaces(indices=(2, 3, 4), z1_key='eigenfreq_real', z2_key='qlog', cmap='nipy_spectral', vmin=2, vmax=8, alpha_default=1)
    # plotter.add_annotations()
    # plotter.ax.set_box_aspect([1, 1, 2])
    # plotter.save_and_show()

    # plotter.new_3d_fig(figsize=(3, 3))
    # plotter.plot_3d_surfaces(indices=(2, 3, 5), z1_key='eigenfreq_real', z2_key='s3', cmap='coolwarm', vmin=-1, vmax=1, alpha_default=1)
    # plotter.add_annotations()
    # plotter.ax.set_box_aspect([1, 1, 2])
    # plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='qlog', cmap='nipy_spectral', vmin=2, vmax=8)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(3.8, 1))
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.02, color='r', field_key='phi', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.03, color='k', field_key='phi', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.05, color='b', field_key='phi', alpha=0.5)
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s1', cmap='coolwarm', vmin=-1, vmax=1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s2', cmap='coolwarm', vmin=-1, vmax=1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s3', cmap='coolwarm', vmin=-1, vmax=1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='eigenfreq_real', cmap='nipy_spectral')
    plotter.add_annotations()
    plotter.save_and_show()

    print("绘图已经整理到其他文件中")
    ####################################################################################################################
