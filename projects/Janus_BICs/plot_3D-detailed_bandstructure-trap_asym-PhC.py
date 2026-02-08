from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.process_multi_dim_params_space import create_data_grid, group_solution

import numpy as np
import pandas as pd

from core.utils import norm_freq, convert_complex

c_const = 299792458

if __name__ == '__main__':
    data_path = "data/StrB-ultra_mesh-geo_merging_trap_asym-around_Γ.csv"
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    def norm_freq(freq, period):
        return freq/(c_const/period)
    period = 400
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=period*1e-9*1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period*1e-9)
    df_sample["phi (rad)"] = df_sample["phi (rad)"].apply(lambda x: x % np.pi)
    # # 筛选m1<0.1的成分
    # df_sample = df_sample[df_sample["m1"] < 0.05]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2", "buffer (nm)", "trap_factor", "rot_angle"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "fake_factor (1)", "频率 (Hz)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            'buffer (nm)': 180,
            'trap_factor': 0.05,
            'rot_angle': 0,
            # 'buffer (nm)': 600,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            "频率 (Hz)": {">": 0.0, "<": 0.5},  # 筛选
        }
    )

    # ###############################################################################################################
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 12))
    # xs = []
    # ys = []
    # zs = []
    # colors = []
    # for i, m1 in enumerate(new_coords['m1']):
    #     for j, m2 in enumerate(new_coords['m2']):
    #         lst_ij = Z_filtered[i][j]
    #         for freq in lst_ij[0]:
    #             xs.append(m1)
    #             ys.append(m2)
    #             zs.append(freq.real)
    #             colors.append(freq.imag)
    # sc = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o', alpha=0.8, s=1)
    # # set aspect
    # ax.set_box_aspect([1, 1, 3])
    # # set view angle
    # ax.view_init(elev=15, azim=45)
    # plt.colorbar(sc, label='Imaginary Part of Frequency (THz)')
    # ax.set_xlabel('m1')
    # ax.set_ylabel('m2')
    # ax.set_zlabel('Frequency (THz)')
    # plt.title('3D Scatter Plot of Eigenfrequencies')
    # plt.show()
    # ###############################################################################################################

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
        max_m=8,
        auto_split_streams=False
    )

    Z_targets = []
    for band_index in range(8):
        new_coords, Z_target = group_solution(
            new_coords, Z_grouped,
            freq_index=band_index  # 第n个频率
        )
        Z_targets.append(Z_target)

    ###################################################################################################################
    from core.process_multi_dim_params_space import extract_adjacent_fields
    from core.prepare_plot import prepare_plot_data
    from core.data_postprocess.data_package import package_stad_C2_data
    from core.prepare_plot import prepare_plot_data

    datasets = []
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
        datasets.append(dataset)

    band_index_A = 5
    Z_target_A = Z_targets[band_index_A]
    full_coords, dataset_A = package_stad_C2_data(
        new_coords, band_index_A, Z_target_A, additional_Z_grouped, z_keys,
        q_key='品质因子 (1)',
        tanchi_key='tanchi (1)',
        phi_key='phi (rad)',
        axis='x',
    )
    data_path = prepare_plot_data(
        coords=full_coords, data_class='Eigensolution', dataset_list=[dataset_A], fixed_params={},
        save_dir='./rsl/2_para_space',
    )

    ####################################################################################################################
    from core.plot_cls import MomentumSpaceEigenVisualizer
    from core.plot_workflow import PlotConfig

    BAND_INDEX = 0
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.5, 1.5), tick_direction='in')

    plotter = MomentumSpaceEigenVisualizer(config=config, data_path=data_path)
    plotter.load_data()

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
    plotter.imshow_field(index=BAND_INDEX, field_key='qlog', cmap='nipy_spectral', vmin=2, vmax=8)
    plotter.add_annotations()
    plotter.save_and_show()

    print("绘图已经整理到其他文件中")
    ####################################################################################################################
