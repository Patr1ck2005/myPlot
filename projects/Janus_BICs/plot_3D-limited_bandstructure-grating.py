from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.process_multi_dim_params_space import create_data_grid, group_solution

import numpy as np
import pandas as pd

from core.utils import norm_freq, convert_complex

c_const = 299792458

if __name__ == '__main__':
    # data_path = 'data/VacuumEnv-ultra_mesh-search0.40-geo_FW_BIC-around.csv'
    data_path = 'data/AsymEnv-ultra_mesh-search0.40-geo_FW_QBIC-around.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    period = 500
    df_sample["特征频率 (THz)"] = (df_sample["特征频率 (THz)"].apply(convert_complex)
                                   .apply(norm_freq, period=period * 1e-9 * 1e12))
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period * 1e-9)
    df_sample["up_cx (V/m)"] = df_sample["up_cx (V/m)"].apply(convert_complex)
    df_sample["up_cy (V/m)"] = df_sample["up_cy (V/m)"].apply(convert_complex)
    df_sample["down_cx (V/m)"] = df_sample["down_cx (V/m)"].apply(convert_complex)
    df_sample["down_cy (V/m)"] = df_sample["down_cy (V/m)"].apply(convert_complex)
    # df_sample = df_sample[df_sample["m1"] <= 0.2]
    # df_sample = df_sample[df_sample["m2"] <= 0.2]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2", "t_ridge (nm)", "fill", "t_tot (nm)", "substrate_n"]
    z_keys = [
        "特征频率 (THz)", "品质因子 (1)",
        "up_tanchi (1)", "up_phi (rad)",
        "down_tanchi (1)", "down_phi (rad)",
        "fake_factor (1)", "频率 (Hz)",
        "U_factor (1)", "up_cx (V/m)", "up_cy (V/m)", "down_cx (V/m)", "down_cy (V/m)"
    ]

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
            "t_tot (nm)": 520,
            "t_ridge (nm)": 520,
            "fill": 0.5,
            "substrate_n": 1.2,
            # "substrate_n": 1.0,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 2},  # 筛选
            # "频率 (Hz)": {">": 0.32, "<": 0.45},  # 筛选
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
        max_m=14,
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
    new_coords, Z_target4 = group_solution(
        new_coords, Z_grouped,
        freq_index=3  # 第n个频率
    )
    new_coords, Z_target5 = group_solution(
        new_coords, Z_grouped,
        freq_index=4  # 第n个频率
    )
    # new_coords, Z_target6 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=5  # 第n个频率
    # )
    # new_coords, Z_target7 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=6  # 第n个频率
    # )

    ###################################################################################################################
    from core.process_multi_dim_params_space import extract_adjacent_fields
    from core.prepare_plot import prepare_plot_data
    from core.data_postprocess.data_package import package_stad_C2_data
    (eigenfreq1, qfactor1, up_tanchi, up_phi, down_tanchi, down_phi, fake_factor, freq, u_factor1,
     up_cx1, up_cy1, down_cx1, down_cy1) = extract_adjacent_fields(
        additional_Z_grouped,
        z_keys=z_keys,
        band_index=0
    )
    (eigenfreq2, qfactor2, up_tanchi, up_phi, down_tanchi, down_phi, fake_factor, freq, u_factor2,
     up_cx2, up_cy2, down_cx2, down_cy2) = extract_adjacent_fields(
        additional_Z_grouped,
        z_keys=z_keys,
        band_index=1
    )

    # imshow up_cx
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    normed_up_cx = up_cx1 / (np.abs(up_cx1)+np.abs(up_cy1))
    # normed_up_cy = up_cy1 / (np.abs(up_cx1)+np.abs(up_cy1))
    normed_up_cy = up_cy1
    c = ax.imshow(np.imag(normed_up_cy).T, origin='lower', extent=(
        new_coords['m1'][0], new_coords['m1'][-1],
        new_coords['m2'][0], new_coords['m2'][-1],
    ), aspect='auto', cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig('./c.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    normed_up_cx = up_cx1 / (np.abs(up_cx1)+np.abs(up_cy1))
    normed_up_cy = up_cy1 / (np.abs(up_cx1)+np.abs(up_cy1))
    phase_diff = (np.angle(up_cx1) - np.angle(up_cy1) + np.pi)%(2*np.pi) - np.pi
    c = ax.imshow(phase_diff.T, origin='lower', extent=(
        new_coords['m1'][0], new_coords['m1'][-1],
        new_coords['m2'][0], new_coords['m2'][-1],
    ), aspect='auto', cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig('./c.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    # 取m2=0处的切片, 在复数平面上绘制normed_up_cy的实部和虚部
    m2_index = np.argmin(np.abs(new_coords['m2'] - 0))
    plt.scatter(np.real(normed_up_cy[:, m2_index]), np.imag(normed_up_cy[:, m2_index]), s=5, marker='+', color='k')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('./c.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    band_index_A = 0
    Z_target_A = eigenfreq1
    band_index_B = 1
    Z_target_B = eigenfreq2
    full_coords, dataset_A = package_stad_C2_data(
        new_coords, band_index_A, Z_target_A, additional_Z_grouped, z_keys,
        q_key='品质因子 (1)',
        tanchi_key='up_tanchi (1)',
        phi_key='up_phi (rad)',
        # tanchi_key='down_tanchi (1)',
        # phi_key='down_phi (rad)',
        axis='y',
    )
    _, dataset_B = package_stad_C2_data(
        new_coords, band_index_B, Z_target_B, additional_Z_grouped, z_keys,
        q_key='品质因子 (1)',
        tanchi_key='up_tanchi (1)',
        phi_key='up_phi (rad)',
        # tanchi_key='down_tanchi (1)',
        # phi_key='down_phi (rad)',
        axis='y',
    )
    data_path = prepare_plot_data(
        coords=full_coords, data_class='Eigensolution', dataset_list=[dataset_A, dataset_B], fixed_params={},
        save_dir='./rsl/2_para_space',
    )

    ####################################################################################################################
    from core.plot_cls import MomentumSpaceEigenVisualizer
    from core.plot_workflow import PlotConfig
    print("绘图已经整理到其他文件中")
    ####################################################################################################################
