from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.plot_3D_params_space_plt import *
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/die_BIC-t_space.csv'
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

    KEY_X = 'h_die_grating (nm)'

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            'k': 0,
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
                plt.scatter(new_coords[KEY_X][i], np.real(val), color='blue', s=10)
    plt.xlabel(KEY_X)
    plt.ylabel('Re(eigenfreq) (THz)')
    plt.title('Filtered Eigenfrequencies before Grouping')
    plt.grid(True)
    plt.show()
    # ============================================================================================================

    Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=2
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


    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")

    dataset1 = {'eigenfreq': Z_target1}
    dataset2 = {'eigenfreq': Z_target2}

    data_path = prepare_plot_data(
        new_coords, data_class='Eigensolution', dataset_list=[
            dataset1,
            dataset2,
        ], fixed_params={},
        save_dir='./rsl/eigensolution',
    )
    from projects.EPBIC.plot_band_1Dspace import main

    main(data_path)

