from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.plot_3D_params_space_plt import *
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/FP_PhC-diff_FP-detailed-14eigenband-450P-200T-0.35r-0.1k.csv'
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


    period = 450
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=period*1e-9*1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period*1e-9)
    df_sample["k"] = df_sample["m1"]+df_sample["m2"]/2.414
    # 识别s和p偏振
    df_sample["phi (rad)"] = df_sample["phi (rad)"].apply(lambda x: x % np.pi)
    df_sample["sp_polar_show"] = recognize_sp(df_sample["phi (rad)"], df_sample["m1"], df_sample["m2"])
    # # 筛选m1<0.1的成分
    # df_sample = df_sample[df_sample["m1"] < 0.3]
    # 指定用于构造网格的参数以及目标数据列
    # param_keys = ["k", "buffer (nm)", "sp_polar_show"]
    param_keys = ["k", "buffer (nm)"]
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
            'buffer (nm)': 235,
            # "sp_polar_show": 1,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            "频率 (Hz)": {"<": 0.59, ">": 0.50},  # 筛选
        }
    )

    deltas3 = (1e-3,)  # n个维度的网格间距
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

    # 通过散点的方式绘制出来，看看效果
    for i in range(Z_new.shape[0]):
        z_vals = Z_new[i]
        for val in z_vals:
            if val is not None:
                plt.scatter(new_coords['k'][i], np.real(val), color='blue', s=10)
    plt.xlabel('k')
    plt.ylabel('Re(eigenfreq) (THz)')
    plt.title('Filtered Eigenfrequencies before Grouping')
    plt.grid(True)
    plt.show()

    Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=4,
        auto_split_streams=False,
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_solution(
        new_coords, Z_grouped,
        freq_index=0  # 第n个频率
    )
    #################################### advanced analysis ####################################
    Z_target = Z_target1
    # extract BIC modes
    BIC_Q_threshold = 1e5
    # find peaks in Z_target1's Qfactor
    Qfactors = Z_target.real / Z_target.imag / 2
    # vis Qfactors
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.scatter(new_coords['k'], Qfactors, color='blue', s=10)
    plt.xlabel('k')
    plt.ylabel('Qfactor')
    plt.title('Qfactors of First Eigenmode')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(Qfactors, height=BIC_Q_threshold)
    print(f"Found {len(peaks)} BIC modes with Q > {BIC_Q_threshold}")
    # locate and print BIC modes' coordinates and frequencies
    k_lst = []
    f_lst = []
    for peak in peaks:
        coord = {key: new_coords[key][peak] for key in new_coords}
        complex_freq = Z_target[peak]
        Qfactor = Qfactors[peak]
        k_lst.append(coord['k'])
        f_lst.append(complex_freq.real)
        print(f"BIC mode at {coord}, frequency: {complex_freq}, Qfactor: {Qfactor}")
    aniso_f_factor = (f_lst[2] - f_lst[1]) / (f_lst[0] - f_lst[1])
    aniso_k_factor = k_lst[2] / k_lst[0]
    print("Anisotropy factors:" + "f_factor={:.2f}, k_factor={:.2f}".format(aniso_f_factor, aniso_k_factor))
    #################################### advanced analysis ####################################
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
    # new_coords, Z_target5 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=4  # 第n个频率
    # )
    # new_coords, Z_target6 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=5  # 第n个频率
    # )
    # new_coords, Z_target7 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=6  # 第n个频率
    # )
    # new_coords, Z_target8 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=7  # 第n个频率
    # )
    # new_coords, Z_target9 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=8  # 第n个频率
    # )
    # new_coords, Z_target10 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=9  # 第n个频率
    # )
    # new_coords, Z_target11 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=10  # 第n个频率
    # )

    dataset1 = {'eigenfreq': Z_target1}
    dataset2 = {'eigenfreq': Z_target2}
    dataset3 = {'eigenfreq': Z_target3}
    dataset4 = {'eigenfreq': Z_target4}
    # dataset5 = {'eigenfreq': Z_target5}
    # dataset6 = {'eigenfreq': Z_target6}
    # dataset7 = {'eigenfreq': Z_target7}
    # dataset8 = {'eigenfreq': Z_target8}
    # dataset9 = {'eigenfreq': Z_target9}
    # dataset10 = {'eigenfreq': Z_target10}
    # dataset11 = {'eigenfreq': Z_target11}
    # dataset12 = {'eigenfreq': Z_target12}
    # dataset13 = {'eigenfreq': Z_target13}
    # dataset14 = {'eigenfreq': Z_target14}


    data_path = prepare_plot_data(
        new_coords, data_class='Eigensolution', dataset_list=[
            dataset1, dataset2, dataset3, dataset4,
            # dataset5, dataset6, dataset7, dataset8,
        ], fixed_params={},
        save_dir='./rsl/eigensolution',
    )

    from projects.MergingBICs.plot_thickband import main

    main(data_path)

