from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.plot_3D_params_space_plt import *
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/3fold-TE-DC_space-S_space-eigen.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    def norm_freq(freq, period):
        return freq/(c_const/period)
    # period = 1000 nm
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=1e3*1e-9*1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=1e3*1e-9)

    # 筛选m1<0.1的成分
    df_sample = df_sample[df_sample["m1"] < 0.3]

    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2", "loss_k", "w_delta_factor", "w1 (nm)"]
    # param_keys = ["m1", "m2", "loss_k", "w_delta_factor"]
    # param_keys = ["m1", "m2", "loss_k"]
    # z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "fake_factor (1)", "频率 (Hz)"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "fake_factor (1)", "频率 (Hz)"]

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
            # "m1": 0,
            "m2": 0,
            "loss_k": 0,
            "w1 (nm)": 236+8*-1,  # 0.4 S
            # "w1 (nm)": 252+8*1,  # 0.2 S
            "w_delta_factor": 0.4,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1.0},  # 筛选
            # "m1": {"<": .1},  # 筛选
            "频率 (Hz)": {">": 0.0, "<": 1.0},  # 筛选
        }
    )

    deltas3 = (1,)  # n个维度的网格间距
    # 当沿维度 d 生长时，值差权重矩阵（n×n）
    # 例如：value_weights[d, j] = 在 grow_dir=d 时，对维度 j 的值差权重
    value_weights = np.array([
        [1e8,],   # 沿维度生长时
    ])
    # 当沿维度 d 生长时，导数不连续权重矩阵（n×n）
    deriv_weights = np.array([
        [1e8,],
    ])
    # 创建一个新的数组，用于存储更新后的结果
    Z_new = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        Z_new[i] = Z_filtered[i][0]  # 提取每个 lst_ij 的第 b 列
        # Z_new[i] = np.log10(Z_filtered[i][1])  # 提取每个 lst_ij 的第 b 列

    # rename m1 to k
    new_coords['k'] = new_coords.pop('m1')

    fig, ax = plt.subplots(figsize=(8, 8))
    # 通过散点的方式绘制出来，看看效果
    for i in range(Z_new.shape[0]):
        z_vals = Z_new[i]
        for val in z_vals:
            if val is not None:
                plt.scatter(new_coords['k'][i], np.real(val), color='black', s=10)
    plt.xlabel('k')
    plt.ylabel('Re(eigenfreq) (THz)')
    plt.title('Filtered Eigenfrequencies before Grouping')
    plt.grid(True)
    plt.show()

    Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=7,
        auto_split_streams=False,
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
    # new_coords, Z_target3 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=3  # 第n个频率
    # )
    # new_coords, Z_target4 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=4  # 第n个频率
    # )

    dataset1 = {'eigenfreq': Z_target1}
    dataset2 = {'eigenfreq': Z_target2}
    # dataset3 = {'eigenfreq': Z_target3}
    # dataset4 = {'eigenfreq': Z_target4}

    data_path = prepare_plot_data(
        new_coords, data_class='Eigensolution', dataset_list=[
            dataset1, dataset2,
            # dataset3, dataset4
        ], fixed_params={},
        save_dir='./rsl/eigensolution',
    )

    from projects.SE.plot_thickband import main

    main(data_path)

