from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.plot_3D_params_space_plt import *
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

if __name__ == '__main__':
    # data_path = 'data/2fold-TE-eigen.csv'
    # data_path = 'data/SE/2fold-TE-k_loss0-eigen.csv'
    # data_path = 'data/SE/3fold-TE-delta0.1-k_loss0-eigen.csv'
    data_path = './data/3fold-TE-delta0.2-eigen.csv'
    # data_path = 'data/3fold-TE-delta_spcae-eigen.csv
    # data_path = './data/1fold-TM-k_loss0-eigen.csv'
    # data_path = 'data/1fold_weak-TM-eigen.csv'
    # data_path = './data/1fold-TM-eigen.csv'
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
    # param_keys = ["m1", "m2", "loss_k", "w_delta_factor"]
    param_keys = ["m1", "m2", "loss_k"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "fake_factor (1)", "频率 (Hz)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    # 示例查询某个参数组合对应的数据
    query = {"m1": 0.00, "m2": 0.00}
    result = query_data_grid(grid_coords, Z, query)
    print("\n查询结果（保留列表）：", result)

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={"m2": 0, "loss_k": 0},  # 固定
        # fixed_params={"m1": 0, "m2": 0, "loss_k": 1e-3*0},  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 100.0},  # 筛选
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
        Z_new[i] = Z_filtered[i][1]  # 提取每个 lst_ij 的第 b 列
        Z_new[i] = np.log10(Z_filtered[i][1])  # 提取每个 lst_ij 的第 b 列

    Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=14
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_solution(
        new_coords, Z_grouped,
        freq_index=6  # 第n个频率
    )
    new_coords, Z_target2 = group_solution(
        new_coords, Z_grouped,
        freq_index=8  # 第n个频率
    )

    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")

    data_path = prepare_plot_data(
        new_coords, [Z_target1, Z_target2], x_key="m1", fixed_params={},
        save_dir='./rsl/eigensolution',
    )

    from plot_3D.projects.SE.plot_thickband import main

    main(data_path)

