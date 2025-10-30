from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.plot_3D_params_space_plt import *
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

if __name__ == '__main__':
    # data_path = 'data/eigen-Rect-18ACSNano-0.3kr.csv'
    # data_path = 'data/1fold-PCS_Rod-25NanoLetters-eigen.csv'
    # data_path = 'data/1fold-PCS_Rod-25NanoLetters-BIC-eigen.csv'
    # data_path = 'data/1fold-PCS_FP-2EP-eigen.csv'
    data_path = 'data/1fold-PCS_FP-lowQ-eigen.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    def norm_freq(freq, period):
        return freq/(c_const/period)
    period = 1300
    # period = 560
    # period = 700
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=period*1e-9*1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period*1e-9)
    # df_sample["k"] = df_sample.apply(lambda row: row["k_r"] if row["k_azimu"] == 0 else -row["k_r"], axis=1)
    df_sample["k"] = df_sample["m1"]+df_sample["m2"]/2.414
    # # 筛选m1<0.1的成分
    # df_sample = df_sample[df_sample["m1"] < 0.3]

    # 指定用于构造网格的参数以及目标数据列
    # param_keys = ["k_r", "k_azimu"]
    param_keys = ["k", "buffer (nm)"]
    # z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "S_pml_prop (1)", "频率 (Hz)"]
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
            'buffer (nm)': 875,
        },  # 固定
        # fixed_params={"m1": 0, "m2": 0, "loss_k": 1e-3*0},  # 固定
        filter_conditions={
            # "S_pml_prop (1)": {"<": 1},  # 筛选
            "fake_factor (1)": {"<": 1},  # 筛选
            # "m1": {"<": .1},  # 筛选
            # "频率 (Hz)": {">": 0.0, "<": 0.58},  # 筛选
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
    plt.ylabel('Re(特征频率) (THz)')
    plt.title('聚类前的频率分布')
    plt.grid(True)
    plt.show()

    Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=7
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
    new_coords, Z_target6 = group_solution(
        new_coords, Z_grouped,
        freq_index=5  # 第n个频率
    )
    new_coords, Z_target7 = group_solution(
        new_coords, Z_grouped,
        freq_index=6  # 第n个频率
    )
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


    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")

    data_path = prepare_plot_data(
        new_coords, [
            Z_target1,
            Z_target2,
            Z_target3,
            Z_target4,
            Z_target5,
            Z_target6,
            # Z_target7,
            # Z_target8,
            # Z_target9,
            # Z_target10,
            # Z_target11,
        ], x_key="k", fixed_params={},
        save_dir='./rsl/eigensolution',
    )

    from plot_3D.projects.SE.plot_thickband import main

    main(data_path)

