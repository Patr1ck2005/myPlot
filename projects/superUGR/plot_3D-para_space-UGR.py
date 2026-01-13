from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458


if __name__ == '__main__':
    data_path = 'data/test.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex)

    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["a", "vert_pos_factor"]
    z_keys = ["特征频率 (THz)"]
    # z_keys = ["U_factor (1)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=True)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={

        },  # 固定
        filter_conditions={
            # "fake_factor (1)": {"<": 1},  # 筛选
            # "频率 (Hz)": {"<": 0.52, ">": 0},  # 筛选
        }
    )

    deltas = (.1, .1)  # n个维度的网格间距
    # 当沿维度 d 生长时，值差权重矩阵（n×n）
    # 例如：value_weights[d, j] = 在 grow_dir=d 时，对维度 j 的值差权重
    value_weights = np.array([
        [1, 1],   # 沿维度0生长时，对 0,1,2 维度的值差权重
        [1, 1],   # 沿维度2生长时
    ])

    # 当沿维度 d 生长时，导数不连续权重矩阵（n×n）
    deriv_weights = np.array([
        [0, 0],
        [0, 0],
    ])
    # 创建一个新的数组，用于存储更新后的结果
    Z_new = np.empty_like(Z, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z_new[i, j] = Z[i, j][0]  # 提取每个 lst_ij 的第 b 列

    Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_solution(
        grid_coords, Z_grouped,
        freq_index=6  # 第n个频率
    )
    _, Z_target2 = group_solution(
        grid_coords, Z_grouped,
        freq_index=7  # 第n个频率
    )
    # new_coords, Z_target3 = group_eigensolution(
    #     grid_coords, Z_grouped,
    #     freq_index=2  # 第n个频率
    # )
    # new_coords, Z_target4 = group_eigensolution(
    #     grid_coords, Z_grouped,
    #     freq_index=3  # 第n个频率
    # )

    # 画二维曲面：a vs w1 对 Δ频率
    plot_params = {
        'zlabel': '***',
        'cmap1': 'Blues',
        'cmap2': 'Reds',
        'log_scale': False,
        'alpha': 1,
        'data_scale': [100, 10e-4, 5],
        # 'data_scale': [10000, 1, 1],
        # 'vmax_real': 95,
        # 'vmax_imag': 1,
        'render_real': True,
        'render_imag': False,
        'apply_abs': True
    }
    plot_Z_diff_pyvista(
        new_coords, [Z_target1, Z_target2],
        # new_coords, [Z_target2],
        x_key="a",
        y_key="vert_pos_factor",
        fixed_params={
        },
        plot_params=plot_params,
        show_live=True
    )
    # 画二维曲面：a vs w1 对 Δ频率
    plot_params = {
        'zlabel': '***',
        'cmap1': 'Blues',
        'cmap2': 'Reds',
        'log_scale': False,
        'alpha': 1,
        'data_scale': [100, 10e-3, 100],
        # 'data_scale': [10000, 1, 1],
        # 'vmax_real': 95,
        # 'vmax_imag': 1,
        'render_real': False,
        'render_imag': True,
        'apply_abs': True
    }
    plot_Z_diff_pyvista(
        new_coords, [Z_target1, Z_target2],
        # new_coords, [Z_target2],
        x_key="a",
        y_key="vert_pos_factor",
        fixed_params={
        },
        plot_params=plot_params,
        show_live=True
    )
