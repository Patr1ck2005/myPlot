from plot_3D.core.data_postprocess.data_filter import advanced_filter_eigensolution
from plot_3D.core.data_postprocess.data_grouper import *
from plot_3D.core.plot_3D_params_space_plt import *
from plot_3D.core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from plot_3D.core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

def group_eigensolution(grid_coords, Z, freq_index=1):

    # 2. 构造新的 coords
    new_coords = grid_coords

    # 3. 新数组的 shape
    new_shape = list(Z.shape)
    Z_new = np.zeros(shape=new_shape, dtype=complex)

    # 4. 遍历所有索引，计算差值
    #    我们需要对原始 Z 的所有索引 i1,...,iN（含 bg_n_dim）取两次值
    it = np.ndindex(*new_shape)
    for idx in it:
        idx_full_A = list(idx)

        # 取出对应的列表
        list_A = Z[tuple(idx_full_A)]

        # 检查长度
        if len(list_A) <= freq_index:
            raise IndexError(f"在索引 {idx} 处，列表长度不足以取到第 {freq_index + 1} 个元素")

        Z_new[idx] = list_A[freq_index]

    return new_coords, Z_new


if __name__ == '__main__':
    data_path = '../projects/VBG/data/k_space-lossy_material.csv'
    # data_path = './data/VBG/k_space-lossy_material-2.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    def norm_freq(freq, period):
        return freq/(c_const/period)
    def convert_special_k(k):
        return k*np.sqrt(2) if k < 0 else k
    # period = 720e-9*1.8
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=720e-9*1.8*1e12)
    df_sample["m1"] = df_sample["m1"].apply(convert_special_k)

    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "delta_shrink (nm)"]
    z_keys = ["特征频率 (THz)", "tanchi (1)", "phi (rad)", "S_air_prop (1)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=True)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    # 示例查询某个参数组合对应的数据
    query = {"m1": 0.00, "delta_shrink (nm)": 60}
    result = query_data_grid(grid_coords, Z, query)
    print("\n查询结果（保留列表）：", result)

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={"delta_shrink (nm)": 85},  # 固定 delta_shrink=85
        # fixed_params={"delta_shrink (nm)": 60},  # 固定 delta_shrink=60
        filter_conditions={
            "S_air_prop (1)": {"<": 10.0},  # 筛选 S_air_prop < 10.0
            # "频率 (Hz)": {">": 0.6e14},  # 筛选
        }
    )

    deltas3 = (1e-2,)  # n个维度的网格间距
    # 当沿维度 d 生长时，值差权重矩阵（n×n）
    # 例如：value_weights[d, j] = 在 grow_dir=d 时，对维度 j 的值差权重
    value_weights = np.array([
        [1,],   # 沿维度生长时
    ])
    # 当沿维度 d 生长时，导数不连续权重矩阵（n×n）
    deriv_weights = np.array([
        [1e-2,],
    ])
    # 创建一个新的数组，用于存储更新后的结果
    Z_new = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        Z_new[i] = Z_filtered[i][0]  # 提取每个 lst_ij 的第 b 列

    Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=6
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=0  # 第n个频率
    )
    new_coords, Z_target2 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=1  # 第n个频率
    )
    new_coords, Z_target3 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=2  # 第n个频率
    )
    new_coords, Z_target4 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=3  # 第n个频率
    )

    new_coords, Z_target5 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=4  # 第n个频率
    )

    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")
    print("Z 形状：", Z_target1.shape)
    # # 示例查询某个参数组合对应的数据
    # query = {"a": 0.00, "b": 0.0000}
    # result = query_data_grid(grid_coords, Z_target1, query)
    # print("\n差值查询结果（保留列表）：", result)

    # 假设已经得到 new_coords, Z_target
    # 画一维曲线：params 对 target

    plot_Z(new_coords,
           [Z_target1, Z_target2, Z_target3, Z_target4, Z_target5],
           x_key="m1",
           # plot_params={
           #     'zlabel': "f", 'enable_line_fill': True, 'alpha': 0.3, 'legend': False,
           #     'line_colors': ['blue', 'red', 'red', 'blue', 'red']
           # },
           # plot_params={
           #     'figsize': (3, 3),
           #     'zlabel': "freq (c/P)", 'xlabel': r"k ($2\pi/P$)",
           #     'enable_fill': True, 'enable_line_fill': True, 'cmap': 'magma',
           #     'add_colorbar': False,
           #     "global_color_vmin": 0, "global_color_vmax": 0.02, "default_color": 'gray', 'legend': False,
           #     'alpha_fill': 0.5,
           #     'edge_color': 'none', 'title': False,
           # },
           plot_params={
               'figsize': (3, 3),
               'zlabel': "freq (c/P)", 'xlabel': r"k ($2\pi/P$)",
               'enable_fill': True, 'gradient_fill': True, 'gradient_direction': 'z3', 'cmap': 'magma', 'add_colorbar': False,
               "global_color_vmin": 0, "global_color_vmax": 0.02, "default_color": 'gray', 'legend': False, 'alpha_fill': 0.5,
               'edge_color': 'none', 'title': False,
           },
           fixed_params={}, show=True)

