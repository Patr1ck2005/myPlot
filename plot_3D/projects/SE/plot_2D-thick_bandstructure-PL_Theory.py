from plot_3D.core.data_postprocess.data_filter import advanced_filter_eigensolution
from plot_3D.core.data_postprocess.data_grouper import *
from plot_3D.core.plot_3D_params_space_plt import *
from plot_3D.core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from plot_3D.core.prepare_plot import prepare_plot_data
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
    # data_path = 'data/SE/2fold-TE-k_loss0-eigen.csv'
    # data_path = 'data/SE/3fold-TE-delta0.1-k_loss0-eigen.csv'
    # data_path = 'data/SE/3fold-TE-delta0.2-eigen.csv'
    # data_path = 'data/3fold-TE-delta_spcae-eigen.csv
    # data_path = './data/1fold-TM-k_loss0-eigen.csv'
    # data_path = 'data/SE/1fold_weak-TM-eigen.csv'
    data_path = './data/1fold-TM-eigen.csv'
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
            "频率 (Hz)": {">": 0.3, "<": 1.0},  # 筛选
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

    Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=10
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=3  # 第n个频率
    )
    new_coords, Z_target2 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=4  # 第n个频率
    )
    new_coords, Z_target3 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=5  # 第n个频率
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
    # 集成保存
    # data_path = prepare_plot_data(
    #     new_coords, [Z_target1], x_key="w_delta_factor", fixed_params={},
    #     save_dir='./rsl/eigensolution',
    # )

    data_path = prepare_plot_data(
        new_coords, [Z_target1, Z_target2], x_key="m1", fixed_params={},
        save_dir='./rsl/eigensolution',
    )

    from plot_3D.projects.SE.plot_thickband import main

    main(data_path)

    # plot_Z(new_coords,
    #        [Z_target1],
    #        # x_key="m1",
    #        x_key="w_delta_factor",
    #        plot_params={
    #            'figsize': (3.5, 2.5),
    #            'zlabel': "", 'xlabel': r"$\delta$",
    #            'alpha': 1.0,
    #            # 'default_color': 'black',
    #            'default_color_list': ['red'],
    #            'title': False, 'legend': False,
    #            'log_scale': True,
    #            'advanced_process': None
    #        },
    #        fixed_params={}, show=True)

    # plot_Z(new_coords,
    #        [Z_target2, Z_target1],
    #        x_key="m1",
    #        plot_params={
    #            'figsize': (5, 1.5),
    #            'zlabel': "f (c/P)", 'xlabel': r"k ($2\pi/P$)",
    #            'title': False, 'legend': False,
    #     #            'log_scale': True,
    #     #             'alpha': 1.0,
    #            # 'default_color': 'black',
    #            'default_color_list': ['red', 'blue'],
    #           'advanced_process': 'y_mirror'
    #        },
    #        # plot_params={
    #        #     'figsize': (3, 3),
    #        #     'zlabel': "freq (c/P)", 'xlabel': r"k ($2\pi/P$)",
    #        #     'enable_fill': True, 'enable_line_fill': True, 'cmap': 'magma',
    #        #     'add_colorbar': False,
    #        #     "global_color_vmin": 0, "global_color_vmax": 0.02, "default_color": 'gray', 'legend': False,
    #        #     'alpha_fill': 0.5,
    #        #     'edge_color': 'none', 'title': False,
    #        #     'scale': 1,
    #        # },
    #        # plot_params={
    #        #     'figsize': (3, 4),
    #        #     'zlabel': "f (c/P)", 'xlabel': r"k ($2\pi/P$)",
    #        #     'enable_fill': True, 'gradient_fill': True, 'gradient_direction': 'z3', 'cmap': 'magma', 'add_colorbar': False,
    #        #     "global_color_vmin": 0, "global_color_vmax": 5e-3, "default_color": 'gray', 'legend': False, 'alpha_fill': 1,
    #        #     'edge_color': 'none', 'title': False, 'scale': 1,
    #        # },
    #        fixed_params={}, show=True)

