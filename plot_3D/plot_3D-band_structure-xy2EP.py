from plot_3D.core.data_postprocess.data_grouper import group_surfaces_one_sided_hungarian
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
    Z_diff = np.zeros(shape=new_shape, dtype=complex)

    # 4. 遍历所有索引，计算差值
    #    我们需要对原始 Z 的所有索引 i1,...,iN（含 bg_n_dim）取两次值
    it = np.ndindex(*new_shape)
    for idx in it:
        # 将去掉 bg_n_dim 的 idx 扩回到原始维度
        idx_full_A = list(idx)
        idx_full_B = list(idx)

        # 取出对应的列表
        list_A = Z[tuple(idx_full_A)]
        list_B = Z[tuple(idx_full_B)]

        # 检查长度
        if len(list_A) <= freq_index or len(list_B) <= freq_index:
            raise IndexError(f"在索引 {idx} 处，列表长度不足以取到第 {freq_index + 1} 个元素")

        Z_diff[idx] = list_A[freq_index]

    return new_coords, Z_diff


if __name__ == '__main__':
    # data_path = './data/expanded-xy2EP-BIC-test.csv'
    data_path = './data/xy2EP-BIC-test.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex)

    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2"]
    # z_key = "特征频率 (THz)"
    # z_key = "phi (rad)"
    z_keys = ["特征频率 (THz)", "tanchi (1)", "phi (rad)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=True)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    for m1 in grid_coords['m1']:
        for m2 in grid_coords['m2']:
            # print(f"m1={m1}, m2={m2}")
            # print(Z[(m1, m2)])
            pass

    # 示例查询某个参数组合对应的数据
    query = {"m1": 0.00, "m2": 0.00}
    result = query_data_grid(grid_coords, Z, query)
    print("\n查询结果（保留列表）：", result)
    # 示例查询某个参数组合对应的数据
    query = {"m1": 0.00, "m2": 0.01}
    result = query_data_grid(grid_coords, Z, query)
    print("\n查询结果（保留列表）：", result)

    deltas3 = (1.0, 1.0)  # n个维度的网格间距
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

    Z = group_surfaces_one_sided_hungarian(
        Z_new, deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_eigensolution(
        grid_coords, Z,
        freq_index=5  # 第n个频率
    )
    new_coords, Z_target2 = group_eigensolution(
        grid_coords, Z,
        freq_index=6  # 第n个频率
    )
    # new_coords, Z_target3 = group_eigensolution(
    #     grid_coords, Z,
    #     freq_index=6  # 第n个频率
    # )
    # new_coords, Z_target3 = group_eigensolution(
    #     grid_coords, Z,
    #     freq_index=2  # 第n个频率
    #     # freq_index=2  # 第n个频率
    # )

    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")
    print("Z_diff 形状：", Z_target1.shape)
    # # 示例查询某个参数组合对应的数据
    # query = {"a": 0.00, "b": 0.0000}
    # result = query_data_grid(grid_coords, Z_target1, query)
    # print("\n差值查询结果（保留列表）：", result)

    # 假设已经得到 new_coords, Z_target
    # 画一维曲线：params 对 target
    plot_Z(
        new_coords, Z_target1,
        x_key="m1",
        fixed_params={
            "m2": 0.0000
        },
        plot_params={
            'zlabel': '***',
            'imag': False,
        }
    )
    # 画二维曲面：a vs w1 对 Δ频率
    plot_params = {
        'zlabel': '***',
        'cmap1': 'Blues',
        'cmap2': 'Reds',
        'log_scale': False,
        'alpha': 1,
        'data_scale': [100, 2e-3, 100],
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
        x_key="m1",
        y_key="m2",
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
        x_key="m1",
        y_key="m2",
        fixed_params={
        },
        plot_params=plot_params,
        show_live=True
    )
