from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.plot_3D_params_space_plt import *
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.process_multi_dim_params_space import *

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
    # data_path = './data/3EP-test.csv'
    data_path = 'data/3EP-topological_band.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex)

    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["a", "b"]
    z_keys = ["特征频率 (THz)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            # "sp_polar_show": 1,
        },  # 固定
        filter_conditions={
            # "fake_factor (1)": {"<": 1},  # 筛选
            # "频率 (Hz)": {"<": 0.52, ">": 0},  # 筛选
        }
    )

    deltas3 = (1e-3, 1e-3)  # n个维度的网格间距
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
        for j in range(Z_filtered.shape[0]):
            Z_new[i, j] = Z_filtered[i][j][0]

    # 通过散点的方式绘制出来，看看效果
    for i in range(Z_new.shape[0]):
        z_vals = Z_new[i, 0]
        for val in z_vals:
            if val is not None:
                plt.scatter(new_coords['a'][i], np.real(val), color='blue', s=10)
    plt.xlabel('k')
    plt.ylabel('Re(eigenfreq) (THz)')
    plt.title('Filtered Eigenfrequencies before Grouping')
    plt.grid(True)
    plt.show()

    Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=3
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_eigensolution(
        grid_coords, Z_grouped,
        freq_index=0  # 第n个频率
        # freq_index=2  # 第n个频率
    )
    _, Z_target2 = group_eigensolution(
        grid_coords, Z_grouped,
        freq_index=1  # 第n个频率
        # freq_index=2  # 第n个频率
    )
    _, Z_target3 = group_eigensolution(
        grid_coords, Z_grouped,
        freq_index=2  # 第n个频率
        # freq_index=2  # 第n个频率
    )

    # 画二维曲面：a vs w1 对 Δ频率
    plot_params = {
        'zlabel': 'RIU',
        'cmap1': 'Blues',
        'cmap2': 'Reds',
        'log_scale': False,
        'alpha': 1,
        'data_scale': [20000, 2e-3, 100],
        # 'data_scale': [10000, 1, 1],
        # 'vmax_real': 95,
        # 'vmax_imag': 1,
        'render_real': True,
        'render_imag': False,
        'apply_abs': True
    }
    plot_Z_diff_pyvista(
        new_coords, [Z_target1, Z_target2, Z_target3],
        # new_coords, [Z_target2],
        x_key="a",
        y_key="b",
        fixed_params={
        },
        plot_params=plot_params,
        show_live=True
    )
