from plot_3D.core.plot_3D_params_space import plot_Z_diff
from plot_3D.core.process_multi_dim_params_space import *

import numpy as np

c_const = 299292458

def compute_bg_n_difference(grid_coords, Z, bg_n_key="bg_n", freq_index=1):
    """
    针对最后一维（bg_n）做差值计算：
    X = (Z[..., 1] at bg_n=A) - (Z[..., 1] at bg_n=B)

    参数:
        grid_coords: dict, 原始每个参数的唯一取值数组
        Z: np.ndarray, dtype=object, 最后一维是 bg_n，不同 bg_n 对应不同列表
        bg_n_key: str, 在 grid_coords 中对应折射率维度的 key
        freq_index: int, 选择每个列表中的第几个频率（0-base），默认 1 即第二个

    返回:
        new_coords: dict, 去掉 bg_n_key 后的坐标字典
        Z_diff: np.ndarray, 新的多维数组，shape = 原始 shape 去掉 bg_n 维度
                每个元素是两个频率的差值 (float)
    """
    # 1. 找到 bg_n 维度的位置和长度
    keys = list(grid_coords.keys())
    bg_n_dim = keys.index(bg_n_key)
    bg_n_vals = grid_coords[bg_n_key]
    if len(bg_n_vals) != 2:
        raise ValueError(f"{bg_n_key} 维度长度应为 2，但现在是 {len(bg_n_vals)}")

    # 2. 构造新的 coords（去掉 bg_n）
    new_coords = {k: v for k, v in grid_coords.items() if k != bg_n_key}

    # 3. 新数组的 shape
    new_shape = list(Z.shape)
    del new_shape[bg_n_dim]
    Z_diff = np.zeros(shape=new_shape, dtype=complex)

    # 4. 遍历所有索引，计算差值
    #    我们需要对原始 Z 的所有索引 i1,...,iN（含 bg_n_dim）取两次值
    it = np.ndindex(*new_shape)
    for idx in it:
        # 将去掉 bg_n_dim 的 idx 扩回到原始维度
        idx_full_A = list(idx)
        idx_full_B = list(idx)
        # 插入 bg_n 的两个位置：A 在 0，B 在 1
        idx_full_A.insert(bg_n_dim, 0)
        idx_full_B.insert(bg_n_dim, 1)

        # 取出对应的列表
        list_A = Z[tuple(idx_full_A)]
        list_B = Z[tuple(idx_full_B)]

        # 检查长度
        if len(list_A) <= freq_index or len(list_B) <= freq_index:
            raise IndexError(f"在索引 {idx} 处，列表长度不足以取到第 {freq_index + 1} 个元素")

        # 计算差值
        val_A = 1e9 * c_const / list_A[freq_index] / 1e12
        val_B = 1e9 * c_const / list_B[freq_index] / 1e12

        diff_wavelength_nm = val_A - val_B
        Z_diff[idx] = diff_wavelength_nm

    return new_coords, Z_diff


if __name__ == '__main__':
    data_path = './data/3EP-test.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex)

    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["w1 (nm)", "buffer (nm)", "h_grating (nm)", "a", "bg_n"]
    z_key = "特征频率 (THz)"

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_key)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)
    print("数据网格 Z：\n", Z)

    # 示例查询某个参数组合对应的数据
    query = {"w1 (nm)": 215.00, "buffer (nm)": 1345.0, "h_grating (nm)": 113.50, "a": 0.0085000, "bg_n": 1.3330}
    result = query_data_grid(grid_coords, Z, query)
    print("\n查询结果（保留列表）：", result)

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_diff = compute_bg_n_difference(
        grid_coords, Z,
        bg_n_key="bg_n",
        freq_index=0  # 第n个频率
        # freq_index=3  # 第n个频率
    )

    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")
    print("Z_diff 形状：", Z_diff.shape)
    print("示例差值：", Z_diff.flatten()[:10])

    # 假设已经得到 new_coords, Z_diff
    # 画一维曲线：w1 对 Δ频率
    plot_Z_diff(
        new_coords, Z_diff,
        x_key="a",
        fixed_params={
            "buffer (nm)": 1347.0,
            "h_grating (nm)": 115.5,
            "w1 (nm)": 219.
        },
        plot_params={
            'zlabel': 'RIU'
        }
    )

    # 画二维曲面：w1 vs buffer 对 Δ频率
    plot_Z_diff(
        new_coords, Z_diff,
        x_key="a",
        y_key="w1 (nm)",
        fixed_params={
            "h_grating (nm)": 115.5,
            "buffer (nm)": 1347.0,
        },
        plot_params={
            'zlabel': r'$log_{10}RIU$',
            'cmap1': 'magma',
            'cmap2': 'Blues',
            'log_scale': True,
        }
    )