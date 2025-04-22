from plot_3D.core.plot_3D_params_space_plt import *
from plot_3D.core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from plot_3D.core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

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

        # # 计算频率域差值
        # val_A_freqTHz = list_A[freq_index]
        # val_B_freqTHz = list_B[freq_index]
        # diff_freq_THz = val_A_freqTHz - val_B_freqTHz
        # Z_diff[idx] = diff_freq_THz

        # 计算波长域差值
        val_A_nm = 1e9 * c_const / list_A[freq_index].real / 1e12 + 1j * 1e9 * c_const / list_A[freq_index].imag / 1e12
        val_B_nm = 1e9 * c_const / list_B[freq_index].real / 1e12 + 1j * 1e9 * c_const / list_B[freq_index].imag / 1e12
        diff_wavelength_nm = val_A_nm - val_B_nm
        Z_diff[idx] = diff_wavelength_nm

        # # 计算频率域相对差值
        # val_A_freqTHz = abs(list_A[0].real-list_A[2].real) + 1j*abs(list_A[0].imag-list_A[2].imag)
        # val_B_freqTHz = abs(list_B[0].real-list_B[2].real) + 1j*abs(list_B[0].imag-list_B[2].imag)
        # diff_freq_THz = val_A_freqTHz - val_B_freqTHz
        # Z_diff[idx] = diff_freq_THz

    return new_coords, Z_diff


if __name__ == '__main__':
    # data_path = './data/3EP-test.csv'
    data_path = './data/3EP-4geo_dim_params_space.csv'
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

    # 示例查询某个参数组合对应的数据
    query = {"w1 (nm)": 210.00, "buffer (nm)": 1347.0, "h_grating (nm)": 115.50, "a": 0.0085000, "bg_n": 1.3330}
    result = query_data_grid(grid_coords, Z, query)
    print("\n查询结果（保留列表）：", result)

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_diff = compute_bg_n_difference(
        grid_coords, Z,
        bg_n_key="bg_n",
        freq_index=0  # 第n个频率
        # freq_index=2  # 第n个频率
    )

    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")
    print("Z_diff 形状：", Z_diff.shape)
    # 示例查询某个参数组合对应的数据
    query = {"w1 (nm)": 210.00, "buffer (nm)": 1347.0, "h_grating (nm)": 115.50, "a": 0.0085000}
    result = query_data_grid(grid_coords, Z_diff, query)
    print("\n差值查询结果（保留列表）：", result)

    # 假设已经得到 new_coords, Z_diff
    # 画一维曲线：params 对 Δ
    plot_Z_diff_plt(
        new_coords, Z_diff,
        x_key="a",
        fixed_params={
            "buffer (nm)": 1347.0,
            "h_grating (nm)": 115.5,
            "w1 (nm)": 210,
        },
        plot_params={
            'zlabel': 'RIU'
        }
    )
    # 画二维曲面：a vs w1 对 Δ频率
    plot_params = {
        'zlabel': 'RIU',
        'cmap1': 'Blues',
        'cmap2': 'Reds',
        'log_scale': False,
        'alpha': 1,
        'data_scale': [10000, 1e-3, 1],
        # 'data_scale': [10000, 1, 1],
        'vmax_real': 2,
        # 'vmax_imag': 1,
        'render_real': True,
        'render_imag': False,
        'apply_abs': True
    }
    plot_Z_diff_pyvista(
        new_coords, Z_diff,
        x_key="w1 (nm)",
        y_key="a",
        fixed_params={
            "h_grating (nm)": 115.5,
            "buffer (nm)": 1347.0,
        },
        plot_params=plot_params
    )
    for h_g in [113.5, 114.5, 115.5, 116.5, 117.5]:
        for buffer in [1345., 1346., 1347., 1348., 1349.]:
            plot_Z_diff_pyvista(
                new_coords, Z_diff,
                x_key="w1 (nm)",
                y_key="a",
                fixed_params={
                    "buffer (nm)": buffer,
                    "h_grating (nm)": h_g
                },
                plot_params=plot_params
            )

    new_coords_compressed, Z_diff_compressed = compress_data_axis(
        new_coords, Z_diff,
        axis_key="a",
        aggregator=lambda x, **kwargs: np.max(abs(x.real), **kwargs)+1j*np.max(abs(x.imag), **kwargs),
        selection_range=(0, 8),
    )

    # 假设已经得到 new_coords, Z_diff
    # 画一维曲线：params 对 Δ
    plot_Z_diff_plt(
        new_coords_compressed, Z_diff_compressed,
        x_key="w1 (nm)",
        fixed_params={
            "buffer (nm)": 1347.0,
            "h_grating (nm)": 115.5,
            # "a": 0.0085,
        },
        plot_params={
            'zlabel': 'RIU',
            'log_scale': False,
        }
    )
    # 画二维曲面：w1 vs h_grating 对 Δ频率
    plot_params['data_scale'][0] = 1
    for buffer in [1345., 1346., 1347., 1348., 1349.]:
        plot_Z_diff_pyvista(
            new_coords_compressed, Z_diff_compressed,
            x_key="w1 (nm)",
            y_key="h_grating (nm)",
            fixed_params={
                "buffer (nm)": buffer,
            },
            plot_params=plot_params
        )
    for h_g in [113.5, 114.5, 115.5, 116.5, 117.5]:
        plot_Z_diff_pyvista(
            new_coords_compressed, Z_diff_compressed,
            x_key="w1 (nm)",
            y_key="buffer (nm)",
            fixed_params={
                "h_grating (nm)": h_g,
            },
            plot_params=plot_params
        )
    plot_Z_diff_pyvista(
        new_coords_compressed, Z_diff_compressed,
        x_key="w1 (nm)",
        y_key="h_grating (nm)",
        fixed_params={
            "buffer (nm)": 1347.,
        },
        plot_params=plot_params,
        show_live=True,
    )