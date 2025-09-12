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


def filter_eigensolution(grid_coords, Z, z_keys, fixed_params):


    # 2. 构造新的 coords
    new_coords = grid_coords

    # 3. 新数组的 shape
    new_shape = list(Z.shape)
    Z_new = np.zeros(shape=new_shape, dtype=object)
    min_lens = 999
    # 4. 遍历所有索引，计算差值
    #    我们需要对原始 Z 的所有索引 i1,...,iN（含 bg_n_dim）取两次值
    it = np.ndindex(*new_shape)
    for idx in it:
        idx_full_A = list(idx)

        # 取出对应的列表
        data_elements = Z[tuple(idx_full_A)]
        # 构造一个 4×10 的数组
        arr = np.array(data_elements)
        # 生成一个布尔掩码，选出第 3 行（第 4 个列表）中 >0.5 的列
        mask = arr[3] > 0.5
        mask_lens = sum(mask)
        if mask_lens < min_lens:
            min_lens = mask_lens
        # 用布尔掩码在列维度上筛选
        # filtered_arr = arr[:, mask]
        # filtered = filtered_arr.tolist()
        arr[0, mask] = -1
        filtered = arr.tolist()

        Z_new[idx] = filtered

    return new_coords, Z_new, min_lens


def advanced_filter_eigensolution(grid_coords, Z, z_keys, fixed_params=None, filter_conditions=None):
    """
    高级筛选函数：基于固定参数切片网格（移除固定参数的coords并自动缩减维度），并基于数据条件过滤每个cell的列。

    参数:
    - grid_coords: dict, 参数网格坐标，如 {"delta_shrink (nm)": array, "m1": array, "m2": array}。
    - Z: np.array(dtype=object), 数据网格，每个元素是列表。
    - z_keys: list, 数据键列表，如 ["特征频率 (THz)", "tanchi (1)", "phi (rad)", "S_air_prop (1)"]。
    - fixed_params: dict, 如 {"delta_shrink (nm)": 60}，用于切片网格并移除对应coords。
    - filter_conditions: dict, 如 {"S_air_prop (1)": {">": 1.0}}，支持运算符 >, <, ==, >=, <=。

    返回:
    - new_coords: dict，仅包含未固定的参数坐标。
    - Z_new: np.array(dtype=object)，维度缩减后的数组，每个元素是筛选后的列列表。
    - min_lens: 最小剩余列数。
    """
    if fixed_params is None:
        fixed_params = {}
    if filter_conditions is None:
        filter_conditions = {}

    # 1. 初始化
    param_keys = list(grid_coords.keys())  # e.g., ["delta_shrink (nm)", "m1", "m2"]
    new_coords = {k: v for k, v in grid_coords.items() if k not in fixed_params}  # 直接排除固定参数的键
    Z_temp = Z.copy()
    slices = [slice(None)] * len(Z.shape)  # 初始化全切片

    # 2. 基于fixed_params切片网格（使用整数索引自动缩减维度）
    for param_key, fixed_value in fixed_params.items():
        if param_key not in param_keys:
            raise ValueError(f"固定参数 {param_key} 不在网格参数中")
        dim_idx = param_keys.index(param_key)
        coord_values = grid_coords[param_key]  # 用原grid_coords查找
        match_indices = np.where(coord_values == fixed_value)[0]
        if len(match_indices) == 0:
            raise ValueError(f"在 {param_key} 中未找到值 {fixed_value}")
        slices[dim_idx] = match_indices[0]  # 取第一个匹配的整数索引（假设唯一）

    # 应用切片到Z_temp（整数切片会自动移除固定维度）
    Z_temp = Z_temp[tuple(slices)]
    # 无需squeeze：numpy已自动处理维度减少

    # 3. 新数组的shape和Z_new
    new_shape = Z_temp.shape
    Z_new = np.empty(new_shape, dtype=object)  # 用empty更适合object数组
    min_lens = float('inf')

    # 4. 遍历所有索引，应用filter_conditions
    it = np.ndindex(*new_shape)
    for idx in it:
        data_elements = Z_temp[idx]
        if not isinstance(data_elements, list) or len(data_elements) == 0:
            Z_new[idx] = []
            continue

        # 转为arr: shape (len(z_keys), N)，处理complex
        arr = np.array(data_elements, dtype=object)  # 保持object以支持complex

        # 生成整体mask：所有条件的AND（假设阈值针对real部分）
        if arr.size > 0 and arr.shape[1] > 0:  # 避免空arr
            mask = np.ones(arr.shape[1], dtype=bool)
            for z_key, cond_dict in filter_conditions.items():
                if z_key not in z_keys:
                    raise ValueError(f"过滤键 {z_key} 不在z_keys中")
                row_idx = z_keys.index(z_key)
                row_data = np.array(
                    [np.real(item) if np.iscomplexobj(item) else item for item in arr[row_idx]])  # 统一取real
                for op, value in cond_dict.items():
                    if op == ">":
                        mask &= (row_data > value)
                    elif op == "<":
                        mask &= (row_data < value)
                    elif op == "==":
                        mask &= (row_data == value)
                    elif op == ">=":
                        mask &= (row_data >= value)
                    elif op == "<=":
                        mask &= (row_data <= value)
                    else:
                        raise ValueError(f"不支持的运算符 {op}")
        else:
            mask = np.array([], dtype=bool)

        # 过滤arr
        if arr.shape[1] > 0:
            filtered_arr = arr[:, mask]
            mask_lens = sum(mask)
        else:
            filtered_arr = np.empty((len(z_keys), 0), dtype=object)
            mask_lens = 0
        if mask_lens < min_lens:
            min_lens = mask_lens

        # 转为列表
        Z_new[idx] = filtered_arr.tolist()

    if min_lens == float('inf'):
        min_lens = 0

    return new_coords, Z_new, min_lens


if __name__ == '__main__':
    data_path = './data/VBG/k_space-lossy_material.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex)

    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["delta_shrink (nm)", "m1"]
    z_keys = ["特征频率 (THz)", "tanchi (1)", "phi (rad)", "S_air_prop (1)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=True)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    # 示例查询某个参数组合对应的数据
    query = {"delta_shrink (nm)": 60, "m1": 0.00}
    result = query_data_grid(grid_coords, Z, query)
    print("\n查询结果（保留列表）：", result)

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={"delta_shrink (nm)": 60},  # 固定 delta_shrink=60
        filter_conditions={"S_air_prop (1)": {"<": 100.0}}  # 筛选 S_air_prop < 10.0
    )

    deltas3 = (.1, .1)  # n个维度的网格间距
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
    Z_new = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        Z_new[i] = Z_filtered[i][0]  # 提取每个 lst_ij 的第 b 列

    Z_grouped = group_surfaces_one_sided_hungarian(
        Z_new, deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=0  # 第n个频率
    )
    new_coords, Z_target2 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=0  # 第n个频率
    )
    new_coords, Z_target3 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=0  # 第n个频率
    )
    new_coords, Z_target4 = group_eigensolution(
        new_coords, Z_grouped,
        freq_index=0  # 第n个频率
    )

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
        # fixed_params={
        # },
        plot_params={
            'zlabel': '***',
            'imag': False,
        },
        show=True,
    )
