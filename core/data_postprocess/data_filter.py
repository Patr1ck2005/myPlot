import numpy as np


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
        match_indices = np.where(np.isclose(coord_values, fixed_value))[0]
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