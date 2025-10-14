import pickle
import time
import numpy as np
import os
from .utils import safe_str  # 假设你有这个工具函数


def prepare_plot_data(new_coords, Z_list, x_key, y_key=None, fixed_params=None, save_dir='./rsl/',
                      advanced_process=False):
    """
    阶段1: 生成纯净绘图数据并保存。

    参数:
        new_coords: dict, 坐标字典。
        Z_list: list, 目标数据列表 (e.g., [Z_target1])。
        x_key: str, X轴键。
        y_key: str, optional, Y轴键 (None for 1D)。
        fixed_params: dict, 固定参数。
        save_dir: str, 保存目录。
        advanced_process: bool or str, e.g., 'y_mirror' for 对称处理。

    返回: str, 保存的数据文件路径。
    """
    fixed_params = fixed_params or {}
    if not isinstance(Z_list, list):
        Z_list = [Z_list]

    # Step 1: 验证输入 (从plot_Z复制)
    keys = list(new_coords.keys())
    assert x_key in keys, f"{x_key} 不在 new_coords 中"
    if y_key:
        assert y_key in keys and y_key != x_key, f"{y_key} 不在 new_coords 中或与 {x_key} 重复"
    for k in fixed_params:
        assert k in keys and k not in (x_key, y_key or x_key), f"固定参数 {k} 无效"

    # Step 2: 计算 slicer 和 subs (从plot_Z复制)
    slicer = []
    for k in keys:
        if k == x_key or k == (y_key or k):
            slicer.append(slice(None))
        else:
            val = fixed_params.get(k)
            assert val is not None, f"参数 {k} 未在 fixed_params 中指定"
            idx = np.where(new_coords[k] == val)[0]
            assert idx.size == 1, f"{k} 中未找到值 {val}"
            slicer.append(idx[0])
    slicer = tuple(slicer)
    subs = [Z[slicer] for Z in Z_list]
    x_vals = new_coords[x_key]
    y_vals = new_coords.get(y_key, None) if y_key else None

    # Step 3: 清洗数据 (从plot_Z_2D复制，处理NaN和advanced_process)
    is_1d = (y_key is None)
    if advanced_process == 'y_mirror' and not is_1d:
        x_vals = np.concatenate([-x_vals[::-1], x_vals])
        subs = [np.concatenate([sub[::-1], sub]) for sub in subs]
        y_vals = np.concatenate([-y_vals[::-1], y_vals])  # 假设y也需镜像

    # # 当前不适合处理NaN
    # cleaned_subs = []
    # if is_1d:
    #     for sub in subs:
    #         mask = np.isnan(sub)
    #         if np.any(mask):
    #             print(f"Warning: 存在非法数据，已移除。")
    #             cleaned_sub = sub[~mask]
    #             # 注意：1D下x_vals也需相应切片，但当前代码中temp_x_vals未全局应用；建议统一
    #             temp_x_vals = x_vals[~mask]
    #             # 为简单，假设所有曲线共享x_vals；若不，需存多个x
    #         else:
    #             cleaned_sub = sub
    #             temp_x_vals = x_vals
    #         cleaned_subs.append(cleaned_sub)
    #     x_vals = temp_x_vals  # 更新x_vals
    # else:
    #     cleaned_subs = subs  # 2D暂无NaN处理，可扩展

    # Step 4: 创建 PlotData 对象 (规范结构)
    plot_data = {
        'x_vals': x_vals,
        'y_vals': y_vals,
        'subs': subs,  # list of Z subs
        'is_1d': is_1d,
        'metadata': {
            'x_key': x_key,
            'y_key': y_key,
            'fixed_params': fixed_params,
            'timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'data_shape': [sub.shape for sub in subs]
        }
    }

    # Step 5: 保存到规范文件夹
    timestamp = plot_data['metadata']['timestamp']
    data_dir = os.path.join(save_dir, f"{timestamp}")
    os.makedirs(data_dir, exist_ok=True)

    # 生成唯一文件名
    param_str = '_'.join([f"{k}-{safe_str(v)}" for k, v in sorted(fixed_params.items())])
    if y_key:
        param_str += f"_x-{safe_str(x_key)}_y-{safe_str(y_key)}"
    else:
        param_str += f"_x-{safe_str(x_key)}_1d"
    filename = f"plot_data_{param_str}.pkl"
    if len(filename) > 200:
        filename = filename[:200] + ".pkl"
    file_path = os.path.join(data_dir, filename)

    with open(file_path, 'wb') as f:
        pickle.dump(plot_data, f)
    print(f"纯净绘图数据已保存为：{file_path} 🎉")
    print(f"绝对路径：{os.path.abspath(file_path)}")
    # 将数据加载到剪切板
    ...
    # 再保存一份临时数据在当前目录，方便快速访问
    temp_path = os.path.join('.', 'temp_plot_data.pkl')
    with open(temp_path, 'wb') as f:
        pickle.dump(plot_data, f)

    return file_path


