import pickle
import time
import numpy as np
import os
from .utils import safe_str  # 假设你有这个工具函数


def prepare_plot_data(coords, dataset_list, fixed_params=None, save_dir='./rsl/', save_manual_name=None):
    """
    生成纯净绘图数据并保存。
    返回: str, 保存的数据文件路径。
    """
    fixed_params = fixed_params or {}

    # Step 4: 创建 PlotData 对象 (规范结构)
    plot_data = {
        'coords': coords,
        'data_list': dataset_list,
        'metadata': {
            'fixed_params': fixed_params,
            'timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'version': '2.0',
        }
    }

    # Step 5: 保存到规范文件夹
    timestamp = plot_data['metadata']['timestamp']
    data_dir = os.path.join(save_dir, f"{timestamp}")
    os.makedirs(data_dir, exist_ok=True)

    # 生成唯一文件名
    param_str = '_'.join([f"{k}-{safe_str(v)}" for k, v in sorted(fixed_params.items())])
    filename = f"plot_data_{param_str}.pkl"
    if len(filename) > 200:
        filename = filename[:200] + ".pkl"
    if save_manual_name:
        filename = f"{save_manual_name}.pkl"
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


