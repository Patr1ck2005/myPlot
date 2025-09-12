from plot_3D.core.data_postprocess.data_grouper import group_surfaces_one_sided_hungarian
from plot_3D.core.plot_3D_params_space_plt import *
from plot_3D.core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from plot_3D.core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

if __name__ == '__main__':

    def load_data(data_path):
        df_sample = pd.read_csv(data_path, sep='\t')

        # 指定用于构造网格的参数以及目标数据列
        param_keys = ["freq (THz)", "a"]
        z_keys = ["emission_power (W/m)"]

        # 构造数据网格，此处不进行聚合，每个单元格保存列表
        grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys)
        print("网格参数：")
        for key, arr in grid_coords.items():
            print(f"  {key}: {arr}")
        print("数据网格 Z 的形状：", Z.shape)

        # 创建一个新的数组，用于存储更新后的结果
        Z_new = np.empty_like(Z, dtype=float)
        # 使用直接的循环来更新 Z_new
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z_new[i, j] = Z[i, j][0][0]

        return grid_coords, Z_new

    grid_coords1, Z_new1 = load_data('data/SE/2EP-lz-pos-mid-air_loss0.001.txt')
    grid_coords2, Z_new2 = load_data('data/SE/2EP-lz-pos3.8-air_loss0.001.txt')

    grid_coords = grid_coords1
    Z_new = Z_new1 + Z_new2



    # 假设已经得到 new_coords, Z_target
    # 画一维曲线：params 对 target
    plot_Z(
        grid_coords, Z_new,
        x_key="freq (THz)",
        y_key="a",
        fixed_params={
        },
        plot_params={
            'zlabel': 'Power',
            'imag': False,
        },
        show=True,
    )
