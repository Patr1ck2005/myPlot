from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.plot_3D_params_space_plt import *
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *
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
    # data_path = 'data/1fold-TM-loc_0.25w-SE.csv'
    # data_path = 'data/3fold-TE-loc_0-SE.csv'
    # data_path = 'data/3fold-TE-loc_BIC-SE.csv'
    # data_path = 'data/1fold_weak-TM-mid-SE.csv'
    data_path = 'data/1fold_weak-TM-right_0.2P-SE.csv'
    df_sample = pd.read_csv(data_path, sep='\t', comment='%')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    def norm_freq(freq, period):
        return freq/(c_const/period)
    # period = 1000 nm
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=1e3*1e-9)

    # 筛选m1<*的成分
    df_sample = df_sample[df_sample["m1"] < 0.3]

    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2", "loss_k", "频率 (Hz)"]
    z_keys = ["emission_power (W/m)", "total_power (W/m)"]

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
        fixed_params={"m2": 0, "loss_k": 1e-3*1},  # 固定
        # fixed_params={"m1": 0, "m2": 0, "loss_k": 1e-3},  # 固定
        filter_conditions={
        }
    )

    # 创建一个新的数组，用于存储更新后的结果
    Z_new = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        for j in range(Z_filtered.shape[1]):
            Z_new[i, j] = Z_filtered[i, j][0]  # 提取每个 lst_ij 的第 . 列

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_eigensolution(
        new_coords, Z_new,
        freq_index=0  # 第n个数据
    )


    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")
    print("Z 形状：", Z_new.shape)

    # 假设已经得到 new_coords, Z_target
    # 画一维曲线：params 对 target

    # 集成保存
    data_path = prepare_plot_data(
        new_coords, data_class='Spectrum', dataset_list=[Z_target1], fixed_params={},
        save_dir='./rsl/k_space',
    )

    # from plot_heatmap_k_space import main
    # main(data_path)

