from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.process_multi_dim_params_space import *

import numpy as np

from utils.functions import ellipse2stokes

c_const = 299792458


if __name__ == '__main__':
    # data_path = 'data/Hex-simple_annular-full-QBIC-0.03.csv'
    # data_path = 'data/Hex-simple_annular-full-QBIC-0.04.csv'
    data_path = 'data/Hex-simple_annular-full-QBIC-0.05.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    def norm_freq(freq, period):
        return freq/(c_const/period)
    # period = 300
    # df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=period*1e-9*1e12)
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex)
    # df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period*1e-9)
    df_sample["phi (rad)"] = df_sample["phi (rad)"].apply(lambda x: x % np.pi)
    # # 筛选m1<0.1的成分
    # df_sample = df_sample[df_sample["m1"] < 0.05]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2"]
    # param_keys = ["m1", "m2", "buffer (nm)"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "fake_factor (1)", "频率 (Hz)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            # 'buffer (nm)': 480,
            # 'buffer (nm)': 410,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            # "频率 (Hz)": {">": 280e12},  # 筛选
            "频率 (Hz)": {">": 400e12},  # 筛选
        }
    )

    # 测试3D散点绘图, 颜色映射取决于虚部
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 12))
    xs = []
    ys = []
    zs = []
    colors = []
    for i, m1 in enumerate(new_coords['m1']):
        for j, m2 in enumerate(new_coords['m2']):
            lst_ij = Z_filtered[i][j]
            for freq in lst_ij[0]:
                print(freq)
                xs.append(m1)
                ys.append(m2)
                zs.append(freq.real)
                colors.append(freq.imag)
    sc = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o', alpha=0.8, s=1)
    # set aspect
    ax.set_box_aspect([1,1,3])
    # set view angle
    ax.view_init(elev=15, azim=45)
    plt.colorbar(sc, label='Imaginary Part of Frequency (THz)')
    ax.set_xlabel('m1')
    ax.set_ylabel('m2')
    ax.set_zlabel('Frequency (THz)')
    plt.title('3D Scatter Plot of Eigenfrequencies')
    plt.show()

    deltas3 = (1e-3, 1e-3)  # n个维度的网格间距
    # 当沿维度 d 生长时，值差权重矩阵（n×n）
    # 例如：value_weights[d, j] = 在 grow_dir=d 时，对维度 j 的值差权重
    value_weights = np.array([
        [1, 1], [1, 1]   # 沿维度生长时
    ])
    # 当沿维度 d 生长时，导数不连续权重矩阵（n×n）
    deriv_weights = np.array([
        [1, 1], [1, 1]
    ])
    # 创建一个新的数组，用于存储更新后的结果
    Z_new = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        for j in range(Z_filtered.shape[1]):
            Z_new[i, j] = Z_filtered[i][j][0]  # 提取每个 lst_ij 的第 b 列

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas3,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=15,
        auto_split_streams=True
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_solution(
        new_coords, Z_grouped,
        freq_index=0  # 第n个频率
    )
    new_coords, Z_target2 = group_solution(
        new_coords, Z_grouped,
        freq_index=1  # 第n个频率
    )
    new_coords, Z_target3 = group_solution(
        new_coords, Z_grouped,
        freq_index=2  # 第n个频率
    )
    new_coords, Z_target4 = group_solution(
        new_coords, Z_grouped,
        freq_index=3  # 第n个频率
    )
    new_coords, Z_target5 = group_solution(
        new_coords, Z_grouped,
        freq_index=4  # 第n个频率
    )
    new_coords, Z_target6 = group_solution(
        new_coords, Z_grouped,
        freq_index=5  # 第n个频率
    )
    new_coords, Z_target7 = group_solution(
        new_coords, Z_grouped,
        freq_index=6  # 第n个频率
    )
    new_coords, Z_target8 = group_solution(
        new_coords, Z_grouped,
        freq_index=7  # 第n个频率
    )
    new_coords, Z_target9 = group_solution(
        new_coords, Z_grouped,
        freq_index=8  # 第n个频率
    )
    new_coords, Z_target10 = group_solution(
        new_coords, Z_grouped,
        freq_index=9  # 第n个频率
    )
    new_coords, Z_target11 = group_solution(
        new_coords, Z_grouped,
        freq_index=10  # 第n个频率
    )
    new_coords, Z_target12 = group_solution(
        new_coords, Z_grouped,
        freq_index=11  # 第n个频率
    )

    from core.process_multi_dim_params_space import extract_basic_analysis_fields, plot_advanced_surface
    import matplotlib.pyplot as plt
    from core.plot_cls import MomentumSpaceEigenPolarizationPlotter
    from core.plot_workflow import PlotConfig
    from core.prepare_plot import prepare_plot_data
    from core.data_postprocess.data_package import package_stad_data

    _, dataset1 = package_stad_data(new_coords, 0, Z_target1, additional_Z_grouped, z_keys)
    _, dataset2 = package_stad_data(new_coords, 1, Z_target2, additional_Z_grouped, z_keys)
    _, dataset3 = package_stad_data(new_coords, 2, Z_target3, additional_Z_grouped, z_keys)
    _, dataset4 = package_stad_data(new_coords, 3, Z_target4, additional_Z_grouped, z_keys)
    _, dataset5 = package_stad_data(new_coords, 4, Z_target5, additional_Z_grouped, z_keys)
    _, dataset6 = package_stad_data(new_coords, 5, Z_target6, additional_Z_grouped, z_keys)
    _, dataset7 = package_stad_data(new_coords, 6, Z_target7, additional_Z_grouped, z_keys)
    _, dataset8 = package_stad_data(new_coords, 7, Z_target8, additional_Z_grouped, z_keys)
    _, dataset9 = package_stad_data(new_coords, 8, Z_target9, additional_Z_grouped, z_keys)
    _, dataset10 = package_stad_data(new_coords, 9, Z_target10, additional_Z_grouped, z_keys)
    _, dataset11 = package_stad_data(new_coords, 10, Z_target11, additional_Z_grouped, z_keys)
    full_coords, dataset12 = package_stad_data(new_coords, 11, Z_target12, additional_Z_grouped, z_keys)
    data_path = prepare_plot_data(
        coords=full_coords, data_class='Eigensolution', dataset_list=[dataset1, dataset2, dataset3, dataset4, dataset5,
                                          dataset6, dataset7, dataset8, dataset9, dataset10, dataset11, dataset12], fixed_params={},
    )

    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.25, 1.25), tick_direction='in')
    plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()

    BAND_INDEX = 5

    plotter.plot_skyrmion_analysis(index=BAND_INDEX)

    plotter.new_3d_fig(temp_figsize=(3, 3))
    rgba = plotter.get_advanced_color_mapping(index=BAND_INDEX)
    plotter.plot_3D_surface(index=BAND_INDEX, rbga=rgba, shade=False)
    # plotter.plot_3D_surface(index=8)
    plotter.add_annotations()
    plotter.ax.set_zlim(450, 460)
    plotter.save_and_show()

