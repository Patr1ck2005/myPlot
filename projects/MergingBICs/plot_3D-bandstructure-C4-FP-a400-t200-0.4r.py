from matplotlib import pyplot as plt

from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/FP_PhC-full-diff_FP-detailed-14eigenband-400P-200T-0.4r-250L-0.1k.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))


    def norm_freq(freq, period):
        return freq / (c_const / period)


    period = 400
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=period * 1e-9 * 1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period * 1e-9)
    df_sample["phi (rad)"] = df_sample["phi (rad)"].apply(lambda x: x % np.pi)
    # # 筛选 m1<0.1 的成分
    # df_sample = df_sample[df_sample["m1"] < 0.05]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2", "buffer (nm)"]
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
            # 'buffer (nm)': 214,
            'buffer (nm)': 250,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            # "频率 (Hz)": {">": 0.0, "<": 0.360},  # 筛选
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
                xs.append(m1)
                ys.append(m2)
                zs.append(freq.real)
                colors.append(freq.imag)
    sc = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o', alpha=0.8, s=1)
    # set aspect
    ax.set_box_aspect([1, 1, 3])
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
        for j in range(Z_filtered.shape[1]):
            Z_new[i, j] = Z_filtered[i][j][0]  # 提取每个 lst_ij 的第 b 列

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas3,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=6,
        auto_split_streams=False
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
    # new_coords, Z_target7 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=6  # 第n个频率
    # )
    # new_coords, Z_target8 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=7  # 第n个频率
    # )
    # new_coords, Z_target9 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=8  # 第n个频率
    # )
    # new_coords, Z_target10 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=9  # 第n个频率
    # )
    # new_coords, Z_target11 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=10  # 第n个频率
    # )

    from core.process_multi_dim_params_space import extract_basic_analysis_fields
    from core.data_postprocess.momentum_space_toolkits import complete_C4_polarization, geom_complete
    from core.plot_cls import MomentumSpaceEigenVisualizer
    from core.plot_workflow import PlotConfig
    from core.prepare_plot import prepare_plot_data

    band_index = 2
    Z_target = Z_target3

    # 提取 band= 的附加场数据
    phi, tanchi, qlog, freq_real = extract_basic_analysis_fields(additional_Z_grouped, z_keys=z_keys,
                                                                 band_index=band_index)

    full_coords, phi_f, tanchi_f = complete_C4_polarization(new_coords, phi, tanchi)
    _, Z_f = geom_complete(new_coords, Z_target, mode='C4')
    _, qlog_f = geom_complete(new_coords, qlog, mode='C4')
    s1 = np.cos(2 * phi_f) * (1 - tanchi_f ** 2) / (1 + tanchi_f ** 2)
    s2 = np.sin(2 * phi_f) * (1 - tanchi_f ** 2) / (1 + tanchi_f ** 2)
    s3 = 2 * tanchi_f / (1 + tanchi_f ** 2)
    dataset1 = {
        'eigenfreq': Z_f,
        's1': s1,
        's2': s2,
        's3': s3,
        'qlog': qlog_f,
    }
    data_path = prepare_plot_data(
        coords=full_coords, data_class='Eigensolution', dataset_list=[dataset1], fixed_params={},
    )

    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.25, 1.25), tick_direction='in')
    plotter = MomentumSpaceEigenVisualizer(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()

    plotter.new_2d_fig()
    plotter.plot_polarization_ellipses(index=0, step=(1, 1), scale=0.001)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_field(index=0, field_key='s1', cmap='coolwarm', vmin=-1, vmax=1)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_field(index=0, field_key='s2', cmap='coolwarm', vmin=-1, vmax=1)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_field(index=0, field_key='s3', cmap='coolwarm', vmin=-1, vmax=1)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_field(index=0, field_key='qlog', cmap='nipy_spectral', vmin=2, vmax=7)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.prepare_chi_phi_data()
    plotter.plot_phi_families_regimes(index=0)
    plotter.plot_phi_families_split(index=0)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_3d_fig()
    plotter.plot_3D_surface(index=0)
    plotter.add_annotations()
    plotter.save_and_show()
