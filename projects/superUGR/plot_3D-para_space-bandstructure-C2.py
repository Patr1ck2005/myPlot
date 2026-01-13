from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.plot_3D_params_space_plt import *
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458


if __name__ == '__main__':
    # data_path = 'data/4000Q-diff_h-diff_kx-7eigens.csv'
    # data_path = 'data/DUGR-diff_h-diff_kx-7eigens.csv'
    data_path = 'data/DUGR-diff_fill-diff_kx-7eigens.csv'
    df_sample = pd.read_csv(data_path, sep='\t')


    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))


    def norm_freq(freq, period):
        return freq / (c_const / period)


    def recognize_sp(phi_arr, kx_arr, ky_arr):
        # 对于 ky=0 的情况，phi=π/2 为 s 偏振, phi=0 为 p 偏振
        # 对于 ky=kx 的情况，phi=π/4 为 s 偏振，phi=3*π/4 为 p 偏振
        sp_polar = []
        for phi, kx, ky in zip(phi_arr, kx_arr, ky_arr):
            if np.isclose(ky, 0):
                if np.isclose(phi, np.pi / 2, atol=1e-1):
                    sp_polar.append(1)
                else:
                    sp_polar.append(0)
            elif np.isclose(ky, kx):
                if np.isclose(phi, np.pi / 4, atol=1e-1):
                    sp_polar.append(1)
                else:
                    sp_polar.append(0)
            else:
                sp_polar.append(-1)
        return sp_polar


    period = 2160
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq,
                                                                                           period=period * 1e-9 * 1e12)
    # df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period*1e-9)
    df_sample["频率 (Hz)"] = np.real(df_sample["特征频率 (THz)"])
    df_sample["k"] = df_sample["a"] - df_sample["b"]
    df_sample["fake_factor (1)"] = 1 / df_sample["Prop_m (1)"]
    # # 筛选k的成分
    df_sample = df_sample[df_sample["k"] <= 0.080]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["k", "fill", "delta_factor", "h_grating (nm)"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "up_S3 (1)", "down_S3 (1)", "fake_factor (1)", "U_factor (1)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    KEY_X = 'k'
    KEY_Y = 'fill'
    # KEY_Y = 'h_grating (nm)'

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            'delta_factor': 2.,
            # 'fill': 0.1773,
            # 'fill': 0.117,
            'h_grating (nm)': 308.1,
            # "sp_polar_show": 1,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            "特征频率 (THz)": {"<": 0.7, ">": 0},  # 筛选
        }
    )

    # 测试3D散点绘图, 颜色映射取决于虚部
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 12))
    xs = []
    ys = []
    zs = []
    colors = []
    for i, key_x in enumerate(new_coords[KEY_X]):
        for j, key_y in enumerate(new_coords[KEY_Y]):
            lst_ij = Z_filtered[i][j]
            for idx, freq in enumerate(lst_ij[0]):
                xs.append(key_x)
                ys.append(key_y)
                zs.append(freq.real)
                # colors.append(freq.imag)
                colors.append(idx)  # 第不同个频率用不同颜色
    sc = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o', alpha=0.8, s=1)
    # set aspect
    ax.set_box_aspect([1, 1, 3])
    # set view angle
    ax.view_init(elev=15, azim=45)
    plt.colorbar(sc, label='Imaginary Part of Frequency (THz)')
    ax.set_xlabel(KEY_X)
    ax.set_ylabel(KEY_Y)
    ax.set_zlabel('Frequency (THz)')
    plt.title('3D Scatter Plot of Eigenfrequencies')
    plt.show()

    deltas = (1e-3, 1e-3)  # n个维度的网格间距
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
        [Z_new], deltas,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=3
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
    # new_coords, Z_target4 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=3  # 第n个频率
    # )
    # new_coords, Z_target5 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=4  # 第n个频率
    # )
    # new_coords, Z_target6 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=5  # 第n个频率
    # )
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

    from core.process_multi_dim_params_space import extract_basic_analysis_fields, plot_advanced_surface
    import matplotlib.pyplot as plt
    from core.data_postprocess.momentum_space_toolkits import complete_C4_polarization, geom_complete
    from core.plot_cls import MomentumSpaceEigenPolarizationPlotter
    from core.plot_workflow import PlotConfig
    from core.prepare_plot import prepare_plot_data

    # band_index = 0
    # Z_target = Z_target1
    band_index = 1
    Z_target = Z_target2
    # band_index = 2
    # Z_target = Z_target3

    # 提取 band= 的附加场数据
    eigenfreq, qfactor, top_S3, btn_S3, fake_factor, u_factor = extract_adjacent_fields(
        additional_Z_grouped,
        z_keys=z_keys,
        band_index=band_index
    )
    qlog = np.log10(qfactor)
    ulog = np.log10(-u_factor)
    freq_real = np.real(eigenfreq)

    # imshow 绘图
    fig, ax = plt.subplots(figsize=(3, 2))
    im = ax.imshow(ulog.T, origin='lower',
              extent=(new_coords[KEY_X][0], new_coords[KEY_X][-1], new_coords[KEY_Y][0], new_coords[KEY_Y][-1]),
              aspect='auto', cmap='rainbow')
    ax.set_xlabel(KEY_X)
    ax.set_ylabel(KEY_Y)
    fig.colorbar(im, ax=ax)
    plt.savefig('ulog_plot.svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    # imshow 绘图
    fig, ax = plt.subplots(figsize=(3, 2))
    im = ax.imshow(qlog.T, origin='lower',
              extent=(new_coords[KEY_X][0], new_coords[KEY_X][-1], new_coords[KEY_Y][0], new_coords[KEY_Y][-1]),
              aspect='auto', cmap='hot')
    ax.set_xlabel(KEY_X)
    ax.set_ylabel(KEY_Y)
    fig.colorbar(im, ax=ax)
    plt.savefig('qlog_plot.svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    # imshow 绘图
    fig, ax = plt.subplots(figsize=(3, 2))
    im = ax.imshow(freq_real.T, origin='lower',
              extent=(new_coords[KEY_X][0], new_coords[KEY_X][-1], new_coords[KEY_Y][0], new_coords[KEY_Y][-1]),
              aspect='auto', cmap='hot')
    ax.set_xlabel(KEY_X)
    ax.set_ylabel(KEY_Y)
    fig.colorbar(im, ax=ax)
    plt.savefig('freq_real_plot.svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


    dataset1 = {
        'eigenfreq': eigenfreq,
        # 's1': s1,
        # 's2': s2,
        # 's3': s3,
        # 's1': s1,
        # 's2': s2,
        # 's3': top_S3,
        's3': u_factor,
        'qlog': qlog,
    }
    data_path = prepare_plot_data(
        coords=new_coords, data_class='Eigensolution', dataset_list=[dataset1], fixed_params={},
    )

    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.25, 1.25), tick_direction='in')
    plotter = TwoDimParaSpacePlotter(config=config, data_path=data_path, coordinate_keys={'x': KEY_X, 'y': KEY_Y})
    plotter.load_data()
    plotter.prepare_data()

    # plotter.new_2d_fig()
    # plotter.plot_polarization_ellipses(index=0)
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.509, 0.510, 0.511))
    # plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_qlog(index=0)
    plotter.save_and_show()

    # plotter.new_2d_fig()
    # plotter.prepare_chi_phi_data()
    # plotter.plot_phi_families_regimes(index=0)
    # plotter.plot_phi_families_split(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()

    plotter.new_3d_fig()
    plotter.plot_3D_surface(index=0)
    plotter.add_annotations()
    plotter.save_and_show()
