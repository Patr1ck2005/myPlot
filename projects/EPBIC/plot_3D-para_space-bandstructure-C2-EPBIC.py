from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.plot_3D_params_space_plt import *
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

if __name__ == '__main__':
    # data_path = 'data/low_mesh-3.58Si/slide_sym-0-0.20P_EP_BIC.csv'
    # data_path = 'data/low_mesh-3.58Si/slide_asym-EP_BIC.csv'
    data_path = 'data/EP_BIC-asym_slide-0.25P.csv'
    # data_path = 'data/EP_BIC-asym_slide-0.20P.csv'
    # data_path = 'data/EP_(Q)BIC-asym_slide-0.0,0.10P(to_narrow).csv'
    # data_path = 'data/EP_(Q)BIC-asym_slide-0.0,0.10P.csv'
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


    period = 1500
    df_sample["特征频率 (THz)"] = (df_sample["特征频率 (THz)"].apply(convert_complex)
                                   .apply(norm_freq, period=period * 1e-9 * 1e12))
    # df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period*1e-9)
    df_sample["频率 (Hz)"] = np.real(df_sample["特征频率 (THz)"])
    df_sample["k"] = df_sample["a"]
    df_sample["fake_factor (1)"] = 1 / df_sample["S3 (1)"]
    # param_keys = ["k", "spacer (nm)", "h_die_grating (nm)", "layer_shift (m)"]
    # param_keys = ["k", "spacer (nm)", "h_die_grating (nm)", "layer_shift_top (m)", "layer_shift_btn (m)"]
    param_keys = ["k", "spacer (nm)", "h_die_grating (nm)", "layer_shift_top (nm)", "layer_shift_btn (nm)",  "plas_wid_scale (nm)"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "fake_factor (1)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    KEY_X = 'spacer (nm)'
    KEY_Y = 'h_die_grating (nm)'

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            'k': 0,
            'plas_wid_scale (nm)': 0,
            # 'plas_wid_scale (nm)': 50,
            # 'plas_wid_scale (nm)': 100,
            # 'layer_shift (m)': 0.000e+00,
            # 'layer_shift (m)': -3.750e-08,
            # 'layer_shift (m)': -7.500e-08,
            # 'layer_shift (m)': -1.125e-07,
            # 'layer_shift (m)': -1.500e-07,
            # 'layer_shift_top (m)': 0.000e+00,
            # 'layer_shift_btn (m)': -3.75e-07,
            'layer_shift_top (nm)': 0,
            'layer_shift_btn (nm)': -375,
            # 'layer_shift_top (nm)': -75,
            # 'layer_shift_btn (nm)': -375,

            # 'layer_shift_top (nm)': +75,
            # 'layer_shift_btn (nm)': -75,

            # 'layer_shift_top (nm)': +0,
            # 'layer_shift_btn (nm)': -0,

            # 'layer_shift_top (m)': -0.75e-07,
            # 'layer_shift_btn (m)': -3.75e-07,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            # "特征频率 (THz)": {"<": 0.65, ">": 0},  # 筛选
            "特征频率 (THz)": {"<": 0.60, ">": 0},  # 筛选
        }
    )

    # # 测试3D散点绘图, 颜色映射取决于虚部
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 12))
    # xs = []
    # ys = []
    # zs = []
    # colors = []
    # for i, key_x in enumerate(new_coords[X_KEY]):
    #     for j, key_y in enumerate(new_coords[KEY_Y]):
    #         lst_ij = Z_filtered[i][j]
    #         for idx, freq in enumerate(lst_ij[0]):
    #             xs.append(key_x)
    #             ys.append(key_y)
    #             zs.append(freq.real)
    #             # colors.append(freq.imag)
    #             colors.append(idx)  # 第不同个频率用不同颜色
    # sc = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o', alpha=0.8, s=1)
    # # set aspect
    # ax.set_box_aspect([1, 1, 3])
    # # set view angle
    # ax.view_init(elev=15, azim=45)
    # plt.colorbar(sc, label='Imaginary Part of Frequency (THz)')
    # ax.set_xlabel(X_KEY)
    # ax.set_ylabel(KEY_Y)
    # ax.set_zlabel('Frequency (THz)')
    # plt.title('3D Scatter Plot of Eigenfrequencies')
    # plt.show()

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
            # Z_new[i, j] = np.imag(Z_filtered[i][j][0])  # 提取每个 lst_ij 的第 b 列

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=3,
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
    from core.plot_cls import TwoDimFieldVisualizer
    from core.plot_workflow import PlotConfig
    from core.prepare_plot import prepare_plot_data

    band_index = 0
    Z_target = Z_target1
    # band_index = 1
    # Z_target = Z_target2
    # band_index = 2
    # Z_target = Z_target3

    # =================================================================================================================
    # 计算三个Z_target的两两最大距离和两两最小距离
    Z_target_list = [Z_target1, Z_target2, Z_target3]
    num_bands = len(Z_target_list)
    max_distances = np.zeros(Z_target1.shape, dtype=float)
    min_distances = np.full(Z_target1.shape, np.inf, dtype=float)
    for i in range(num_bands):
        for j in range(i + 1, num_bands):
            dist = np.abs(Z_target_list[i] - Z_target_list[j])
            max_distances = np.maximum(max_distances, dist)
            min_distances = np.minimum(min_distances, dist)
    # 绘制最大距离的等高线图
    plt.rcParams['font.size'] = 9
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(1, 1))
    im = ax.imshow(max_distances.T, origin='lower',
                   extent=(new_coords[KEY_X][0], new_coords[KEY_X][-1], new_coords[KEY_Y][0], new_coords[KEY_Y][-1]),
                   aspect='auto', cmap='magma', vmin=0)
    # ax.set_xlabel(X_KEY)
    # ax.set_ylabel(KEY_Y)
    # fig.colorbar(im, ax=ax)
    # 清空坐标轴刻度和标签
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.savefig('max_distances_plot.svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
    # 绘制最小距离的等高线图
    fig, ax = plt.subplots(figsize=(1, 1))
    im = ax.imshow(min_distances.T, origin='lower',
                   extent=(new_coords[KEY_X][0], new_coords[KEY_X][-1], new_coords[KEY_Y][0], new_coords[KEY_Y][-1]),
                   aspect='auto', cmap='magma', vmin=0)
    # ax.set_xlabel(X_KEY)
    # ax.set_ylabel(KEY_Y)
    # fig.colorbar(im, ax=ax)
    # 清空坐标轴刻度和标签
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.savefig('min_distances_plot.svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
    # =================================================================================================================

    # 提取 band= 的附加场数据
    eigenfreq1, qfactor1, fake_factor1 = extract_adjacent_fields(
        additional_Z_grouped,
        z_keys=z_keys,
        band_index=0
    )
    eigenfreq2, qfactor2, fake_factor2 = extract_adjacent_fields(
        additional_Z_grouped,
        z_keys=z_keys,
        band_index=1
    )
    # print(additional_Z_grouped)
    eigenfreq3, qfactor3, fake_factor3 = extract_adjacent_fields(
        additional_Z_grouped,
        z_keys=z_keys,
        band_index=2
    )
    qlog1 = np.log10(qfactor1).real
    qlog2 = np.log10(qfactor2).real
    qlog3 = np.log10(qfactor3).real
    freq_real1 = np.real(eigenfreq1)

    # =================================================================================================================
    # # imshow 绘图
    # fig, ax = plt.subplots(figsize=(3, 2))
    # im = ax.imshow(qlog.T, origin='lower',
    #                extent=(new_coords[X_KEY][0], new_coords[X_KEY][-1], new_coords[KEY_Y][0], new_coords[KEY_Y][-1]),
    #                aspect='auto', cmap='hot')
    # ax.set_xlabel(X_KEY)
    # ax.set_ylabel(KEY_Y)
    # fig.colorbar(im, ax=ax)
    # plt.savefig('qlog_plot.svg', dpi=300, transparent=True, bbox_inches='tight')
    # plt.show()
    #
    # # imshow 绘图
    # fig, ax = plt.subplots(figsize=(3, 2))
    # im = ax.imshow(freq_real.T, origin='lower',
    #                extent=(new_coords[X_KEY][0], new_coords[X_KEY][-1], new_coords[KEY_Y][0], new_coords[KEY_Y][-1]),
    #                aspect='auto', cmap='hot')
    # ax.set_xlabel(X_KEY)
    # ax.set_ylabel(KEY_Y)
    # fig.colorbar(im, ax=ax)
    # plt.savefig('freq_real_plot.svg', dpi=300, transparent=True, bbox_inches='tight')
    # plt.show()
    # =================================================================================================================

    dataset1 = {
        'eigenfreq': eigenfreq1,
        'eigenfreq_real': eigenfreq1.real,
        'eigenfreq_imag': eigenfreq1.imag,
        'qlog': qlog1,
    }
    dataset2 = {
        'eigenfreq': eigenfreq2,
        'eigenfreq_real': eigenfreq2.real,
        'eigenfreq_imag': eigenfreq2.imag,
        'qlog': qlog2,
    }
    dataset3 = {
        'eigenfreq': eigenfreq3,
        'eigenfreq_real': eigenfreq3.real,
        'eigenfreq_imag': eigenfreq3.imag,
        'qlog': qlog3,
    }
    data_path = prepare_plot_data(
        coords=new_coords, data_class='Eigensolution', dataset_list=[dataset1, dataset2, dataset3], fixed_params={},
    )
# =====================================================================================================================
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.25, 1.25), tick_direction='in')
    plotter = TwoDimFieldVisualizer(config=config, data_path=data_path)
    plotter.load_data()
    # plotter.prepare_data(key_x=X_KEY, key_y=KEY_Y, key_fields=['eigenfreq', 'eigenfreq_real', 'eigenfreq_imag', 'qlog'])
    plotter.new_3d_fig(temp_figsize=(1.25, 1.5))
    # plotter.new_3d_fig(temp_figsize=(1.75, 2))
    # plotter.new_3d_fig(temp_figsize=(3, 2))
    plotter.plot_3d_surface(
        index=0, x_key=KEY_X, y_key=KEY_Y, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        elev=20, azim=120, alpha=1, vmin=0.0, vmax=0.0014,
        cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False
    )
    plotter.plot_3d_surface(
        index=1, x_key=KEY_X, y_key=KEY_Y, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        elev=20, azim=120, alpha=1, vmin=0.0, vmax=0.0014,
        cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False
    )
    plotter.plot_3d_surface(
        index=2, x_key=KEY_X, y_key=KEY_Y, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        elev=20, azim=120, alpha=1, vmin=0.0, vmax=0.0014,
        cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False
    )
    # plotter.plot_3d_surface(
    #     index=0, x_key=X_KEY, y_key=KEY_Y, z1_key='eigenfreq_imag', z2_key='eigenfreq_real',
    #     elev=20, azim=120, alpha=1, vmin=0.573, vmax=0.5755,
    #     cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False
    # )
    # plotter.plot_3d_surface(
    #     index=1, x_key=X_KEY, y_key=KEY_Y, z1_key='eigenfreq_imag', z2_key='eigenfreq_real',
    #     elev=20, azim=120, alpha=1, vmin=0.573, vmax=0.5755,
    #     cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False
    # )
    # plotter.plot_3d_surface(
    #     index=2, x_key=X_KEY, y_key=KEY_Y, z1_key='eigenfreq_imag', z2_key='eigenfreq_real',
    #     elev=20, azim=120, alpha=1, vmin=0.573, vmax=0.5755,
    #     cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False
    # )
    plotter.add_annotations()
    plotter.save_and_show()

