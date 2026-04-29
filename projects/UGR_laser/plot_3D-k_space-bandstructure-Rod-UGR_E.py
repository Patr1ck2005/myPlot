from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.process_multi_dim_params_space import create_data_grid, group_solution

import numpy as np
import pandas as pd

from core.utils import norm_freq, convert_complex

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/Tri_Rod-I-search0.55-detailed_k_space2Dim-norm_mesh-UGR_E-(tri0.02,400t).csv'
    # data_path = 'data/Tri_Rod-I-search0.55-detailed_k_space2Dim-norm_mesh-UGR_E-(tri0.05,400t).csv'
    # data_path = 'data/Tri_Rod-I-search0.55-detailed_k_space2Dim-norm_mesh-UGR_E-(tri0.10,400t).csv'
    # data_path = 'data/Tri_Rod-I-search0.55-detailed_k_space2Dim-norm_mesh-UGR_E-(tri0.15,400t,14eigens).csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    period = 500
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq,
                                                                                           period=period * 1e-9 * 1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period * 1e-9)
    df_sample["up_cx (V/m)"] = df_sample["up_cx (V/m)"].apply(convert_complex)
    df_sample["up_cy (V/m)"] = df_sample["up_cy (V/m)"].apply(convert_complex)
    df_sample["down_cx (V/m)"] = df_sample["down_cx (V/m)"].apply(convert_complex)
    df_sample["down_cy (V/m)"] = df_sample["down_cy (V/m)"].apply(convert_complex)
    # df_sample = df_sample[df_sample["m1"] <= 0.2]
    # df_sample = df_sample[df_sample["m2"] <= 0.2]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2", "t_slab_factor", "t_tot (nm)", "fill", "tri_factor", "substrate (nm)", "dpml (nm)"]
    z_keys = [
        "特征频率 (THz)", "品质因子 (1)",
        "up_tanchi (1)", "up_phi (rad)",
        "down_tanchi (1)", "down_phi (rad)",
        "fake_factor (1)", "频率 (Hz)",
        "U_factor (1)", "up_cx (V/m)", "up_cy (V/m)", "down_cx (V/m)", "down_cy (V/m)"
    ]

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
            't_slab_factor': 0.1795,
            't_tot (nm)': 400,
            'fill': 0.675,
            'tri_factor': 0.02,
            # 't_slab_factor': 0.179,
            # 't_tot (nm)': 400,
            # 'fill': 0.675,
            # 'tri_factor': 0.05,
            # 't_slab_factor': 0.1785,
            # 't_tot (nm)': 400,
            # 'fill': 0.678,
            # 'tri_factor': 0.10,
            # 't_slab_factor': 0.1795,
            # 't_tot (nm)': 400,
            # 'fill': 0.684,
            # 'tri_factor': 0.15,
            'substrate (nm)': 3000,
            'dpml (nm)': 600,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            # "品质因子 (1)": {"<": 1e6},  # 筛选
            # "特征频率 (THz)": {"<": 0.60, ">": 0},  # 筛选
        }
    )

    ###############################################################################################################
    from matplotlib import pyplot as plt

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
    ###############################################################################################################
    MAX_NUM = 20

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
            # Z_new[i, j] = np.imag(Z_filtered[i][j])[0]

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=MAX_NUM,
        auto_split_streams=False
    )

    Z_targets = []
    for freq_index in range(8):
        new_coords, Z_target = group_solution(
            new_coords, Z_grouped,
            freq_index=freq_index  # 第n个频率
        )
        Z_targets.append(Z_target)

    ###################################################################################################################
    from core.process_multi_dim_params_space import extract_adjacent_fields
    from core.plot_cls import MomentumSpaceEigenVisualizer
    from core.plot_workflow import PlotConfig
    from core.prepare_plot import prepare_plot_data
    from core.data_postprocess.data_package import package_stad_C2_data
    from core.data_postprocess.momentum_space_toolkits import geom_complete

    raw_datasets = []
    selected_bands = [5]
    for i in selected_bands:
        Z_target = Z_targets[i]
        eigenfreq_real = Z_target.real
        eigenfreq_imag = Z_target.imag
        eigenfreq, qfactor, up_tanchi, up_phi, down_tanchi, down_phi, fake_factor, freq, u_factor, \
            up_cx, up_cy, down_cx, down_cy = extract_adjacent_fields(
            additional_Z_grouped,
            z_keys=z_keys,
            band_index=i
        )
        qlog = np.log10(qfactor).real
        u_eff = -(1-np.abs(u_factor.real))/(1+np.abs(u_factor.real))
        u_factor = np.abs(u_factor.real)
        full_coords, qlog_supped = geom_complete(new_coords, qlog, mode='x')
        _, u_eff_supped = geom_complete(new_coords, u_eff, mode='x')
        _, u_factor_supped = geom_complete(new_coords, u_factor, mode='x')
        _, eigenfreq_real_supped = geom_complete(new_coords, eigenfreq_real, mode='x')
        _, eigenfreq_imag_supped = geom_complete(new_coords, eigenfreq_imag, mode='x')
        raw_dataset = {
            'eigenfreq_real': eigenfreq_real_supped, 'eigenfreq_imag': eigenfreq_imag_supped,
            'qlog': qlog_supped, 'u_eff': u_eff_supped, 'u_factor': u_factor_supped,
        }
        print(f"Band {i}: qlog range = [{raw_dataset['qlog'].min()}, {raw_dataset['qlog'].max()}]")
        raw_datasets.append(raw_dataset)

    # datasets = []
    # selected_bands = [5]
    # for i in selected_bands:
    #     Z_target = Z_targets[i]
    #     full_coords, dataset = package_stad_C2_data(
    #         new_coords, i, Z_target, additional_Z_grouped, z_keys,
    #         q_key='品质因子 (1)',
    #         tanchi_key='up_tanchi (1)',
    #         phi_key='up_phi (rad)',
    #         u_key='up_phi (rad)',
    #         # tanchi_key='down_tanchi (1)',
    #         # phi_key='down_phi (rad)',
    #     )
    #     datasets.append(dataset)

    data_path = prepare_plot_data(
        coords=full_coords, data_class='Eigensolution', dataset_list=raw_datasets, fixed_params={},
        save_dir='./rsl/2_para_space',
    )

    ####################################################################################################################
    BAND_INDEX = 0
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.5, 1.5), tick_direction='in')

    plotter = MomentumSpaceEigenVisualizer(config=config, data_path=data_path)
    plotter.load_data()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='qlog', cmap='nipy_spectral', vmin=2, vmax=4)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='u_eff', cmap='RdBu', vmin=-1, vmax=1)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_3d_fig(figsize=(3, 3))
    # plotter.plot_3d_surfaces(
    #     indices=(0, 1), z1_key='eigenfreq_real', z2_key='qlog', cmap='rainbow', elev=45, vmin=2, vmax=7, shade=False
    # )
    # rbga = plotter.get_advanced_color_mapping(index=BAND_INDEX)
    # plotter.plot_3d_surface(index=BAND_INDEX, z1_key='eigenfreq_real', rgba=rbga, cmap='rainbow', elev=45, shade=False)
    plotter.plot_3d_surface(
        index=BAND_INDEX, z1_key='eigenfreq_real', z2_key='u_eff', cmap='RdBu', elev=45, vmin=-1, vmax=1, shade=False
    )
    plotter.ax.set_box_aspect([1, 1, 0.75])
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s1', cmap='coolwarm', vmin=-1, vmax=1)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s2', cmap='coolwarm', vmin=-1, vmax=1)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s3', cmap='coolwarm', vmin=-1, vmax=1)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_compare_datas(
        index_A=0, index_B=1,
        field_key='eigenfreq_real',
        cmap='nipy_spectral',
        vmin=0,
    )
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_compare_datas(
        index_A=0, index_B=1,
        field_key='eigenfreq_imag',
        cmap='nipy_spectral',
        vmin=0,
    )
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_compare_datas(
        index_A=0, index_B=1,
        field_key='eigenfreq',
        cmap='nipy_spectral',
        vmin=0,
    )
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    # plotter.plot_field_regimes(index=BAND_INDEX, z_key='s2')
    # plotter.plot_field_splits(index=BAND_INDEX, s1_key='s2', s2_key='s1')
    # plotter.plot_field_regimes(index=BAND_INDEX, z_key='s1')
    # plotter.plot_field_splits(index=BAND_INDEX, s1_key='s1', s2_key='s2')
    plotter.plot_field_regimes(index=BAND_INDEX, z_key='s3', colors=("lightcoral", "lightblue"))
    plotter.plot_field_splits(index=BAND_INDEX, s1_key='s3', s2_key='s1')
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.25, 1.25))
    import matplotlib.colors as mcolors

    bounds = [-1, -0.995, -0.99, -0.95, -0.5, -0.05, 0.05, 0.5, 0.95, 0.99, 0.995, 1]
    norm = mcolors.BoundaryNorm(bounds, ncolors=256)
    plotter.imshow_field(index=BAND_INDEX, field_key='s2', cmap='coolwarm', norm=norm)
    # plotter.plot_skyrmion_quiver(index=BAND_INDEX, step=(1, 1), cmap='coolwarm', s1_key='s1', s2_key='s3', s3_key='s2')
    # plotter.plot_polar_quiver(index=BAND_INDEX, step=(1, 1), color='gray', s1_key='s1', s2_key='s2', s3_key='s3')
    plotter.plot_polar_quiver(index=BAND_INDEX, step=(1, 1), color='gray', s1_key='s1', s2_key='s3', s3_key='s2')
    # plotter.add_annotations()
    plotter.save_and_show()

    import matplotlib.colors as mcolors

    plotter.new_2d_fig(figsize=(1.25, 1.25))
    bounds = [-1, -0.995, -0.99, -0.95, -0.5, -0.05, 0.05, 0.5, 0.95, 0.99, 0.995, 1]
    norm = mcolors.BoundaryNorm(bounds, ncolors=256)
    plotter.imshow_field(index=BAND_INDEX, field_key='s3', cmap='coolwarm', norm=norm)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(4, 4), scale=2e-2, cmap='coolwarm')
    # plotter.plot_iso_contours2D(index=BAND_INDEX, levels=(0.36, 0.37, 0.38), z_key='eigenfreq_real', colors=('r', 'k', 'b'))
    # plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1, 1))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(4, 4), scale=2e-2, cmap='coolwarm')
    plotter.ax.set_xlim(-0.1, 0.1)
    plotter.ax.set_ylim(-0.1, 0.1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(3.8, 1))
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.03, color='r', field_key='phi',
                                        alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.05, color='k', field_key='phi',
                                        alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.07, color='b', field_key='phi',
                                        alpha=0.5)
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(3.8, 1))
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.03, color='r', field_key='s3',
                                        alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.05, color='k', field_key='s3',
                                        alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.07, color='b', field_key='s3',
                                        alpha=0.5)
    plotter.save_and_show()
    ####################################################################################################################
