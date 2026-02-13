from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.process_multi_dim_params_space import create_data_grid, group_solution

import numpy as np
import pandas as pd

from core.utils import norm_freq, convert_complex

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/AsymEnv-ultra_mesh-search0.40-geo_520T-around.csv'
    # data_path = 'data/VacuumEnv-ultra_mesh-search0.40-geo_Arrow-around_X_BIC.csv'
    # data_path = 'data/VacuumEnv-ultra_mesh-search0.40-geo_Trap-around_X_BIC.csv'
    # data_path = 'data/Vacuum-ultra_mesh-search0.40-geo540T-around_X_BIC_0.015k.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    period = 500
    df_sample["特征频率 (THz)"] = (df_sample["特征频率 (THz)"].apply(convert_complex)
                                   .apply(norm_freq, period=period * 1e-9 * 1e12))
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period * 1e-9)
    df_sample["up_cx (V/m)"] = df_sample["up_cx (V/m)"].apply(convert_complex)
    df_sample["up_cy (V/m)"] = df_sample["up_cy (V/m)"].apply(convert_complex)
    df_sample["down_cx (V/m)"] = df_sample["down_cx (V/m)"].apply(convert_complex)
    df_sample["down_cy (V/m)"] = df_sample["down_cy (V/m)"].apply(convert_complex)
    # df_sample = df_sample[df_sample["m1"] <= 0.2]
    # df_sample = df_sample[df_sample["m2"] <= 0.2]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = [
        "m1", "m2", "t_ridge (nm)", "fill", "t_tot (nm)", "substrate_n",
        # "M_asym_factor", "P_asym_factor", "Z_asym_factor"
    ]
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
            "t_tot (nm)": 520,
            "t_ridge (nm)": 520,
            "fill": 0.5,
            "substrate_n": 1.0,
            # "M_asym_factor": 0.00,
            # "P_asym_factor": 0.00,
            # "Z_asym_factor": 0.00,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 2},  # 筛选
            # "频率 (Hz)": {">": 0.32, "<": 0.45},  # 筛选
        }
    )

    # ###############################################################################################################
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 12))
    # xs = []
    # ys = []
    # zs = []
    # colors = []
    # for i, m1 in enumerate(new_coords['m1']):
    #     for j, m2 in enumerate(new_coords['m2']):
    #         lst_ij = Z_filtered[i][j]
    #         for freq in lst_ij[0]:
    #             xs.append(m1)
    #             ys.append(m2)
    #             zs.append(freq.real)
    #             colors.append(freq.imag)
    # sc = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o', alpha=0.8, s=1)
    # # set aspect
    # ax.set_box_aspect([1, 1, 3])
    # # set view angle
    # ax.view_init(elev=15, azim=45)
    # plt.colorbar(sc, label='Imaginary Part of Frequency (THz)')
    # ax.set_xlabel('m1')
    # ax.set_ylabel('m2')
    # ax.set_zlabel('Frequency (THz)')
    # plt.title('3D Scatter Plot of Eigenfrequencies')
    # plt.show()
    # ###############################################################################################################

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
        max_m=14,
        auto_split_streams=False
    )

    Z_targets = []
    for freq_index in range(5):
        new_coords, Z_target = group_solution(
            new_coords, Z_grouped,
            freq_index=freq_index  # 第n个频率
        )
        Z_targets.append(Z_target)

    ###################################################################################################################
    from core.process_multi_dim_params_space import extract_adjacent_fields
    from core.prepare_plot import prepare_plot_data
    from core.data_postprocess.data_package import package_stad_C2_data

    raw_datasets = []
    for i, Z_target in enumerate(Z_targets):
        raw_dataset = {'eigenfreq_real': Z_target.real, 'eigenfreq_imag': Z_target.imag}
        eigenfreq, qfactor, up_tanchi, up_phi, down_tanchi, down_phi, fake_factor, freq, u_factor, \
            up_cx, up_cy, down_cx, down_cy = extract_adjacent_fields(
            additional_Z_grouped,
            z_keys=z_keys,
            band_index=i
        )
        qlog = np.log10(qfactor)
        raw_dataset['qlog'] = qlog.real
        raw_dataset['up_cx (V/m)'] = up_cx
        raw_dataset['up_cy (V/m)'] = up_cy
        raw_dataset['down_cx (V/m)'] = up_cx
        raw_dataset['down_cy (V/m)'] = up_cy
        print(f"Band {i}: qlog range = [{raw_dataset['qlog'].min()}, {raw_dataset['qlog'].max()}]")
        raw_datasets.append(raw_dataset)
    # # imshow up_cx
    from matplotlib import pyplot as plt

    # fig, ax = plt.subplots(figsize=(1.25, 1.25))
    # normed_up_cx = up_cx1 / (np.abs(up_cx1)+np.abs(up_cy1))
    # # normed_up_cy = up_cy1 / (np.abs(up_cx1)+np.abs(up_cy1))
    # normed_up_cy = up_cy1
    # c = ax.imshow(np.imag(normed_up_cy).T, origin='lower', extent=(
    #     new_coords['m1'][0], new_coords['m1'][-1],
    #     new_coords['m2'][0], new_coords['m2'][-1],
    # ), aspect='auto', cmap='viridis')
    # fig.colorbar(c, ax=ax)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # plt.savefig('./c.svg', dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(1.25, 1.25))
    # normed_up_cx = up_cx1 / (np.abs(up_cx1)+np.abs(up_cy1))
    # normed_up_cy = up_cy1 / (np.abs(up_cx1)+np.abs(up_cy1))
    # phase_diff = (np.angle(up_cx1) - np.angle(up_cy1) + np.pi)%(2*np.pi) - np.pi
    # c = ax.imshow(phase_diff.T, origin='lower', extent=(
    #     new_coords['m1'][0], new_coords['m1'][-1],
    #     new_coords['m2'][0], new_coords['m2'][-1],
    # ), aspect='auto', cmap='viridis')
    # fig.colorbar(c, ax=ax)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # plt.savefig('./c.svg', dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(1.25, 1.25))
    # # 取m2=0处的切片, 在复数平面上绘制normed_up_cy的实部和虚部
    # m2_index = np.argmin(np.abs(new_coords['m2'] - 0))
    # plt.scatter(np.real(normed_up_cy[:, m2_index]), np.imag(normed_up_cy[:, m2_index]), s=5, marker='+', color='k')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    # ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    # plt.savefig('./c.svg', dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()

    datasets = []
    selected_bands = [0, 1]
    for i in selected_bands:
        Z_target = Z_targets[i]
        full_coords, dataset = package_stad_C2_data(
            new_coords, i, Z_target, additional_Z_grouped, z_keys,
            q_key='品质因子 (1)',
            tanchi_key='up_tanchi (1)',
            phi_key='up_phi (rad)',
            # tanchi_key='down_tanchi (1)',
            # phi_key='down_phi (rad)',
            axis='y',
        )
        datasets.append(dataset)

    data_path = prepare_plot_data(
        coords=full_coords, data_class='Eigensolution', dataset_list=datasets, fixed_params={},
        save_dir='./rsl/2_para_space',
    )

    ####################################################################################################################
    from core.plot_cls import MomentumSpaceEigenVisualizer
    from core.plot_workflow import PlotConfig

    BAND_INDEX = 0
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.5, 1.5), tick_direction='in')

    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    # phase_diff = (np.angle(up_cx2) - np.angle(up_cy2) + np.pi)%(2*np.pi) - np.pi
    # 通过kx, ky, 变换 (x, y) -> (s, p)
    kx, ky = np.meshgrid(new_coords['m1'], new_coords['m2'], indexing='ij')  # 或 new_coords['m1'], new_coords['m2']
    k_par = np.sqrt(kx ** 2 + ky ** 2)
    up_cs = (-ky * raw_datasets[BAND_INDEX]['up_cx (V/m)'] + kx * raw_datasets[BAND_INDEX]['up_cy (V/m)']) / k_par
    up_cp = (kx * raw_datasets[BAND_INDEX]['up_cx (V/m)'] + ky * raw_datasets[BAND_INDEX]['up_cy (V/m)']) / k_par
    phase_diff = (np.angle(up_cs) - np.angle(up_cp) + np.pi) % (2 * np.pi) - np.pi
    c = ax.imshow(phase_diff.T, origin='lower', extent=(
        new_coords['m1'][0], new_coords['m1'][-1],
        new_coords['m2'][0], new_coords['m2'][-1],
    ), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    # cs = ax.contour(kx, ky, phase_diff, levels=[-np.pi / 2, np.pi / 2], colors=['r', 'b'], linewidths=0.5)
    cs = ax.contour(kx, ky, np.real(up_cs*np.conj(up_cp)), levels=[0], colors=['k'], linewidths=0.5)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    plt.savefig('./c.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    plotter = MomentumSpaceEigenVisualizer(config=config, data_path=data_path)
    plotter.load_data()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s1', cmap='coolwarm', vmin=-1, vmax=1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s2', cmap='coolwarm', vmin=-1, vmax=1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    import matplotlib.colors as mcolors

    bounds = [-1, -0.995, -0.99, -0.95, -0.5, -0.05, 0.05, 0.5, 0.95, 0.99, 0.995, 1]
    norm = mcolors.BoundaryNorm(bounds, ncolors=256)
    plotter.imshow_field(index=BAND_INDEX, field_key='s3', cmap='coolwarm', norm=norm)
    # plotter.imshow_field(index=BAND_INDEX, field_key='s3', cmap='coolwarm', vmin=-1, vmax=1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='qlog', cmap='nipy_spectral', vmin=2, vmax=8)
    plotter.add_annotations()
    plotter.save_and_show()
    print("绘图已经整理到其他文件中")
    ####################################################################################################################
