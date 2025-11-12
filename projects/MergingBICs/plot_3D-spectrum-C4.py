from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.process_multi_dim_params_space import *

import numpy as np

from utils.advanced_color_mapping import map_complex2rbg

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/FP_PhC-designD-spectrum.csv'
    df_sample = pd.read_csv(data_path, sep='\t')


    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))


    def norm_freq(freq, period):
        return freq / (c_const / period)


    period = 1300
    df_sample["f (c/P)"] = df_sample["freq (THz)"].apply(norm_freq, period=period * 1e3)
    df_sample["S11 (1)"] = df_sample["S11 (1)"].apply(convert_complex)
    df_sample["S21 (1)"] = df_sample["S21 (1)"].apply(convert_complex)
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2", "f (c/P)"]
    z_keys = ["S11 (1)", "S21 (1)"]

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
            'f (c/P)': 0.655,
        },  # 固定
        filter_conditions={
        }
    )

    # 创建一个新的数组，用于存储更新后的结果
    s11 = np.empty_like(Z_filtered, dtype=object)
    s21 = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        for j in range(Z_filtered.shape[1]):
            s11[i, j] = Z_filtered[i][j][0]
            s21[i, j] = Z_filtered[i][j][1]

    new_coords, s11 = group_solution(
        new_coords, s11,
        freq_index=0  # 第n个
    )

    new_coords, s21 = group_solution(
        new_coords, s21,
        freq_index=0  # 第n个
    )

    from core.process_multi_dim_params_space import extract_basic_analysis_fields, plot_advanced_surface
    import matplotlib.pyplot as plt
    from core.data_postprocess.momentum_space_toolkits import geom_complete, \
        complete_C4_spectrum
    from core.plot_cls import MomentumSpaceEigenPolarizationPlotter, MomentumSpaceSpectrumPlotter
    from core.plot_workflow import PlotConfig
    from core.prepare_plot import prepare_plot_data

    full_coords, s11_f = complete_C4_spectrum(new_coords, s11)
    _, s21_f = complete_C4_spectrum(new_coords, s21)
    dataset1 = {
        's11': s11_f,
        's21': s21_f,
    }
    data_path = prepare_plot_data(
        coords=full_coords, data_class='Spectrum', dataset_list=[dataset1], fixed_params={},
    )

    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    phase_s21 = np.angle(s21_f)
    im = ax.imshow(phase_s21.T, origin='lower', extent=(full_coords['m1'][0], full_coords['m1'][-1],
                                                        full_coords['m2'][0], full_coords['m2'][-1]),
                   cmap='hsv')
    cbar = fig.colorbar(im, ax=ax)
    plt.show()

    fig, ax = plt.subplots(figsize=(1.25, 1.25), dpi=200)
    abs_s21 = np.abs(s21_f) ** 2
    # im = ax.imshow(abs_s21.T, origin='lower', extent=(full_coords['m1'][0], full_coords['m1'][-1],
    #                                                   full_coords['m2'][0], full_coords['m2'][-1]),
    #                cmap='viridis')
    im = ax.imshow(np.real(s21_f).T, origin='lower', extent=(full_coords['m1'][0], full_coords['m1'][-1],
                                                             full_coords['m2'][0], full_coords['m2'][-1]),
                   cmap='RdBu')
    plt.savefig('temp.svg', bbox_inches='tight', transparent=True)
    plt.show()

    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.25, 1.25), tick_direction='in')
    plotter = MomentumSpaceSpectrumPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()

    plotter.new_2d_fig()
    plotter.imshow_s21(index=0)
    # plotter.add_annotations()
    plotter.save_and_show()

    config.update(figsize=(1.25, 1.25), tick_direction='out')
    # 例：取第0个数据的 s21 做传播
    m1, m2 = plotter.m1, plotter.m2  # 你的坐标轴（-0.1~0.1）
    U0 = plotter.s21_list[0]  # 初始角谱（复振幅）
    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    real_U0 = np.real(U0)
    from matplotlib.colors import SymLogNorm
    im = ax.imshow(real_U0.T, origin='lower', extent=(m1[0], m1[-1], m2[0], m2[-1]),
                   cmap='RdBu', norm=SymLogNorm(linthresh=1e-1, vmin=-1, vmax=1))
    cbar = fig.colorbar(im, ax=ax)
    plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=1000)
    plt.show()
    # 做圆形裁剪. 按照比例radius_prop保留中心区域
    radius_prop = 0.4
    m_radius = radius_prop * np.min([np.max(m1) - np.min(m1), np.max(m2) - np.min(m2)])
    M1, M2 = np.meshgrid(m1, m2, indexing='xy')
    mask_circle = M1 ** 2 + M2 ** 2 <= m_radius ** 2
    U0 = U0 * mask_circle
    # 做0填充
    pad_m1 = 2
    pad_m2 = 2
    pad_m1 = int(pad_m1 * len(m1))
    pad_m2 = int(pad_m2 * len(m2))
    U0 = np.pad(U0, ((pad_m1, pad_m1), (pad_m2, pad_m2)), mode='constant', constant_values=0)
    # 更新 m1, m2
    m1 = np.linspace(np.min(m1) - pad_m1 * (m1[1] - m1[0]),
                     np.max(m1) + pad_m1 * (m1[1] - m1[0]), len(m1) + 2 * pad_m1)
    m2 = np.linspace(np.min(m2) - pad_m2 * (m2[1] - m2[0]),
                     np.max(m2) + pad_m2 * (m2[1] - m2[0]), len(m2) + 2 * pad_m2)

    # 物理参数
    n = 1.0  # 折射率

    from core.fourier_module.ms_propagation_toolkit import *

    # 1) 单个 z 面
    z = 0.00
    norm_f = 0.655
    # m1 的单位为 2*π/P
    # 频率的单位为 c/P
    norm_m1 = m1 / norm_f
    norm_m2 = m2 / norm_f
    # 可视化：np.abs(Exy) 或 np.angle(Exy)
    fig, ax = plt.subplots(figsize=(1.25, 1.25), dpi=200)
    im = ax.imshow(np.abs(U0).T ** 2, origin='lower', extent=(norm_m1[0], norm_m1[-1], norm_m2[0], norm_m2[-1]),
                   cmap='magma')
    cbar = fig.colorbar(im, ax=ax)
    ax.set_xlabel('norm m1 ($k_0$)')
    ax.set_ylabel('norm m2 ($k_0$)')
    plt.show()
    Az, (x, y, Exy) = angular_spectrum_propagate(norm_m1, norm_m2, U0, z, k0=1)
    # 可视化：np.abs(Exy) 或 np.angle(Exy)
    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    # im = ax.imshow(np.abs(Exy).T**2, origin='lower', extent=(x[0], x[-1], y[0], y[-1]),
    #                cmap='magma')
    im = ax.imshow(np.real(Exy).T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]),
                   cmap='RdBu')
    # rgb = map_complex2rbg(Exy)
    # im = ax.imshow(rgb, origin='lower', extent=(x[0], x[-1], y[0], y[-1]))
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=1000)
    plt.show()

    # 2) XZ 纵截面（y=0 切片）
    z_list = np.linspace(0, 1e4, 100)
    x, z, Exz = angular_spectrum_xz_slice(m1, m2, U0, z_list, k0=1, y0=0.0, time_sign=1)
    # 可视化：np.abs(Exz) 或 np.angle(Exz)
    fig, ax = plt.subplots(figsize=(4, 1.25))
    im = ax.imshow(np.abs(Exz).T**2, origin='lower', extent=(z[0], z[-1], x[0], x[-1]),
                   aspect='auto', cmap='magma')
    ax.set_ylim(-1.5e3, 1.5e3)
    plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=1000)
    plt.show()
