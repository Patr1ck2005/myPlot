from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.process_multi_dim_params_space import *

import numpy as np

from utils.advanced_color_mapping import map_complex2rbg

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/Hex_annular-ExpTest-spectrum1.csv'
    df_sample = pd.read_csv(data_path, sep='\t')


    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))


    def norm_freq(freq, period):
        return freq / (c_const / period)


    period = 1300
    # df_sample["f (c/P)"] = df_sample["freq (THz)"].apply(norm_freq, period=period * 1e3)
    df_sample["f (c/P)"] = df_sample["freq (THz)"]
    df_sample["k"] = df_sample["m1"] - df_sample["m2"]
    # df_sample["S11 (1)"] = df_sample["S11 (1)"].apply(convert_complex)
    # df_sample["S21 (1)"] = df_sample["S21 (1)"].apply(convert_complex)
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["k", "freq (THz)"]
    z_keys = ["总反射率 (1)"]

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
        },  # 固定
        filter_conditions={
        }
    )

    # 创建一个新的数组，用于存储更新后的结果
    s11 = np.empty_like(Z_filtered, dtype=object)
    s21 = np.empty_like(Z_filtered, dtype=object)
    R = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        for j in range(Z_filtered.shape[1]):
            R[i, j] = Z_filtered[i][j][0]

    new_coords, R = group_solution(
        new_coords, R,
        freq_index=0  # 第n个
    )

    # new_coords, s11 = group_solution(
    #     new_coords, s11,
    #     freq_index=0  # 第n个
    # )
    #
    # new_coords, s21 = group_solution(
    #     new_coords, s21,
    #     freq_index=0  # 第n个
    # )

    from core.process_multi_dim_params_space import extract_basic_analysis_fields, plot_advanced_surface
    import matplotlib.pyplot as plt
    from core.data_postprocess.momentum_space_toolkits import geom_complete, \
        complete_C4_spectrum
    from core.plot_cls import MomentumSpaceEigenPolarizationPlotter, MomentumSpaceSpectrumPlotter
    from core.plot_workflow import PlotConfig
    from core.prepare_plot import prepare_plot_data

    dataset1 = {
        's11': s11,
        's21': s21,
        'R': R,
    }
    data_path = prepare_plot_data(
        coords=new_coords, data_class='Spectrum', dataset_list=[dataset1], fixed_params={},
    )

    fig, ax = plt.subplots(figsize=(1.5, 2))
    im = ax.imshow(R.real.T, origin='lower', extent=(new_coords['k'][0], new_coords['k'][-1],
                                                     new_coords['freq (THz)'][0], new_coords['freq (THz)'][-1]),
                   cmap='gray', aspect='auto', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, label='R')
    ax.set_xlabel('k (2π/P)')
    ax.set_ylabel('Frequency (THz)')
    plt.savefig('temp.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.show()

    # fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    # phase_s21 = np.angle(s21_f)
    # im = ax.imshow(phase_s21.T, origin='lower', extent=(full_coords['m1'][0], full_coords['m1'][-1],
    #                                                     full_coords['m2'][0], full_coords['m2'][-1]),
    #                cmap='hsv')
    # cbar = fig.colorbar(im, ax=ax)
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(1.25, 1.25), dpi=200)
    # abs_s21 = np.abs(s21_f) ** 2
    # # im = ax.imshow(abs_s21.T, origin='lower', extent=(full_coords['m1'][0], full_coords['m1'][-1],
    # #                                                   full_coords['m2'][0], full_coords['m2'][-1]),
    # #                cmap='viridis')
    # im = ax.imshow(np.real(s21_f).T, origin='lower', extent=(full_coords['m1'][0], full_coords['m1'][-1],
    #                                                          full_coords['m2'][0], full_coords['m2'][-1]),
    #                cmap='RdBu')
    # plt.savefig('temp.svg', bbox_inches='tight', transparent=True)
    # plt.show()
