from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.process_multi_dim_params_space import *

import numpy as np

from core.utils import norm_freq, convert_complex
from utils.advanced_color_mapping import map_complex2rbg

c_const = 299792458

if __name__ == '__main__':
    # data_path = 'data/spectrums/spectrum-s-btm-Tri_Rod-I-norm_mesh-0.03k-UGR_E-(tri0.10,400t_tot,0.003k).csv'
    # data_path = 'data/spectrums/spectrum-s-top-Tri_Rod-I-norm_mesh-0.03k-UGR_E-(tri0.10,400t_tot,0.003k).csv'
    # data_path = 'data/spectrums/spectrum-s-top-Tri_Rod-I-norm_mesh-0.03k-UGR_E-(tri0.10,400t_tot,0k).csv'
    data_path = 'data/spectrums/spectrum-s-btm-Tri_Rod-I-norm_mesh-0.03k-UGR_E-(tri0.10,400t_tot,0k).csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    period = 500
    df_sample["f (c/P)"] = df_sample["freq (THz)"].apply(norm_freq, period=period * 1e3)
    df_sample["k"] = df_sample["m1"] + df_sample["m2"]/2.414
    # df_sample["S11 (1)"] = df_sample["S11 (1)"].apply(convert_complex)  # top
    # df_sample["S21 (1)"] = df_sample["S21 (1)"].apply(convert_complex)  # top
    df_sample["S12 (1)"] = df_sample["S12 (1)"].apply(convert_complex)  # btm
    df_sample["S22 (1)"] = df_sample["S22 (1)"].apply(convert_complex)  # btm
    df_sample = df_sample[(df_sample["f (c/P)"] >= 0.524) & (df_sample["f (c/P)"] <= 0.615)]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["k", "f (c/P)"]
    # z_keys = ["总反射率 (1)", "吸收率 (1)", "S11 (1)"]  # top
    z_keys = ["总反射率 (1)", "吸收率 (1)", "S22 (1)"]  # btm

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}, count = {len(arr)}")
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
    s11 = np.empty_like(Z_filtered, dtype=complex)
    s21 = np.empty_like(Z_filtered, dtype=complex)
    R = np.empty_like(Z_filtered, dtype=complex)
    A = np.empty_like(Z_filtered, dtype=complex)
    # 使用直接的循环来更新
    for i in range(Z_filtered.shape[0]):
        for j in range(Z_filtered.shape[1]):
            R[i, j] = Z_filtered[i][j][0][0]
            A[i, j] = Z_filtered[i][j][1][0]
            s11[i, j] = Z_filtered[i][j][2][0]

    from core.process_multi_dim_params_space import extract_basic_analysis_fields
    import matplotlib.pyplot as plt
    from core.prepare_plot import prepare_plot_data

    dataset1 = {
        's11': s11,
        's21': s21,
        'R': R,
        'A': A,
    }
    data_path = prepare_plot_data(
        coords=new_coords, data_class='Spectrum', dataset_list=[dataset1], fixed_params={},
    )

    # fig, ax = plt.subplots(figsize=(1.5, 2))
    # im = ax.imshow(R.real.T, origin='lower', extent=(new_coords['k'][0], new_coords['k'][-1],
    #                                                  new_coords['freq (THz)'][0], new_coords['freq (THz)'][-1]),
    #                cmap='gray', aspect='auto', vmin=0, vmax=1)
    # cbar = fig.colorbar(im, ax=ax, label='R')
    # ax.set_xlabel('k (2π/P)')
    # ax.set_ylabel('Frequency (THz)')
    # plt.savefig('temp.png', bbox_inches='tight', transparent=True, dpi=300)
    # plt.show()

    ################################################################################################################

    fs = 9
    tickdir = 'in'
    font = 'Arial'
    plt.rcParams.update({
        'font.size': fs,
        'font.family': font,
        'xtick.direction': tickdir,
        'ytick.direction': tickdir,
        # 'axes.linewidth': 0.5,
    })

    # freq->wavelength
    fig, ax = plt.subplots(figsize=(2, 2))
    wavelengths = period / (new_coords['f (c/P)'])  # 转换为nm
    print('range of wavelengths (nm):', wavelengths[0], '~', wavelengths[-1])
    # extent = (new_coords['k'][0], new_coords['k'][-1], wavelengths[0], wavelengths[-1])
    extent = (new_coords['k'][0], new_coords['k'][-1], new_coords['f (c/P)'][0], new_coords['f (c/P)'][-1])
    im = ax.imshow(R.real.T, origin='lower', extent=extent,
                     # cmap='Blues', aspect='auto', vmin=0, vmax=1, interpolation='none')
                     cmap='rainbow', aspect='auto', vmin=0, vmax=1, interpolation='none')
    # # plot iso-angle lines (wavelength vs k) -10, -5, 5, 10 degrees
    # iso_angles = [-15, -10, -5, 5, 10, 15]
    # for angle in iso_angles:
    #     wavelength_vals = np.linspace(wavelengths[0], wavelengths[-1], 500)
    #     k_vals = np.sin(np.radians(angle)) * (period / wavelength_vals)
    #     ax.plot(k_vals, wavelength_vals, linestyle='--', label=f'{angle}°', linewidth=1, color='gray')
    #     # text at the end of the line
    #     ax.text(k_vals[-1], wavelength_vals[-1], f'{angle}°', fontsize=6, verticalalignment='bottom')
    # ax.set_xlabel('k (2π/P)')
    # ax.set_ylabel('Wavelength (nm)')
    ax.set_ylim((0.555,0.568))
    plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(2, 2))
    phase_s11 = np.angle(s11)
    im = ax.imshow(phase_s11.T, origin='lower', extent=extent,
                   cmap='hsv', aspect='auto', interpolation='none')
    # cbar = fig.colorbar(im, ax=ax)
    ax.set_ylim((0.555,0.568))
    plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=300)
    plt.show()

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
