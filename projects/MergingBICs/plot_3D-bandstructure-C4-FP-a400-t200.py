from advance_plot_styles.polar_plot import *
from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.data_postprocess.polar_edges import plot_phi_families_split
from core.plot_3D_params_space_plt import *
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

from core.data_postprocess.momentum_space_toolkits import complete_C4_polarization, geom_complete, \
    plot_isofreq_contours2D, sample_fields_along_path, extract_isofreq_paths

c_const = 299792458

if __name__ == '__main__':
    data_path = 'data/FP_PhC-full-14eigens-400nmP-L211,212nm.csv'
    # data_path = 'data/FP_PhC-full-14eigens-400nmP-L214nm.csv'
    df_sample = pd.read_csv(data_path, sep='\t')


    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))


    def norm_freq(freq, period):
        return freq / (c_const / period)


    period = 400
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq,
                                                                                           period=period * 1e-9 * 1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period * 1e-9)
    df_sample["phi (rad)"] = df_sample["phi (rad)"].apply(lambda x: x % np.pi)
    # # 筛选m1<0.1的成分
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
            'buffer (nm)': 212,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            # "频率 (Hz)": {">": 0.0, "<": 0.530},  # 筛选
        }
    )

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
        max_m=9
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
    # new_coords, Z_target10 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=9  # 第n个频率
    # )
    # new_coords, Z_target11 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=10  # 第n个频率
    # )

    from core.process_multi_dim_params_space import extract_basic_analysis_fields, plot_advanced_surface
    import matplotlib.pyplot as plt

    # 提取 band= 的附加场数据
    phi, tanchi, qlog, freq_real = extract_basic_analysis_fields(additional_Z_grouped, z_keys=z_keys, band_index=8)

    m1f, m2f, phi_f, tanchi_f = complete_C4_polarization(new_coords, phi, tanchi)
    (_, _), Z_f = geom_complete(new_coords, Z_target9, mode='C4')
    (_, _), qlog_f = geom_complete(new_coords, qlog, mode='C4')
    # pkl_path = 'rsl/eigensolution/polar_fields.pkl'
    # bundle = dict(m1_full=m1f, m2_full=m2f, Z_full=Z_f, phi_full=phi_f, tanchi_full=tanchi_f, Qlog_full=qlog_f)
    # with open(pkl_path, 'wb') as f:
    #     pickle.dump(bundle, f)
    # print(f"[SAVE] {pkl_path}")
    dataset1 = {
        'eigenfreq': Z_f,
        'phi': phi_f,
        'tanchi': tanchi_f,
        'Qlog': qlog_f,
    }
    data_path = prepare_plot_data(
        coords=new_coords, dataset_list=[dataset1], fixed_params={},
    )

    # 绘图
    Mx, My = np.meshgrid(m1f, m2f, indexing='ij')
    s1 = np.cos(2 * phi_f) * (1 - tanchi_f ** 2) / (1 + tanchi_f ** 2)
    s2 = np.sin(2 * phi_f) * (1 - tanchi_f ** 2) / (1 + tanchi_f ** 2)
    s3 = 2 * tanchi_f / (1 + tanchi_f ** 2)

    fig, ax = plt.subplots(figsize=(3 / 2, 3 / 2))
    ax = plot_polarization_ellipses(
        ax, Mx, My, s1, s2, s3,
        step=(5, 5),  # 适当抽样，防止太密
        scale=1e-2,  # 自动用 0.8*min(dx,dy)
        cmap='RdBu',
        alpha=1, lw=1,
    )
    # 叠加三条等频线
    levels = [0.509, 0.510, 0.511]  # 你想观察的 isofreq
    ax = plot_isofreq_contours2D(ax, m1f, m2f, Z_f, levels=levels,
                                 colors=['k', 'k', 'k'],
                                 linewidths=1.0)
    ax.set_xticklabels([]);
    ax.set_yticklabels([])
    plt.savefig('temp.svg')
    plt.show()

    fig, ax = plt.subplots(figsize=(3 / 2, 3 / 2))
    # np.sin(2*phi)>0 绘制淡绿色区域， np.sin(2*phi)<0 绘制淡红色区域
    color_1 = 'lightgreen'
    color_2 = 'lightcoral'
    # 绘制区域
    ax.contourf(m1f, m2f, np.sin(2 * phi_f) > 0, levels=[-0.5, 0.5], colors=[color_2, color_1], alpha=0.5)
    ax = plot_phi_families_split(ax, m1f, m2f, phi_f, overlay=None, lw=1)
    plt.savefig('temp.svg')
    plt.show()

    # 提取等频线
    paths = extract_isofreq_paths(m1f, m1f, Z_f, level=0.510)
    # 沿每条等频线插值采样
    fields = {'phi': phi_f, 'chi': tanchi_f, 'Q': qlog_f, 'freq': Z_f.real}
    samples_list = []
    for p in paths:
        samp = sample_fields_along_path(m1f, m2f, fields, p, npts=400)
        samples_list.append(samp)
    # 画曲线
    fig, ax = plt.subplots(figsize=(3 / 2, 3 / 2))
    ax.plot(samp['s'], samp['phi'])
    fig, ax = plt.subplots(figsize=(3 / 2, 3 / 2))
    ax.plot(samp['s'], samp['chi'])
    fig, ax = plt.subplots(figsize=(3 / 2, 3 / 2))
    ax.plot(samp['s'], samp['Q'])
    plt.show()

    fig, ax = plt.subplots(figsize=(3 / 2, 3 / 2))
    ax.imshow(qlog_f.T, extent=(m1f.min(), m1f.max(), m2f.min(), m2f.max()), origin='lower', cmap='hot', aspect='equal')
    plt.savefig('temp.svg')
    plt.show()

    fig, ax = plt.subplots(figsize=(3 / 2, 3 / 2))
    ax.imshow(phi_f.T, extent=(m1f.min(), m1f.max(), m2f.min(), m2f.max()), origin='lower', cmap='twilight',
              aspect='equal')
    plt.savefig('temp.svg')
    plt.show()

    fig = plt.figure(figsize=(3, 4))
    ax = fig.add_subplot(111, projection='3d')
    mapping = {
        'cmap': 'hot',
        # 'z2': {'vmin': a, 'vmax': b},  # 可选；未给则自动取数据范围
        # 'z3': {'vmin': c, 'vmax': d},  # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
    }
    ax = plot_advanced_surface(
        ax, mx=m1f, my=m2f,
        mapping=mapping,
        z1=Z_f,
        z2=qlog_f,
        elev=30, azim=25,
        font_size=9,
    )
    plt.savefig('temp.svg')
    plt.show()
