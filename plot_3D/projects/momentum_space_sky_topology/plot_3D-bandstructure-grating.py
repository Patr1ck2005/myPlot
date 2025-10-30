from plot_3D.core.data_postprocess.data_filter import advanced_filter_eigensolution
from plot_3D.core.data_postprocess.data_grouper import *
from plot_3D.core.plot_3D_params_space_plt import *
from plot_3D.core.process_multi_dim_params_space import *

import numpy as np

from plot_3D.advance_plot_styles.polar_plot import plot_on_poincare_sphere, \
    plot_polarization_ellipses

c_const = 299792458


if __name__ == '__main__':
    data_path = 'data/grating-full-7eigen.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    def norm_freq(freq, period):
        return freq/(c_const/period)
    period = 500
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=period*1e-9*1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period*1e-9)
    df_sample["up_phi (rad)"] = df_sample["up_phi (rad)"].apply(lambda x: x % np.pi)
    # # 筛选m1<0.1的成分
    # df_sample = df_sample[df_sample["m1"] < 0.05]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "up_tanchi (1)", "up_phi (rad)", "fake_factor (1)", "频率 (Hz)"]

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
            # 'buffer (nm)': 430,
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
        max_m=5
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
    # new_coords, Z_target6 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=5  # 第n个频率
    # )
    # new_coords, Z_target7 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=6  # 第n个频率
    # )


    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")

    # 暂时简单使用3D绘制数据
    import matplotlib.pyplot as plt
    fs = 9
    plt.rcParams.update({'font.size': fs})

    fig = plt.figure(figsize=(3, 4))
    ax = fig.add_subplot(111, projection='3d')
    m1_vals = new_coords['m1']
    m2_vals = new_coords['m2']

    target_phi = np.zeros(((additional_Z_grouped.shape[0]), (additional_Z_grouped.shape[1])))
    target_tanchi = np.zeros(((additional_Z_grouped.shape[0]), (additional_Z_grouped.shape[1])))
    target_Qfactor_log = np.zeros(((additional_Z_grouped.shape[0]), (additional_Z_grouped.shape[1])))
    target_freq = np.zeros(((additional_Z_grouped.shape[0]), (additional_Z_grouped.shape[1])))
    # 把形状51,51的列表中的[1][3]数据提取出来
    for i in range(additional_Z_grouped.shape[0]):
        for j in range(additional_Z_grouped.shape[1]):
            target_phi[i][j] = additional_Z_grouped[i][j][0][3]
            target_tanchi[i][j] = additional_Z_grouped[i][j][0][2]
            target_Qfactor_log[i][j] = np.log10(additional_Z_grouped[i][j][0][1])
            target_freq[i][j] = additional_Z_grouped[i][j][0][0].real
            print(target_freq[i][j])

    M1, M2 = np.meshgrid(m1_vals, m2_vals, indexing='ij')

    # fig = plt.figure()
    # plt.imshow(target2_phi)
    # plt.colorbar()
    # plt.show()

    # 绘制多个 Z_target, 同时使用 Qfactor 作为颜色映射
    Z_targets = [Z_target1]
    additional_Zs = [target_phi,]
    # Z_targets = [Z_target7,]

    from polar_postprocess import geom_complete, complete_C4_polarization

    pkl_path = 'rsl/eigensolution/polar_fields.pkl'
    Z_target_complex = Z_target1
    m1, m2 = new_coords['m1'], new_coords['m2']

    m1f, m2f, phi_f, chi_f = complete_C4_polarization(m1, m2, target_phi, target_tanchi)
    (_, _), Z_f = geom_complete({'m1':m1, 'm2':m2}, Z_target_complex, mode='C4')
    (_, _), target_Qfactor_log = geom_complete({'m1':m1, 'm2':m2}, target_Qfactor_log, mode='C4')
    # Q_R  = np.concatenate([np.flip(target_Qfactor_log[:,1:],1), target_Qfactor_log], 1)
    # Q_f  = np.concatenate([np.flip(Q_R[1:],0), Q_R], 0)  # Q 偶（几何镜像）
    # 预览结果
    fig = plt.figure(figsize=(6,5))
    plt.imshow(phi_f.T, origin='lower', extent=(m1f[0], m1f[-1], m2f[0], m2f[-1]), aspect='auto', cmap='hsv')
    plt.colorbar(label='φ (rad)'); plt.title('Completed φ field preview'); plt.xlabel('m1'); plt.ylabel('m2')
    plt.show()
    fig = plt.figure(figsize=(6,5))
    S1 = (1-chi_f**2)/(1+chi_f**2)*np.cos(2*phi_f)
    S2 = (1-chi_f**2)/(1+chi_f**2)*np.sin(2*phi_f)
    S3 = 2*chi_f/(1+chi_f**2)
    plt.imshow(S3.T, origin='lower', extent=(m1f[0], m1f[-1], m2f[0], m2f[-1]), aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='S3'); plt.title('Completed S3 field preview'); plt.xlabel('m1'); plt.ylabel('m2')
    plt.show()
    # 1) 椭圆贴片（独立成图）
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    kx, ky = np.meshgrid(m1f, m2f, indexing='ij')
    plot_polarization_ellipses(ax1, kx, ky, S1.T, S2.T, S3.T, S0=None, step=(3,3), scale=0.01)
    # 找到C点: S3 最大或最小的位置
    max_idx = np.unravel_index(np.argmax(S3), S3.shape)
    print("Max S3 at:", (m1f[max_idx[1]], m2f[max_idx[0]]), "Value:", S3[max_idx])
    ax1.plot(m1f[max_idx[0]], m2f[max_idx[1]], 'ro', markersize=8, label='C Point (Max S3)')
    min_idx = np.unravel_index(np.argmin(S3), S3.shape)
    ax1.plot(m1f[min_idx[0]], m2f[min_idx[1]], 'bo', markersize=8, label='C Point (Min S3)')
    plt.show()
    # 5) 投到 Poincaré 球面（独立成图）
    fig5 = plt.figure(figsize=(6, 6))
    ax5 = fig5.add_subplot(111, projection='3d')
    plot_on_poincare_sphere(ax5, S1, S2, S3, S0=None, step=(1,1),
                            c_by='s3', cmap='RdBu', clim=(-1,1),
                            s=8, alpha=0.9, sphere_style='wire')
    plt.show()
    bundle = dict(m1_full=m1f, m2_full=m2f, Z_full=Z_f, phi_full=phi_f, chi_full=chi_f, Q_full=target_Qfactor_log)
    with open(pkl_path, 'wb') as f: pickle.dump(bundle, f)
    print(f"[SAVE] {pkl_path}")

    for idx, Z_target in enumerate(Z_targets):
        FREQ = np.empty(M1.shape)
        Qfactor = np.empty(M1.shape)
        Phi = np.empty(M1.shape)
        tanchi = np.empty(M1.shape)
        for i in range(M1.shape[0]):
            for j in range(M1.shape[1]):
                val = Z_target[i, j]
                FREQ[i, j] = val.real
                Qfactor[i, j] = np.log10(val.real/val.imag/2 if val.imag != 0 else 1e6)
                # Qfactor[i, j] = np.log10(additional_Z_grouped[i][j][5][1])
                # Qfactor[i, j] = target_Qfactor_log[i][j]
                # Phi[i, j] = additional_Zs[idx][i, j]
                # tanchi[i, j] = additional_Z_grouped[i][j][1][2]
                # tanchi[i, j] = target_tanchi[i][j]
        surf_color_data = Qfactor
        # surf_color_data = Phi
        # surf_color_data = tanchi
        # surf_colors = plt.cm.RdBu((surf_color_data - -1) / 2)
        surf_colors = plt.cm.hot((surf_color_data - np.min(surf_color_data)) / (np.max(surf_color_data) - np.min(surf_color_data)))
        # surf_colors = plt.cm.hot((surf_color_data - 2) / (6 - 2))
        # surf_colors = plt.cm.hsv((surf_color_data - np.min(surf_color_data)) / (np.max(surf_color_data) - np.min(surf_color_data)))
        # surf_colors = plt.cm.twilight((surf_color_data - np.min(surf_color_data)) / (np.max(surf_color_data) - np.min(surf_color_data)))
        surf = ax.plot_surface(M1, M2, FREQ, facecolors=surf_colors, rstride=1, cstride=1, alpha=0.8, label=f'Band {idx+1}')
    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap='twilight')
    mappable.set_array(surf_color_data)
    cbar = plt.colorbar(mappable, ax=ax)
    ax.set_xlabel('m1')
    ax.set_ylabel('m2')
    ax.set_zlabel('Frequency (normalized)')

    # 调整视角
    ax.view_init(elev=30, azim=45-20)

    # 设置比例
    ax.set_box_aspect([1, 1, 1])  # 设置xyz轴的比例

    plt.savefig('temp.svg', transparent=True)
    plt.show()







