from core.plot_3D_params_space_plt import *

import numpy as np

from core.data_postprocess.momentum_space_toolkits import load_bundle
from utils.advanced_color_mapping import map_s1s2s3_color
from utils.functions import skyrmion_density, skyrmion_number, divide_regions_by_zero

c_const = 299792458

fs = 14  # 字体大小
plt.rcParams.update({'font.size': fs})

if __name__ == '__main__':

    def load_data(data_path, condition_keys=None):
        # 假设已加载数据：
        data = load_bundle(data_path)
        m1, m2 = data['m1_full'], data['m2_full']
        FREQ = np.real(data['Z_full'])
        phi = data['phi_full']
        chi = data['chi_full']

        # 创建一个新的数组，用于存储更新后的结果
        Z_new = np.empty_like(phi, dtype=float)
        Z_new_complex = np.empty_like(phi, dtype=complex)
        Z_S123 = np.stack((Z_new, Z_new, Z_new, Z_new), axis=-1)
        Z_CP = np.stack((Z_new_complex, Z_new_complex), axis=-1)
        # Z_S123[:, :, 0] = ...
        Z_S123[:, :, 1] = (1-chi**2) * np.cos(2*phi) / (1+chi**2)
        Z_S123[:, :, 2] = (1-chi**2) * np.sin(2*phi) / (1+chi**2)
        Z_S123[:, :, 3] = 2*chi / (1+chi**2)
        # Z_CP[:, :, 3] = 2*chi / (1+chi**2)

        return m1, m2, Z_S123, Z_CP


    data_path = 'rsl/eigensolution/polar_fields.pkl'

    lattice_const = 400  # nm
    freq = 325
    m1, m2, Z_S123, Z_CP = load_data(
        data_path
    )

    m2_min, m2_max = np.min(m2), np.max(m2)
    m1_min, m1_max = np.min(m1), np.max(m1)

    # 定义光速（m/s）
    c_const = 3e8  # 如果你有其他值，请替换
    # 计算波长（m）
    wavelength = c_const / (freq * 1e12)  # freq: THz → Hz
    # 归一化 k 值
    k_values = [0.3, 0.5]
    # 计算入射角度（修正公式：arcsin(k * λ / a)）
    a = lattice_const * 1e-9  # nm → m
    angles = []
    for k in k_values:
        sin_theta = k * wavelength / a
        if sin_theta > 1:
            print(f"警告: 对于 k={k}, sinθ = {sin_theta:.4f} > 1，无法计算真实角度（全内反射或无效）")
            angles.append(np.nan)  # 或处理为 90°
        else:
            theta = np.degrees(np.arcsin(sin_theta))
            angles.append(theta)
    for k, angle in zip(k_values, angles):
        if np.isnan(angle):
            print(f"归一化 k={k} 对应的入射角度: 无效 (sinθ >1)")
        else:
            print(f"归一化 k={k} 对应的入射角度: {angle:.2f} degrees")
    # --- 新功能: 逆向计算 (θ → k) ---
    theta_target = 60  # degrees，可以修改为你想要的角度
    sin_theta = np.sin(np.radians(theta_target))  # 转换为弧度计算 sin
    k_reverse = sin_theta * (a / wavelength)  # 逆向公式
    # 检查 k 是否合理（假设 k 应在 0~1 范围内，根据你的模拟调整）
    if k_reverse < 0 or k_reverse > 1:
        print(f"警告: 计算的 k={k_reverse:.4f} 超出合理范围 (0~1)，可能无物理意义！")
    elif k_reverse > 0.5:
        print(f"注意: 计算的 k={k_reverse:.4f} >0.5，可能接近布里渊区边界。")
    # 输出逆向结果
    print(f"入射角度 θ={theta_target} degrees 对应的归一化 k: {k_reverse:.4f}")


    global_save_pre = data_path.removesuffix('.csv').split('/')[-1]

    def default_momentum_space_show(im, save_tag='default', show=False):
        fig = plt.figure(figsize=(2, 2), dpi=100)
        plt.imshow(im, origin='lower', extent=[m1_min, m1_max, m2_min, m2_max])
        plt.xlabel(r'$k_x (2\pi/a)$')
        plt.ylabel(r'$k_y (2\pi/a)$')
        # plt.axis('off')
        # plt.tight_layout()
        plt.savefig(f'./{save_tag}.png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.savefig(f'./{save_tag}.svg', bbox_inches='tight', pad_inches=0.1, transparent=True)
        if show:
            plt.show()

    def phase_momentum_space_show(im, save_tag='phase', show=False):
        fig = plt.figure(figsize=(2, 2), dpi=100)
        plt.imshow(im, origin='lower', extent=[m1_min, m1_max, m2_min, m2_max], cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.xlabel(r'$k_x (2\pi/a)$')
        plt.ylabel(r'$k_y (2\pi/a)$')
        # plt.axis('off')
        # plt.tight_layout()
        plt.savefig(f'./{save_tag}.png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.savefig(f'./{save_tag}.svg', bbox_inches='tight', pad_inches=0.1, transparent=True)
        if show:
            plt.show()


    def binary_momentum_space_show(reals, save_tag='default', max_abs=None, show=False, cmap='RdBu'):
        fig = plt.figure(figsize=(2, 2), dpi=100)
        max_abs = np.max(np.abs(reals))
        plt.imshow(
            reals, origin='lower', extent=[m1_min, m1_max, m2_min, m2_max],
            vmin=-max_abs, vmax=max_abs,
            cmap=cmap
        )
        # plt.colorbar()
        # plt.axis('equal')
        plt.xlabel(r'$k_x (2\pi/a)$')
        plt.ylabel(r'$k_y (2\pi/a)$')
        # plt.tight_layout()
        plt.savefig(f'./{save_tag}.png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.savefig(f'./{save_tag}.svg', bbox_inches='tight', pad_inches=0.1, transparent=True)
        if show:
            plt.show()


    def intensity_momentum_space_show(reals, save_tag='default', max_value=None, show=False):
        fig = plt.figure(figsize=(2, 2), dpi=100)
        max_value = np.max(reals)
        plt.imshow(
            reals, origin='lower', extent=[m1_min, m1_max, m2_min, m2_max],
            vmin=0, vmax=max_value,
            cmap='gray'
        )
        # plt.colorbar()
        plt.xlabel(r'$k_x (2\pi/a)$')
        plt.ylabel(r'$k_y (2\pi/a)$')
        # plt.tight_layout()
        plt.savefig(f'./{save_tag}.png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.savefig(f'./{save_tag}.svg', bbox_inches='tight', pad_inches=0.1, transparent=True)
        if show:
            plt.show()



    # c_LCP = Z_CP[:, :, 0]
    # phase_momentum_space_show(np.angle(c_LCP), save_tag=f'{global_save_pre}-c_LCP-phase')
    # intensity_momentum_space_show(np.abs(c_LCP)**2, save_tag=f'{global_save_pre}-c_LCP-intensity')
    # binary_momentum_space_show(np.real(c_LCP), save_tag=f'{global_save_pre}-c_LCP-real')
    # c_RCP = Z_CP[:, :, 1]
    # phase_momentum_space_show(np.angle(c_RCP), save_tag=f'{global_save_pre}-c_RCP-phase')
    # intensity_momentum_space_show(np.abs(c_RCP)**2, save_tag=f'{global_save_pre}-c_RCP-intensity')
    # binary_momentum_space_show(np.real(c_RCP), save_tag=f'{global_save_pre}-c_RCP-real')
    # S0 = Z_S123[:, :, 0]
    # intensity_momentum_space_show(S0, save_tag=f'{global_save_pre}-S0')
    S1 = Z_S123[:, :, 1]
    # binary_momentum_space_show(S1, save_tag=f'{global_save_pre}-S1', max_abs=1, cmap='PiYG')
    S2 = Z_S123[:, :, 2]
    # binary_momentum_space_show(S2, save_tag=f'{global_save_pre}-S2', max_abs=1, cmap='PiYG')
    S3 = Z_S123[:, :, 3]
    binary_momentum_space_show(S3, save_tag=f'{global_save_pre}-S3', max_abs=1, cmap='PiYG')

    S3_mask = S3 < 0
    #
    # S3[S3_mask] = 0

    rgb = map_s1s2s3_color(S1, S2, S3, s3_mode='-11', show=False, extent=[m1_min, m1_max, m2_min, m2_max])
    default_momentum_space_show(rgb, save_tag=f'{global_save_pre}-S1S2S3')


    # 计算斯格明子密度
    n_sk = skyrmion_density(S1, S2, S3)

    # 网格间隔（假设是均匀的）
    dx = dy = 1  # 这里假设每个网格的间隔为1

    # 计算斯格明子数
    S3_masks = divide_regions_by_zero(S3, visualize=True)
    mid_mask = S3_masks['middle_mask']
    # n_sk[mid_mask] = -100
    s = skyrmion_number(n_sk, dx, dy, mask=mid_mask)

    binary_momentum_space_show(n_sk, save_tag=f'{global_save_pre}-n_sk-s={s}', cmap='RdBu', show=True)

    print(f"斯格明子数: {s}")
