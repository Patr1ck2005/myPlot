import matplotlib.pyplot as plt

from plot_3D.core.data_postprocess.data_grouper import group_surfaces_one_sided_hungarian
from plot_3D.core.plot_3D_params_space_plt import *
from plot_3D.core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from plot_3D.core.process_multi_dim_params_space import *

import numpy as np

from utils.advanced_color_mapping import plot_S1S2S3_color, convert_complex2rbg
from utils.functions import skyrmion_density, skyrmion_number, divide_regions_by_zero

c_const = 299792458

if __name__ == '__main__':

    def load_data(data_path, condition_keys=None):
        df_sample = pd.read_csv(data_path, sep='\t')
        if condition_keys is not None:
            df_sample = df_sample[df_sample[condition_keys[0]] == condition_keys[1]]

        # 指定用于构造网格的参数以及目标数据列
        param_keys = ["m1", "m2"]
        z_keys = [
            "abs(cx)^2+abs(cy)^2 (kg^2*m^2/(s^6*A^2))",
            "(abs(cx)^2-abs(cy)^2)/(abs(cx)^2+abs(cy)^2) (1)",
            "2*real(conj(cx)*cy)/(abs(cx)^2+abs(cy)^2) (1)",
            "2*imag(conj(cx)*cy)/(abs(cx)^2+abs(cy)^2) (1)"
        ]

        # 构造数据网格，此处不进行聚合，每个单元格保存列表
        grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys)
        print("网格参数：")
        for key, arr in grid_coords.items():
            print(f"  {key}: {arr}")
        print("数据网格 Z 的形状：", Z.shape)

        # 创建一个新的数组，用于存储更新后的结果
        Z_new = np.empty_like(Z, dtype=float)
        Z_S123 = np.stack((Z_new, Z_new, Z_new, Z_new), axis=-1)
        # 使用直接的循环来更新 Z_new
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z_S123[i, j, 0] = Z[i, j][0][0]
                Z_S123[i, j, 1] = Z[i, j][1][0]
                Z_S123[i, j, 2] = Z[i, j][2][0]
                Z_S123[i, j, 3] = Z[i, j][3][0]

        return grid_coords, Z_S123


    # grid_coords1, Z_S123 = load_data(
    #     data_path := 'data/vortexSE/Hex_hole-center-0.4r_0.1t-0.05kr-freqs.csv',
    #     condition_keys=['freq (THz)', 352]  # 350-355
    # )
    # grid_coords1, Z_S123 = load_data(data_path := 'data/vortexSE/Hex_hole-corner-0.4r_0.1t-0.05kr.csv')

    # grid_coords1, Z_S123 = load_data(data_path := 'data/vortexSE/Hex_disk-center-0.4r_0.1t-0.05kr.csv')
    # grid_coords1, Z_S123 = load_data(data_path := 'data/vortexSE/Hex_disk-corner-0.4r_0.1t-0.05kr.csv')
    # grid_coords1, Z_S123 = load_data(data_path := 'data/vortexSE/Hex_disk-corner-0.4r_0.1t-0.15kr.csv')
    # grid_coords1, Z_S123 = load_data(data_path := 'data/vortexSE/Hex_disk-corner-0.4r_0.1t-0.20kr.csv')
    # grid_coords1, Z_S123 = load_data(data_path := 'data/vortexSE/Hex_disk-off-0.4r_0.1t-close-0.05kr.csv')
    # grid_coords1, Z_S123 = load_data(data_path := 'data/vortexSE/Hex_disk-off-0.4r_0.1t-close-0.15kr.csv')

    # grid_coords1, Z_S123 = load_data(data_path := 'data/vortexSE/rect_hole-cross_mid-0.4r_0.1t-0.20kr.csv')
    grid_coords1, Z_S123 = load_data(
        data_path := 'data/vortexSE/rect_hole-cross_mid-0.4r_0.1t-0.05kr-2freqs.csv',
        condition_keys=['freq (THz)', 379.2]  # 341.4 379.2
    )

    # grid_coords1, Z_S123 = load_data(
    #     data_path := 'data/vortexSE/rect_disk-center-0.28r_1.36t-0.05kr-2freqs.csv',
    #     condition_keys=['freq (THz)', 273]  # 273 278
    # )

    global_save_pre = data_path.removesuffix('.csv').split('/')[-1]


    def default_momentum_space_show(im, save_tag='default', show=False):
        fig = plt.figure(figsize=(3, 3), dpi=100)
        plt.imshow(im, origin='lower', extent=[m1_min, m1_max, m2_min, m2_max])
        plt.xlabel(r'$k_x (2\pi/a)$')
        plt.ylabel(r'$k_y (2\pi/a)$')
        # plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'rsl/vortexSE/{save_tag}.png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.savefig(f'rsl/vortexSE/{save_tag}.svg', bbox_inches='tight', pad_inches=0.1)
        if show:
            plt.show()


    def binary_momentum_space_show(reals, save_tag='default', max_abs=None, show=False, cmap='RdBu'):
        fig = plt.figure(figsize=(4, 3), dpi=100)
        max_abs = np.max(np.abs(reals))
        plt.imshow(
            reals, origin='lower', extent=[m1_min, m1_max, m2_min, m2_max],
            vmin=-max_abs, vmax=max_abs,
            cmap=cmap
        )
        plt.colorbar()
        # plt.axis('equal')
        plt.xlabel(r'$k_x (2\pi/a)$')
        plt.ylabel(r'$k_y (2\pi/a)$')
        plt.tight_layout()
        plt.savefig(f'rsl/vortexSE/{save_tag}.png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.savefig(f'rsl/vortexSE/{save_tag}.svg', bbox_inches='tight', pad_inches=0.1)
        if show:
            plt.show()


    def intensity_momentum_space_show(reals, save_tag='default', max_value=None, show=False):
        fig = plt.figure(figsize=(4, 3), dpi=100)
        max_value = np.max(reals)
        plt.imshow(
            reals, origin='lower', extent=[m1_min, m1_max, m2_min, m2_max],
            vmin=0, vmax=max_value,
            cmap='gray'
        )
        plt.colorbar()
        plt.xlabel(r'$k_x (2\pi/a)$')
        plt.ylabel(r'$k_y (2\pi/a)$')
        plt.tight_layout()
        plt.savefig(f'rsl/vortexSE/{save_tag}.png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.savefig(f'rsl/vortexSE/{save_tag}.svg', bbox_inches='tight', pad_inches=0.1)
        if show:
            plt.show()


    grid_coords = grid_coords1
    m1_min, m1_max = grid_coords['m1'][0], grid_coords['m1'][-1]
    m2_min, m2_max = grid_coords['m2'][0], grid_coords['m2'][-1]

    S0 = Z_S123[:, :, 0]
    intensity_momentum_space_show(S0, save_tag=f'{global_save_pre}-S0')
    S1 = Z_S123[:, :, 1]
    binary_momentum_space_show(S1, save_tag=f'{global_save_pre}-S1', max_abs=1, cmap='PiYG')
    S2 = Z_S123[:, :, 2]
    binary_momentum_space_show(S2, save_tag=f'{global_save_pre}-S2', max_abs=1, cmap='PiYG')
    S3 = Z_S123[:, :, 3]
    binary_momentum_space_show(S3, save_tag=f'{global_save_pre}-S3', max_abs=1, cmap='PiYG')

    S3_mask = S3 < 0
    #
    # S3[S3_mask] = 0

    rgb = plot_S1S2S3_color(S1, S2, S3, s3_mode='-11', show=False, extent=[m1_min, m1_max, m2_min, m2_max])
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
