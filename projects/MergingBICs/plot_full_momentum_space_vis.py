import numpy as np

from core.plot_cls import MomentumSpaceEigenPolarizationPlotter
from core.plot_workflow import PlotConfig


if __name__ == '__main__':
    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P400_T200_L250_R0.4.pkl"
    # config = PlotConfig(
    #     plot_params={},
    #     annotations={'show_tick_labels': True},
    # )
    # config.update(figsize=(1.25, 1.25))
    # config.update(tick_direction='in')
    # plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path)
    # plotter.load_data()
    # plotter.prepare_data()
    #
    # plotter.new_2d_fig()
    # plotter.plot_polarization_ellipses(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.prepare_chi_phi_data()
    # plotter.plot_phi_families_regimes(index=0)
    # plotter.plot_phi_families_split(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_phi(index=0, cmap='twilight')
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_advanced_color_mapping(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(3.8, 1))
    # plotter.new_2d_fig()
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.451)
    # plotter.ax.plot(samples_list[0]['s']/np.max(samples_list[0]['s']), samples_list[0]['phi'], 'r-', linewidth=1, alpha=0.5)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.4535)
    # plotter.ax.plot(samples_list[0]['s']/np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.456)
    # plotter.ax.plot(samples_list[0]['s']/np.max(samples_list[0]['s']), samples_list[0]['phi'], 'b-', linewidth=1, alpha=0.5)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(tick_direction='out')
    # plotter.new_2d_fig()
    # plotter.imshow_qlog(index=0, cmap='magma', vmax=7)
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.451, 0.4535, 0.456), colors=('r', 'k', 'b'))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(2, 2))
    # plotter.new_3d_fig()
    # mapping = {
    #             'cmap': 'magma',
    #             'z2': {'vmin': 2, 'vmax': 7},  # 可选；未给则自动取数据范围
    #             # 'z3': {'vmin': c, 'vmax': d},  # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
    #         }
    # plotter.plot_3D_surface(index=0, shade=False, mapping=mapping)
    # plotter.ax.set_box_aspect([1, 1, 0.5])
    # plotter.config.annotations = {'show_tick_labels': False}
    # plotter.add_annotations()
    # plotter.save_and_show()


    data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-RectHole_P400_T200_L180_D0.6.pkl"
    config = PlotConfig(
        plot_params={},
        annotations={'show_tick_labels': True, 'xlim': (-0.10, 0.10), 'ylim': (-0.10, 0.10)},
    )
    config.update(tick_direction='in', figsize=(1.25, 1.25))
    plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()
    plotter.interpolate_data(factor=15)
    plotter.prepare_chi_phi_data()

    plotter.compute_cross_polarization_conversion(index=0, freq=0.453)

    from core.fourier_module.ms_propagation_toolkit import *
    import matplotlib.pyplot as plt
    m1 = plotter.m1
    m2 = plotter.m2
    U0 = plotter.cross_conversion  # 初始角谱（复振幅）
    # 做圆形裁剪. 按照比例radius_prop保留中心区域
    radius_max_prop = 0.20
    radius_min_prop = 0
    max_radius = radius_max_prop * np.min([np.max(m1) - np.min(m1), np.max(m2) - np.min(m2)])
    min_radius = radius_min_prop * np.min([np.max(m1) - np.min(m1), np.max(m2) - np.min(m2)])
    M1, M2 = np.meshgrid(m1, m2, indexing='xy')
    mask_circle = M1 ** 2 + M2 ** 2 <= max_radius ** 2
    mask_circle &= M1 ** 2 + M2 ** 2 >= min_radius ** 2
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
    # 可视化：np.abs(Exy) 或 np.angle(Exy)
    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    real_U0 = np.real(U0)
    # 对于关于0对称但是极小的数据, 取log绝对值后可视化效果更好
    log_real_U0 = np.sign(real_U0) * np.log10(np.abs(real_U0) + 1e-5)
    from matplotlib.colors import SymLogNorm

    im = ax.imshow(real_U0.T, origin='lower', extent=(m1[0], m1[-1], m2[0], m2[-1]),
                   cmap='RdBu', norm=SymLogNorm(linthresh=1e-2, vmin=-1, vmax=1))
                   # cmap='RdBu', vmin=-1, vmax=1)
    # im = ax.imshow(np.abs(U0).T, origin='lower', extent=(m1[0], m1[-1], m2[0], m2[-1]),
    #                # cmap='RdBu', norm=SymLogNorm(linthresh=1e-2, vmin=-1, vmax=1))
    #                cmap='magma', vmin=0, vmax=1)
    ax.set_xlim(-0.10, 0.10)
    ax.set_ylim(-0.10, 0.10)
    # cbar = fig.colorbar(im, ax=ax)
    plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=1000)
    plt.show()
    # 1) 单个 z 面
    z = 0.00
    norm_f = 0.453
    # m1 的单位为 2*π/P
    # 频率的单位为 c/P
    norm_m1 = m1 / norm_f
    norm_m2 = m2 / norm_f
    # # 可视化：np.abs(Exy) 或 np.angle(Exy)
    # fig, ax = plt.subplots(figsize=(1.25, 1.25), dpi=200)
    # real_U0 = np.real(U0)
    # # 对于关于0对称但是极小的数据, 取log绝对值后可视化效果更好
    # log_real_U0 = np.sign(real_U0) * np.log10(np.abs(real_U0) + 1e-5)
    # from matplotlib.colors import SymLogNorm
    # im = ax.imshow(real_U0.T, origin='lower', extent=(m1[0], m1[-1], m2[0], m2[-1]),
    #                cmap='RdBu', norm=SymLogNorm(linthresh=1e-2, vmin=-1, vmax=1))
    # plt.show()
    Az, (x, y, Exy) = angular_spectrum_propagate(norm_m1, norm_m2, U0, z, k0=1, time_sign=+1)
    # 可视化：np.abs(Exy) 或 np.angle(Exy)
    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    # im = ax.imshow(np.abs(Exy).T**2, origin='lower', extent=(x[0], x[-1], y[0], y[-1]),
    #                cmap='magma')
    im = ax.imshow(np.real(Exy).T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]),
                   cmap='RdBu')
    # rgb = map_complex2rbg(Exy)
    # im = ax.imshow(rgb, origin='lower', extent=(x[0], x[-1], y[0], y[-1]))
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=1000)
    plt.show()

    # 2) XZ 纵截面（y=0 切片）
    # z_list = np.linspace(0, 5e3, 200)
    z_list = np.linspace(-1e4, 1e4, 20)
    z_list = np.linspace(-1e4, 0, 100)
    # z_list = np.linspace(0, 1e4, 100)
    x, z, Exz = angular_spectrum_xz_slice(m1, m2, U0, z_list, k0=1, y0=0.0, include_evanescent=False, time_sign=-1)
    # 可视化：np.abs(Exz) 或 np.angle(Exz)
    fig, ax = plt.subplots(figsize=(4, 1.25))
    im = ax.imshow(np.abs(Exz).T**2, origin='lower', extent=(z[0], z[-1], x[0], x[-1]),
                   aspect='auto', cmap='magma')
    ax.set_ylim(-1.5e3, 1.5e3)
    plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=1000)
    plt.show()

    # plotter.new_2d_fig()
    # plotter.plot_polarization_ellipses(index=0, step=(5, 5))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.plot_phi_families_regimes(index=0)
    # plotter.plot_phi_families_split(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_phi(index=0, cmap='twilight')
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.453), colors=('k'))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.453)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1, alpha=1)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['tanchi'], 'k-', linewidth=1, alpha=1)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['qlog'], 'k-', linewidth=1, alpha=1)
    # # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_advanced_color_mapping(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(tick_direction='out')
    # plotter.new_2d_fig()
    # plotter.imshow_qlog(index=0, cmap='magma', vmin=2, vmax=7)
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.451, 0.454, 0.462), colors=('r', 'k', 'b'))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # def regulate_angle(phi_array, min_val=0.0, period=np.pi):
    #     regulated_phi = (phi_array - min_val) % period + min_val
    #     return regulated_phi
    #
    # plotter.config.update(figsize=(3.8, 1))
    # plotter.new_2d_fig()
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.451)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), regulate_angle(samples_list[0]['phi'], min_val=-np.pi/4), 'r-', linewidth=1, alpha=0.5)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.454)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), regulate_angle(samples_list[0]['phi'], min_val=-np.pi/4), 'k-', linewidth=1)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.462)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), regulate_angle(samples_list[0]['phi'], min_val=-np.pi/4), 'b-', linewidth=1, alpha=0.5)
    # # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(2, 2))
    # plotter.new_3d_fig()
    # mapping = {
    #     'cmap': 'magma',
    #     'z2': {'vmin': 2, 'vmax': 7},  # 可选；未给则自动取数据范围
    #     # 'z3': {'vmin': c, 'vmax': d},  # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
    # }
    # plotter.plot_3D_surface(index=0, shade=False, mapping=mapping, elev=60)
    # plotter.ax.set_box_aspect([1, 1, 0.5])
    # plotter.config.annotations = {'show_tick_labels': False}
    # plotter.add_annotations()
    # plotter.save_and_show()



    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P450_T200_L250_R0.3.pkl"
    # config = PlotConfig(
    #     plot_params={},
    #     annotations={'show_tick_labels': True, 'xlim': (-0.15, 0.15), 'ylim': (-0.15, 0.15)},
    # )
    # config.update(tick_direction='in', figsize=(1.25, 1.25))
    # plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path)
    # plotter.load_data()
    # plotter.prepare_data()
    # plotter.interpolate_data(factor=5)
    # plotter.prepare_chi_phi_data()
    #
    # plotter.compute_cross_polarization_conversion(index=0, freq=0.52)
    #
    # from core.fourier_module.ms_propagation_toolkit import *
    # import matplotlib.pyplot as plt
    # m1 = plotter.m1
    # m2 = plotter.m2
    # U0 = plotter.cross_conversion  # 初始角谱（复振幅）
    # # 可视化：np.abs(Exy) 或 np.angle(Exy)
    # fig, ax = plt.subplots(figsize=(1.25, 1.25), dpi=200)
    # real_U0 = np.real(U0)
    # # 对于关于0对称但是极小的数据, 取log绝对值后可视化效果更好
    # log_real_U0 = np.sign(real_U0) * np.log10(np.abs(real_U0) + 1e-5)
    # from matplotlib.colors import SymLogNorm
    # im = ax.imshow(real_U0.T, origin='lower', extent=(m1[0], m1[-1], m2[0], m2[-1]),
    #                cmap='RdBu', norm=SymLogNorm(linthresh=1e-2, vmin=-1, vmax=1))
    # ax.set_xlim(-0.15, 0.15)
    # ax.set_ylim(-0.15, 0.15)
    # cbar = fig.colorbar(im, ax=ax)
    # plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=1000)
    # plt.show()
    # # 做圆形裁剪. 按照比例radius_prop保留中心区域
    # radius_prop = 0.3
    # m_radius = radius_prop * np.min([np.max(m1) - np.min(m1), np.max(m2) - np.min(m2)])
    # M1, M2 = np.meshgrid(m1, m2, indexing='xy')
    # mask_circle = M1 ** 2 + M2 ** 2 <= m_radius ** 2
    # U0 = U0 * mask_circle
    # # 做0填充
    # pad_m1 = 2
    # pad_m2 = 2
    # pad_m1 = int(pad_m1 * len(m1))
    # pad_m2 = int(pad_m2 * len(m2))
    # U0 = np.pad(U0, ((pad_m1, pad_m1), (pad_m2, pad_m2)), mode='constant', constant_values=0)
    # # 更新 m1, m2
    # m1 = np.linspace(np.min(m1) - pad_m1 * (m1[1] - m1[0]),
    #                  np.max(m1) + pad_m1 * (m1[1] - m1[0]), len(m1) + 2 * pad_m1)
    # m2 = np.linspace(np.min(m2) - pad_m2 * (m2[1] - m2[0]),
    #                  np.max(m2) + pad_m2 * (m2[1] - m2[0]), len(m2) + 2 * pad_m2)
    # # 1) 单个 z 面
    # z = 0.00
    # norm_f = 0.52
    # # m1 的单位为 2*π/P
    # # 频率的单位为 c/P
    # norm_m1 = m1 / norm_f
    # norm_m2 = m2 / norm_f
    # # # 可视化：np.abs(Exy) 或 np.angle(Exy)
    # # fig, ax = plt.subplots(figsize=(1.25, 1.25), dpi=200)
    # # real_U0 = np.real(U0)
    # # # 对于关于0对称但是极小的数据, 取log绝对值后可视化效果更好
    # # log_real_U0 = np.sign(real_U0) * np.log10(np.abs(real_U0) + 1e-5)
    # # from matplotlib.colors import SymLogNorm
    # # im = ax.imshow(real_U0.T, origin='lower', extent=(m1[0], m1[-1], m2[0], m2[-1]),
    # #                cmap='RdBu', norm=SymLogNorm(linthresh=1e-2, vmin=-1, vmax=1))
    # # plt.show()
    # Az, (x, y, Exy) = angular_spectrum_propagate(norm_m1, norm_m2, U0, z, k0=1, time_sign=-1)
    # # 可视化：np.abs(Exy) 或 np.angle(Exy)
    # fig, ax = plt.subplots(figsize=(1.25, 1.25))
    # # im = ax.imshow(np.abs(Exy).T**2, origin='lower', extent=(x[0], x[-1], y[0], y[-1]),
    # #                cmap='magma')
    # im = ax.imshow(np.real(Exy).T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]),
    #                cmap='RdBu')
    # # rgb = map_complex2rbg(Exy)
    # # im = ax.imshow(rgb, origin='lower', extent=(x[0], x[-1], y[0], y[-1]))
    # ax.set_xlim(-100, 100)
    # ax.set_ylim(-100, 100)
    # plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=1000)
    # plt.show()
    #
    # # 2) XZ 纵截面（y=0 切片）
    # # z_list = np.linspace(0, 5e3, 200)
    # # z_list = np.linspace(-1e4, 1e4, 10)
    # z_list = np.linspace(0, 1e4, 100)
    # x, z, Exz = angular_spectrum_xz_slice(m1, m2, U0, z_list, k0=1, y0=0.0, include_evanescent=False, time_sign=-1)
    # # 可视化：np.abs(Exz) 或 np.angle(Exz)
    # fig, ax = plt.subplots(figsize=(4, 1.25))
    # im = ax.imshow(np.abs(Exz).T**2, origin='lower', extent=(z[0], z[-1], x[0], x[-1]),
    #                aspect='auto', cmap='magma')
    # ax.set_ylim(-1.5e3, 1.5e3)
    # plt.savefig('temp.svg', bbox_inches='tight', transparent=True, dpi=1000)
    # plt.show()
    #
    # plotter.new_2d_fig()
    # plotter.plot_polarization_ellipses(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.plot_phi_families_regimes(index=0)
    # plotter.plot_phi_families_split(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_phi(index=0, cmap='twilight')
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.52), colors=('k'))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.52)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1, alpha=1)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['tanchi'], 'k-', linewidth=1, alpha=1)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['qlog'], 'k-', linewidth=1, alpha=1)
    # # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_advanced_color_mapping(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(tick_direction='out')
    # plotter.new_2d_fig()
    # plotter.imshow_qlog(index=0, cmap='magma', vmin=2, vmax=7)
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.510, 0.52, 0.527), colors=('r', 'k', 'b'))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(3.8, 1))
    # plotter.new_2d_fig()
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.510)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'r-', linewidth=1, alpha=0.5)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.52)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.527)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'b-', linewidth=1, alpha=0.5)
    # # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(2, 2))
    # plotter.new_3d_fig()
    # mapping = {
    #     'cmap': 'magma',
    #     'z2': {'vmin': 2, 'vmax': 7},  # 可选；未给则自动取数据范围
    #     # 'z3': {'vmin': c, 'vmax': d},  # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
    # }
    # plotter.plot_3D_surface(index=0, shade=False, mapping=mapping)
    # plotter.ax.set_box_aspect([1, 1, 0.5])
    # plotter.config.annotations = {'show_tick_labels': False}
    # plotter.add_annotations()
    # plotter.save_and_show()


    # # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P400_T200_L211_R0.3.pkl"
    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P400_T200_L211.8_R0.3.pkl"
    # # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P400_T200_L212_R0.3.pkl"
    # # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P400_T200_L214_R0.3.pkl"
    # config = PlotConfig(
    #     plot_params={},
    #     annotations={'show_tick_labels': True, 'xlim': (-0.10, 0.10), 'ylim': (-0.10, 0.10)},
    # )
    # config.update(figsize=(1.25, 1.25))
    # config.update(tick_direction='in')
    # plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path)
    # plotter.load_data()
    # plotter.prepare_data()
    #
    # plotter.new_2d_fig()
    # plotter.plot_polarization_ellipses(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # # plotter.new_2d_fig()
    # # plotter.plot_polarization_ellipses(index=0, step=(1, 1), scale=0.001)
    # # plotter.add_annotations()
    # # plotter.save_and_show()
    #
    # # plotter.new_3d_fig()
    # # plotter.plot_on_poincare_sphere(index=0)
    # # plotter.add_annotations()
    # # plotter.save_and_show()
    #
    # # plotter.new_2d_fig()
    # # plotter.imshow_skyrmion_density(index=0)
    # # plotter.add_annotations()
    # # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.prepare_chi_phi_data()
    # plotter.plot_phi_families_regimes(index=0)
    # plotter.plot_phi_families_split(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_phi(index=0, cmap='twilight')
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_advanced_color_mapping(index=0)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(tick_direction='out')
    # plotter.new_2d_fig()
    # plotter.imshow_qlog(index=0, cmap='magma', vmin=2, vmax=8)
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.5088, 0.5095, 0.5108), colors=('r', 'k', 'b'))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(3.8, 1))
    # plotter.config.update(tick_direction='in')
    # plotter.new_2d_fig()
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.5088)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'r-', linewidth=1,
    #                 alpha=0.5)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.5095)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.5108)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'b-', linewidth=1,
    #                 alpha=0.5)
    # # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(2, 2))
    # plotter.new_3d_fig()
    # mapping = {
    #     'cmap': 'magma',
    #     'z2': {'vmin': 2, 'vmax': 8},  # 可选；未给则自动取数据范围
    #     # 'z3': {'vmin': c, 'vmax': d},  # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
    # }
    # plotter.plot_3D_surface(index=0, shade=False, mapping=mapping)
    # plotter.ax.set_box_aspect([1, 1, 0.5])
    # plotter.config.annotations = {'show_tick_labels': False}
    # plotter.add_annotations()
    # plotter.save_and_show()


    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L580_R0.3-btnDP.pkl"
    # data_path_2 = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L580_R0.3-topDP.pkl"
    #
    # # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L585_R0.3-btnDP.pkl"
    # # data_path_2 = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L585_R0.3-topDP.pkl"
    #
    # # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L590_R0.3-btnDP.pkl"
    # # data_path_2 = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L590_R0.3-topDP.pkl"
    # config = PlotConfig(
    #     plot_params={},
    #     annotations={'show_tick_labels': True, 'xlim': (-0.10, 0.10), 'ylim': (-0.10, 0.10)},
    # )
    # config.update(figsize=(1.25, 1.25), tick_direction='in')
    # BAND_INDEX = 0
    # plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path
    # plotter.load_data()
    # plotter.prepare_data()
    #
    # plotter.new_2d_fig()
    # plotter.plot_polarization_ellipses(index=BAND_INDEX)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.prepare_chi_phi_data()
    # plotter.plot_phi_families_regimes(index=BAND_INDEX)
    # plotter.plot_phi_families_split(index=BAND_INDEX)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_phi(index=BAND_INDEX, cmap='twilight')
    # plotter.plot_isofreq_contours2D(index=BAND_INDEX, levels=(0.52), colors=('k'))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # samples_list = plotter.sample_along_isofreq(index=BAND_INDEX, level=0.655)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1, alpha=1)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['tanchi'], 'k-', linewidth=1, alpha=1)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['qlog'], 'k-', linewidth=1, alpha=1)
    # # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_advanced_color_mapping(index=BAND_INDEX)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(tick_direction='out')
    # plotter.new_2d_fig()
    # plotter.imshow_qlog(index=BAND_INDEX, cmap='magma', vmin=2, vmax=8)
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.652, 0.655, 0.659), colors=('r', 'k', 'b'))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(3.8, 1))
    # plotter.config.update(tick_direction='in')
    # plotter.new_2d_fig()
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.652)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'r-', linewidth=1,
    #                 alpha=0.5)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.655)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.659)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'b-', linewidth=1,
    #                 alpha=0.5)
    # # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(2, 2))
    # plotter.new_3d_fig()
    # mapping = {
    #     'cmap': 'magma',
    #     'z2': {'vmin': 2, 'vmax': 8},  # 可选；未给则自动取数据范围
    #     # 'z3': {'vmin': c, 'vmax': d},  # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
    # }
    # plotter.plot_3D_surface(index=0, shade=False, mapping=mapping, alpha=1)
    # plotter.re_initialized(data_path=data_path_2)
    # plotter.load_data()
    # plotter.prepare_data()
    # plotter.plot_3D_surface(index=1, shade=False, mapping=mapping, alpha=0.1)
    # plotter.ax.set_box_aspect([1, 1, 0.5])
    # plotter.config.annotations = {'show_tick_labels': False}
    # plotter.add_annotations()
    # plotter.save_and_show()


    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L580_R0.3-btnDP.pkl"
    # data_path_2 = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L580_R0.3-topDP.pkl"

    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L585_R0.3-btnDP.pkl"
    # data_path_2 = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L585_R0.3-topDP.pkl"

    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L590_R0.3-btnDP.pkl"
    # data_path_2 = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L590_R0.3-topDP.pkl"
    # config = PlotConfig(
    #     plot_params={},
    #     annotations={'show_tick_labels': True, 'xlim': (-0.10, 0.10), 'ylim': (-0.10, 0.10)},
    # )
    # config.update(figsize=(1.25, 1.25), tick_direction='in')
    # BAND_INDEX = 0
    # # plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path)
    # plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path_2)
    # plotter.load_data()
    # plotter.prepare_data()
    #
    # plotter.new_2d_fig()
    # plotter.plot_polarization_ellipses(index=BAND_INDEX)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.prepare_chi_phi_data()
    # plotter.plot_phi_families_regimes(index=BAND_INDEX)
    # plotter.plot_phi_families_split(index=BAND_INDEX)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_phi(index=BAND_INDEX, cmap='twilight')
    # plotter.plot_isofreq_contours2D(index=BAND_INDEX, levels=(0.52), colors=('k'))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # samples_list = plotter.sample_along_isofreq(index=BAND_INDEX, level=0.665)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1, alpha=1)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['tanchi'], 'k-', linewidth=1, alpha=1)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['qlog'], 'k-', linewidth=1, alpha=1)
    # # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.new_2d_fig()
    # plotter.imshow_advanced_color_mapping(index=BAND_INDEX)
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(tick_direction='out')
    # plotter.new_2d_fig()
    # plotter.imshow_qlog(index=BAND_INDEX, cmap='magma', vmin=2, vmax=8)
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.662, 0.665, 0.668), colors=('r', 'k', 'b'))
    # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(3.8, 1))
    # plotter.config.update(tick_direction='in')
    # plotter.new_2d_fig()
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.662)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'r-', linewidth=1,
    #                 alpha=0.5)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.665)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1)
    # samples_list = plotter.sample_along_isofreq(index=0, level=0.668)
    # plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'b-', linewidth=1,
    #                 alpha=0.5)
    # # plotter.add_annotations()
    # plotter.save_and_show()
    #
    # plotter.config.update(figsize=(2, 2))
    # plotter.new_3d_fig()
    # mapping = {
    #     'cmap': 'magma',
    #     'z2': {'vmin': 2, 'vmax': 8},  # 可选；未给则自动取数据范围
    #     # 'z3': {'vmin': c, 'vmax': d},  # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
    # }
    # plotter.plot_3D_surface(index=0, shade=False, mapping=mapping, alpha=1)
    # plotter.re_initialized(data_path=data_path)
    # plotter.load_data()
    # plotter.prepare_data()
    # plotter.plot_3D_surface(index=0, shade=False, mapping=mapping, alpha=0.1)
    # plotter.ax.set_box_aspect([1, 1, 0.5])
    # plotter.config.annotations = {'show_tick_labels': False}
    # plotter.add_annotations()
    # plotter.save_and_show()
