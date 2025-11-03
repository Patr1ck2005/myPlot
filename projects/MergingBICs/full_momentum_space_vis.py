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

    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P450_T200_L250_R0.3.pkl"
    # config = PlotConfig(
    #     plot_params={},
    #     annotations={'show_tick_labels': True, 'xlim': (-0.15, 0.15), 'ylim': (-0.15, 0.15)},
    # )
    # config.update(tick_direction='in', figsize=(1.25, 1.25))
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

    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L585_R0.3-btnDP.pkl"
    # data_path_2 = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L585_R0.3-topDP.pkl"

    data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L590_R0.3-btnDP.pkl"
    data_path_2 = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\3Deigen-Hole_P1300_T200_L590_R0.3-topDP.pkl"
    config = PlotConfig(
        plot_params={},
        annotations={'show_tick_labels': True, 'xlim': (-0.10, 0.10), 'ylim': (-0.10, 0.10)},
    )
    config.update(figsize=(1.25, 1.25))
    config.update(tick_direction='in')
    plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()

    plotter.new_2d_fig()
    plotter.plot_polarization_ellipses(index=0)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.prepare_chi_phi_data()
    plotter.plot_phi_families_regimes(index=0)
    plotter.plot_phi_families_split(index=0)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_phi(index=0, cmap='twilight')
    plotter.plot_isofreq_contours2D(index=0, levels=(0.52), colors=('k'))
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    samples_list = plotter.sample_along_isofreq(index=0, level=0.655)
    plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1, alpha=1)
    plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['tanchi'], 'k-', linewidth=1, alpha=1)
    plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['qlog'], 'k-', linewidth=1, alpha=1)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_advanced_color_mapping(index=0)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.config.update(tick_direction='out')
    plotter.new_2d_fig()
    plotter.imshow_qlog(index=0, cmap='magma', vmin=2, vmax=8)
    plotter.plot_isofreq_contours2D(index=0, levels=(0.652, 0.655, 0.659), colors=('r', 'k', 'b'))
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.config.update(figsize=(3.8, 1))
    plotter.config.update(tick_direction='in')
    plotter.new_2d_fig()
    samples_list = plotter.sample_along_isofreq(index=0, level=0.652)
    plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'r-', linewidth=1,
                    alpha=0.5)
    samples_list = plotter.sample_along_isofreq(index=0, level=0.655)
    plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'k-', linewidth=1)
    samples_list = plotter.sample_along_isofreq(index=0, level=0.659)
    plotter.ax.plot(samples_list[0]['s'] / np.max(samples_list[0]['s']), samples_list[0]['phi'], 'b-', linewidth=1,
                    alpha=0.5)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.config.update(figsize=(2, 2))
    plotter.new_3d_fig()
    mapping = {
        'cmap': 'magma',
        'z2': {'vmin': 2, 'vmax': 8},  # 可选；未给则自动取数据范围
        # 'z3': {'vmin': c, 'vmax': d},  # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
    }
    plotter.plot_3D_surface(index=0, shade=False, mapping=mapping)
    plotter.re_initialized(data_path=data_path_2)
    plotter.load_data()
    plotter.prepare_data()
    plotter.plot_3D_surface(index=0, shade=False, mapping=mapping, alpha=0.1)
    plotter.ax.set_box_aspect([1, 1, 0.5])
    plotter.config.annotations = {'show_tick_labels': False}
    plotter.add_annotations()
    plotter.save_and_show()
