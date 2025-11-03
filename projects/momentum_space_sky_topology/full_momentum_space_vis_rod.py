from core.plot_cls import MomentumSpaceEigenPolarizationPlotter
from core.plot_workflow import PlotConfig


if __name__ == '__main__':
    # data_path = r"./manual\3Deigen-Rod-tri0.05-H410-3eigens.pkl"
    # data_path = r"./manual\3Deigen-Rod-tri0.1-H430-3eigens.pkl"
    data_path = r"./manual\3Deigen-Rod-tri0.2-H480-3eigens.pkl"
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.25, 1.25), tick_direction='in')
    plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()

    band_index = 1

    plotter.new_2d_fig()
    plotter.plot_polarization_ellipses(index=band_index, step=(5, 5))
    plotter.plot_isofreq_contours2D(index=band_index, levels=(0.509, 0.510, 0.511))
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.plot_skyrmion_quiver(index=band_index, step=(5, 5))
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_advanced_color_mapping(index=band_index)
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.509, 0.510, 0.511))
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.prepare_chi_phi_data()
    plotter.plot_phi_families_regimes(index=band_index)
    plotter.plot_phi_families_split(index=band_index)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_3d_fig(temp_figsize=(3, 3))
    plotter.plot_on_poincare_sphere(index=band_index)
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_skyrmion_density(index=band_index)
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_qlog(index=band_index)
    plotter.show_colorbar()
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_phi(index=band_index)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_3d_fig(temp_figsize=(3, 3))
    rgba = plotter.get_advanced_color_mapping(index=band_index)
    # plotter.plot_3D_surface(index=0, shade=False)
    # plotter.plot_3D_surface(index=1, shade=False)
    plotter.plot_3D_surface(index=band_index, rbga=rgba, shade=False)
    # plotter.ax.set_zlim(0.55, 0.70)
    plotter.add_annotations()
    plotter.save_and_show()


