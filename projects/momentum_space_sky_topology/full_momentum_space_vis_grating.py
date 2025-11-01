from core.plot_cls import MomentumSpaceEigenPolarizationPlotter
from core.plot_workflow import PlotConfig


if __name__ == '__main__':
    # data_path = r"D:\DELL\Documents\myPlots\projects\momentum_space_sky_topology\manual\3Deigen-grating-H400nm.pkl"
    # data_path = r"D:\DELL\Documents\myPlots\projects\momentum_space_sky_topology\manual\3Deigen-grating-H450nm.pkl"
    # data_path = r"D:\DELL\Documents\myPlots\projects\momentum_space_sky_topology\manual\3Deigen-grating-H500nm.pkl"
    # data_path = r"D:\DELL\Documents\myPlots\projects\momentum_space_sky_topology\manual\3Deigen-grating-H470nm.pkl"
    # data_path = r"D:\DELL\Documents\myPlots\projects\momentum_space_sky_topology\manual\3Deigen-grating-H500nm.pkl"
    data_path = r"D:\DELL\Documents\myPlots\projects\momentum_space_sky_topology\manual\3Deigen-detailed-grating-H500nm.pkl"
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.figsize = (1.5, 3)
    config.tick_direction = 'in'
    plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()

    band_index = 0

    plotter.new_2d_fig()
    plotter.plot_polarization_ellipses(index=band_index, step=(1, 1))
    plotter.plot_isofreq_contours2D(index=band_index, levels=(0.509, 0.510, 0.511))
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_advanced_color_mapping(index=band_index)
    # plotter.plot_isofreq_contours2D(index=0, levels=(0.509, 0.510, 0.511))
    plotter.save_and_show()

    plotter.new_3d_fig()
    plotter.imshow_s3(index=band_index)
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.prepare_chi_phi_data()
    plotter.plot_phi_families_regimes(index=band_index)
    plotter.plot_phi_families_split(index=band_index)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_3d_fig()
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

    plotter.new_3d_fig()
    rgba = plotter.get_advanced_color_mapping(index=band_index)
    # plotter.plot_3D_surface(index=0, shade=False)
    # plotter.plot_3D_surface(index=1, shade=False)
    plotter.plot_3D_surface(index=band_index, rbga=rgba, shade=False)
    # plotter.ax.set_zlim(0.55, 0.70)
    plotter.add_annotations()
    plotter.save_and_show()


