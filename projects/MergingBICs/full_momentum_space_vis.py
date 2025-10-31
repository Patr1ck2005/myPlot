from core.plot_cls import MomentumSpaceEigenPolarizationPlotter
from core.plot_workflow import PlotConfig


if __name__ == '__main__':
    data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\rsl\manual\3Deigen-Hole_P400_T200_L211.pkl"
    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\rsl\manual\3Deigen-Hole_L212nm.pkl"
    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\rsl\manual\3Deigen-Hole_L214nm.pkl"
    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\rsl\manual\3Deigen-Hole_P450_T200_L250.pkl"
    # data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\rsl\manual\3Deigen-Hole_P450_T200_L250.pkl"
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.figsize = (1.5, 3)
    config.tick_direction = 'in'
    plotter = MomentumSpaceEigenPolarizationPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()

    plotter.new_2d_fig()
    plotter.plot_polarization_ellipses(index=0)
    plotter.plot_isofreq_contours2D(index=0, levels=(0.509, 0.510, 0.511))
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.prepare_chi_phi_data()
    plotter.plot_phi_families_regimes(index=0)
    plotter.plot_phi_families_split(index=0)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_qlog(index=0)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig()
    plotter.imshow_phi(index=0)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_3d_fig()
    plotter.plot_3D_surface(index=0)
    plotter.add_annotations()
    plotter.save_and_show()


