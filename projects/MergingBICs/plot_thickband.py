from core.plot_cls import BandPlotterOneDim
from core.plot_workflow import PlotConfig, LinePlotter


class MyScriptPlotter(BandPlotterOneDim):
    pass


def main(data_path):
    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={
            'xlabel': 'k (2$\pi$/P)', 'ylabel': 'f (c/P)', 'show_axis_labels': True, 'show_tick_labels': True,
            # 'ylim': (0.51, 0.57),
        },
    )
    config.figsize = (1.25, 3.25)
    config.update(tick_direction='in')
    plotter = MyScriptPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()
    plotter.new_2d_fig()
    plotter.plot_thick_bg()
    plotter.plot_colored_line(vmin=2, vmax=7, cmap='magma')
    plotter.adjust_view_2dim()
    plotter.add_annotations()
    plotter.save_and_show()

if __name__ == '__main__':
    data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\2Deigen-Hole_P450_T200_L250_R0.3.pkl"
    main(data_path)
