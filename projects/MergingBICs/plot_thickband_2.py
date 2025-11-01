from core.plot_cls import BandPlotterOneDim
from core.plot_workflow import PlotConfig, LinePlotter


class MyScriptPlotter(BandPlotterOneDim):
    pass


def main(data_path):
    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={
            'xlabel': 'k (2$\pi$/P)', 'ylabel': 'f (c/P)', 'show_axis_labels': True, 'show_tick_labels': True,
            'ylim': (0.35, 0.52),
        },
    )
    config.figsize = (1.5, 3)
    config.update(tick_direction='in')
    plotter = MyScriptPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()
    plotter.new_2d_fig()
    plotter.plot_thick_bg()
    plotter.plot_colored_line()
    plotter.adjust_view_2dim()
    plotter.add_annotations()
    plotter.save_and_show()

if __name__ == '__main__':
    pass
