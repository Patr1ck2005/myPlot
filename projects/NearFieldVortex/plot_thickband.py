# 脚本3示例：继承LinePlotter（填充多线变体）
import numpy as np

from core.plot_workflow import PlotConfig
from core.plot_cls import BandPlotterOneDim


class MyScriptPlotter(BandPlotterOneDim):
    pass


def main(data_path):
    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={'xlabel': 'k (2$\pi$/P)', 'ylabel': 'f (c/P)', 'show_axis_labels': True, 'show_tick_labels': True},
    )
    config.figsize = (3, 4)
    config.tick_direction = 'in'
    plotter = MyScriptPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()
    plotter.new_2d_fig()
    plotter.plot_colored_bg()
    plotter.adjust_view_2dim()
    # plot yspan
    # compute norm freq span of 900nm to 930nm
    # period = 350e-9  # C4
    period = 400e-9  # C6
    c_const = 299792458
    f1 = c_const / 930e-9 / (c_const / period)
    f2 = c_const / 900e-9 / (c_const / period)
    plotter.ax.axhspan(f1, f2, color='gray', alpha=0.3, linewidth=0, zorder=-1)
    plotter.add_annotations()
    plotter.save_and_show()

if __name__ == '__main__':
    pass
