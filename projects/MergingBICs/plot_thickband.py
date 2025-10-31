# 脚本3示例：继承LinePlotter（填充多线变体）
import numpy as np
from matplotlib import pyplot as plt

from core.plot_3D_params_space_plt import plot_1d_lines
from core.plot_cls import BandPlotterOneDim
from core.plot_workflow import PlotConfig, LinePlotter


class MyScriptPlotter(BandPlotterOneDim):
    pass


def main(data_path):
    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={
            'xlabel': 'k (2$\pi$/P)', 'ylabel': 'f (c/P)', 'show_axis_labels': True, 'show_tick_labels': True,
            # 'ylim': (0.51, 0.58),
        },
    )
    config.figsize = (1.5, 3)
    config.tick_direction = 'in'
    plotter = MyScriptPlotter(config=config, data_path=data_path)
    plotter.run_full()

if __name__ == '__main__':
    pass
