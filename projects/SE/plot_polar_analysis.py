# 使用示例：每个脚本一个自定义类（继承分类子类，重写prepare_data/plot）
# 脚本1示例：MyScript1Plotter（继承LinePlotter/PolarPlotter混合，用组合或多实例）
from typing import Any

from core.plot_workflow import PlotConfig, LinePlotter, PolarPlotter, HeatmapPlotter, ScatterPlotter
from core.utils import load_lumerical_jsondata, structure_lumerical_jsondata
from core.advanced_data_analysis.spectrum_fit_core import *


class MyScriptPlotter(ScatterPlotter, LinePlotter, PolarPlotter, HeatmapPlotter):
    """脚本自定义：prepare_data手动重写；compute_xxx返回数据，便于main后处理/输出"""

    def prepare_data(self) -> None:
        pass

    def plot(self) -> None:
        """抽象：留空，用户在main手动调用绘图"""
        pass


# 脚本示例：MyScriptPlotter
if __name__ == '__main__':
    config = PlotConfig(
        plot_params={
            'add_colorbar': False, 'cmap': 'magma',
        },
        annotations={
            'xlim': (-0.1, 1.1), 'ylim': (0, 2e2), 'add_grid': True,
        }
    )
    config.figsize = (3, 3)
    plotter = MyScriptPlotter(config=config,
                              data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_NAs\PL_Analysis.json')
    plotter.load_data()
    plotter.prepare_data()
    plotter.new_2d_fig()

    plotter.add_annotations()  # 注解
    plotter.save_and_show()  # 保存
