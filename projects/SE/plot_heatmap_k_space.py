# 脚本2示例：继承HeatmapPlotter
import pickle

import numpy as np

from core.plot_workflow import HeatmapPlotter, PlotConfig


class MyScript2Plotter(HeatmapPlotter):
    # def _load_pickle(self) -> None:
    #     """Pickle加载骨架"""
    #     with open(self.data_path, 'rb') as f:
    #         self.raw_dataset = pickle.load(f)
    #     self.x_vals = self.raw_dataset.get('x_vals', np.array([]))
    #     self.y_vals = self.raw_dataset.get('y_vals', np.array([]))
    #     self.subs = self.raw_dataset.get('subs', [])

    # def prepare_data(self) -> None:  # 手动重写：简单Z1
    #     self.Z1 = self.subs[0][:, :]  # 无NaN
    #     print("脚本2准备：Z1直接取subs[0]")
    #
    # def plot(self) -> None:  # 重写：调用骨架
    #     self.plot_heatmap(self.Z1, self.x_vals, self.y_vals,)

    def prepare_data(self) -> None:  # 手动重写：简单Z1
        self.Z1 = self.data_list[0][:, :]  # 无NaN
        print(np.max(self.Z1))
        print("脚本2准备：Z1直接取subs[0]")

    def plot(self) -> None:  # 重写：调用骨架
        self.plot_heatmap(self.Z1, self.coordinates['m1'], self.coordinates['频率 (Hz)'],)


def main(data_path: str, config: PlotConfig):
    plotter = MyScript2Plotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()
    plotter.new_2d_fig()
    plotter.plot()
    plotter.add_annotations()
    plotter.save_and_show()

if __name__ == '__main__':
    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma',
    #         'title': False,
    #         'title_colorbar': 'P',
    #         'global_color_vmax': 2387654,
    #     },
    #     annotations={
    #         'xlabel': r'k (2$\pi$/P)', 'ylabel': 'f (c/P)',
    #         'enable_tight_layout': True,
    #         'show_axis_labels': True, 'show_tick_labels': True,
    #         'ylim': (0.55, 0.65),
    #     }
    # )
    # config.figsize = (2.75, 3)
    # main(r"D:\DELL\Documents\myPlots\projects\SE\rsl\manual_datas\PL\1fold_PL_BIC-rad-k_space.pkl", config)

    config = PlotConfig(
        plot_params={
            'add_colorbar': True, 'cmap': 'magma',
            'title': False,
            'title_colorbar': 'P',
            'global_color_vmax': 2387654,
        },
        annotations={
            'xlabel': r'k (2$\pi$/P)', 'ylabel': 'f (c/P)',
            'enable_tight_layout': True,
            'show_axis_labels': True, 'show_tick_labels': True,
            'ylim': (0.45, 0.55),
        }
    )
    config.figsize = (2.75, 3)
    main(r"D:\DELL\Documents\myPlots\projects\SE\rsl\manual_datas\PL\1fold_weak_PL_BIC-rad-k_space.pkl", config)

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma',
    #         'title': False,
    #     },
    #     annotations={'xlabel': r'f (c/P)', 'ylabel': 'P'}
    # )
    # main(r"D:\DELL\Documents\myPlots\projects\SE\rsl\manual_datas\PL\3fold_PL_QGM-rad-k_space.pkl", config)

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma',
    #         'title': False,
    #     },
    #     annotations={'xlabel': r'f (c/P)', 'ylabel': 'P', 'ylim': (0.55, 0.65)}
    # )
    # main(r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\PL\3fold_PL_QGM.pkl", config)
