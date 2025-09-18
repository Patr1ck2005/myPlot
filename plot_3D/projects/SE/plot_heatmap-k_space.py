# 脚本2示例：继承HeatmapPlotter
from plot_3D.core.plot_workflow import HeatmapPlotter, PlotConfig


class MyScript2Plotter(HeatmapPlotter):
    def prepare_data(self) -> None:  # 手动重写：简单Z1
        self.Z1 = self.subs[0][:, :]  # 无NaN
        print("脚本2准备：Z1直接取subs[0]")

    def plot(self) -> None:  # 重写：调用骨架
        self.plot_heatmap(self.Z1)


if __name__ == '__main__':
    config = PlotConfig(
        plot_params={
            'add_colorbar': True, 'cmap': 'magma',
            'title': False,
        },
        annotations={'xlabel': r'f (c/P)', 'ylabel': 'P', 'ylim': (0.55, 0.65)}
    )
    plotter = MyScript2Plotter(config=config,
                               data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE/rsl/k_space\20250916_173704\plot_data__x-m1_y-频率Hz.pkl')
    plotter.run_full()  # 一键，或手动链
