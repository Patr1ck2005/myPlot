# 脚本2示例：继承HeatmapPlotter
from core.plot_workflow import HeatmapPlotter, PlotConfig


class MyScript2Plotter(HeatmapPlotter):
    def prepare_data(self) -> None:  # 手动重写：简单Z1
        self.Z1 = self.subs[0][:, :]  # 无NaN
        print("脚本2准备：Z1直接取subs[0]")

    def plot(self) -> None:  # 重写：调用骨架
        self.plot_heatmap(self.Z1, self.x_vals, self.y_vals,)


def main(data_path: str, config: PlotConfig):
    plotter = MyScript2Plotter(config=config, data_path=data_path)
    plotter.run_full()  # 一键，或手动链式调用

if __name__ == '__main__':
    config = PlotConfig(
        plot_params={
            'add_colorbar': True, 'cmap': 'magma',
            'title': False,
        },
        annotations={'xlabel': r'f (c/P)', 'ylabel': 'P'}
    )
    main(r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\PL\3fold_PL_QGM.pkl", config)

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma',
    #         'title': False,
    #     },
    #     annotations={'xlabel': r'f (c/P)', 'ylabel': 'P', 'ylim': (0.55, 0.65)}
    # )
    # main(r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\PL\3fold_PL_QGM.pkl", config)
