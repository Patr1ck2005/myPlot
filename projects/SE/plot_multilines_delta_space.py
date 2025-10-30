# 脚本4示例：继承HeatmapPlotter（2D多线）
from core.plot_workflow import HeatmapPlotter, PlotConfig


class MyScript4Plotter(HeatmapPlotter):
    def prepare_data(self) -> None:  # 手动重写：采样+ Z1/Z2
        # self.x_vals = self.raw_data['x_vals']
        self.y_vals = self.y_vals[::4]
        self.Z1 = self.subs[0][:, ::4]
        self.Z2 = self.subs[1][:, ::4]

    def plot(self) -> None:  # 重写：两次叠加

        self.config.plot_params = {
            'add_colorbar': False, 'cmap': 'magma',
            'default_color': 'gray', 'alpha': 0.5,
            'title': False,
        }
        self.plot_multiline_2d(self.Z2, x_vals=self.x_vals, y_vals=self.y_vals)

        self.config.plot_params = {
            'add_colorbar': False, 'cmap': 'magma',
            'title': False,
        }
        self.plot_multiline_2d(self.Z1, x_vals=self.x_vals, y_vals=self.y_vals)

if __name__ == '__main__':
    config = PlotConfig(
        plot_params={
            'add_colorbar': False, 'cmap': 'magma',
            'default_color': 'gray', 'alpha': 0.5,
            'title': False,
        },
        annotations={
            'xlim': (0.430, 0.440), 'ylim': (0, 1.15e11)
        }
    )
    plotter = MyScript4Plotter(config=config, data_path=r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\PL\3fold_PL_QGM-tot_rad-delta_space.pkl")
    plotter.run_full()
