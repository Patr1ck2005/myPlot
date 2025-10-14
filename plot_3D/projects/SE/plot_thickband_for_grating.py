# 脚本3示例：继承LinePlotter（填充多线变体）
import numpy as np

from plot_3D.core.plot_3D_params_space_plt import plot_1d_lines
from plot_3D.core.plot_workflow import PlotConfig, LinePlotter


class MyScript3Plotter(LinePlotter):
    def prepare_data(self) -> None:  # 手动重写：NaN过滤
        self.x_vals_list = []
        self.y_vals_list = []
        for sub in self.subs:
            mask = np.isnan(sub)
            if np.any(mask):
                print("Warning: NaN移除⚠️")
                y_vals = sub[~mask]
                temp_x = self.x_vals[~mask]
            else:
                y_vals = sub
                temp_x = self.x_vals
            self.x_vals_list.append(temp_x)
            self.y_vals_list.append(y_vals)

    def plot(self) -> None:  # 重写：整体+循环填充
        params = {
            'enable_fill': False,
            'gradient_fill': False,
            'cmap': 'magma',
            'add_colorbar': False,
            'global_color_vmin': 0, 'global_color_vmax': 5e-3,
            'default_color': 'gray', 'alpha_fill': 1,
            'edge_color': 'none'
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            self.plot_line(x, z1=y.real, z2=y.imag, z3=y.imag, **params)  # 填充
            widths = np.abs(y.imag)
            y_mins.append(np.min(y.real - widths))
            y_maxs.append(np.max(y.real + widths))
        self.ax.set_xlim(self.x_vals.min(), self.x_vals.max())
        self.ax.set_ylim(np.nanmin(y_mins) * 0.98, np.nanmax(y_maxs) * 1.02)

def main(data_path):
    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={'ylabel': 'f (c/P)', 'show_axis_labels': True, 'show_tick_labels': True},
    )
    config.figsize = (3, 4)
    plotter = MyScript3Plotter(config=config, data_path=data_path)
    plotter.run_full()

if __name__ == '__main__':
    main(r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\eigens\2fold-kloss0-k_space.pkl")
