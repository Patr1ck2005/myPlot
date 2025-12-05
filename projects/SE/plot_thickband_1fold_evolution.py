# 脚本3示例：继承LinePlotter（填充多线变体）
import numpy as np

from core.plot_3D_params_space_plt import plot_1d_lines
from core.plot_workflow import PlotConfig, LinePlotter


# class MyScriptPlotter(LinePlotter):
#     def prepare_data(self) -> None:  # 手动重写：NaN过滤
#         self.x_vals_list = []
#         self.y_vals_list = []
#         for sub in self.subs:
#             mask = np.isnan(sub)
#             if np.any(mask):
#                 print("Warning: NaN移除⚠️")
#                 y_vals = sub[~mask]
#                 temp_x = self.x_vals[~mask]
#             else:
#                 y_vals = sub
#                 temp_x = self.x_vals
#             self.x_vals_list.append(temp_x)
#             self.y_vals_list.append(y_vals)
#
#     def plot(self) -> None:  # 重写：整体+循环填充
#         params = {
#             'enable_fill': True,
#             'gradient_fill': True,
#             'cmap': 'magma',
#             'add_colorbar': False,
#             'global_color_vmin': 0, 'global_color_vmax': 5e-3,
#             'default_color': 'gray', 'alpha_fill': 1,
#             'edge_color': 'none',
#             'gradient_direction': 'z3',
#         }
#         y_mins, y_maxs = [], []
#         for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
#             self.plot_line(x, z1=y.real, z2=y.imag, z3=y.imag, **params)  # 填充
#             widths = np.abs(y.imag)
#             y_mins.append(np.min(y.real - widths))
#             y_maxs.append(np.max(y.real + widths))
#         self.ax.set_xlim(self.x_vals.min(), self.x_vals.max())
#         self.ax.set_ylim(np.nanmin(y_mins) * 0.98, np.nanmax(y_maxs) * 1.02)
#
# def main(data_path):
#     config = PlotConfig(
#         plot_params={'scale': 1},
#         annotations={'ylabel': 'f (c/P)', 'show_axis_labels': True, 'show_tick_labels': True},
#     )
#     config.figsize = (3, 6)
#     plotter = MyScriptPlotter(config=config, data_path=data_path)
#     plotter.run_full()


from core.plot_cls import BandPlotterOneDim


class MyScriptPlotter(BandPlotterOneDim):
    pass

def main(data_path):
    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={
            'xlabel': 'k (2$\pi$/P)', 'ylabel': 'f (c/P)', 'show_axis_labels': True, 'show_tick_labels': True,
            # 'enable_tight_layout': True,
            # 'ylim': (0.45-0.0175, 0.65+0.028),
            'ylim': (0.40, 0.9),
        },
    )
    config.update(figsize=(2, 3), tick_direction='out')
    # config.update(figsize=(2.20, 2.9), tick_direction='out')
    plotter = MyScriptPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()
    plotter.new_2d_fig()
    plotter.plot_diffraction_cone(alpha=0.5, color='gray')
    plotter.plot_colored_bg(vmax=6e-2, alpha=0.8)
    # plotter.plot_colored_bg(vmax=5e-3, alpha=1)
    # plotter.plot_colored_bg(vmax=4e-2, alpha=1)
    plotter.adjust_view_2dim()
    plotter.add_annotations()
    plotter.save_and_show()

if __name__ == '__main__':
    # main(r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\eigens\2fold-kloss0-k_space.pkl")
    pass