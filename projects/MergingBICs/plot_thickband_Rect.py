# 脚本3示例：继承LinePlotter（填充多线变体）
import numpy as np
from matplotlib import pyplot as plt

from core.plot_3D_params_space_plt import plot_1d_lines
from core.plot_cls import BandPlotterOneDim
from core.plot_workflow import PlotConfig, LinePlotter


class MyScriptPlotter(BandPlotterOneDim):

    def plot(self) -> None:  # 重写：整体+循环填充
        params_bg = {
            'enable_fill': True,
            'gradient_fill': False,
            'enable_dynamic_color': False,
            'cmap': None,
            'add_colorbar': False,
            'global_color_vmin': 1, 'global_color_vmax': 8,
            'default_color': 'gray', 'alpha_fill': 0.3,
            'edge_color': 'none',
            'gradient_direction': 'z3',
            'linewidth_base': 0,
        }
        params_line = {
            'enable_fill': False,
            'gradient_fill': False,
            'enable_dynamic_color': True,
            # 'cmap': 'Greys',
            # 'cmap': 'viridis',
            # 'cmap': 'Reds',
            # 'cmap': 'magma',
            'cmap': 'hot',
            'add_colorbar': False,
            'global_color_vmin': 1, 'global_color_vmax': 8,
            'default_color': 'gray', 'alpha_fill': 1,
            'linewidth_base': 2,
            'edge_color': 'none',
        }
        y_mins, y_maxs = [], []
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            Qfactor = np.where(y.imag != 0, np.abs(y.real / (2 * y.imag)), 1e10)
            Qfactor_log = np.log10(Qfactor)
            self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_bg)  # 填充
            self.plot_line(x, z1=y.real, z2=y.imag, z3=Qfactor_log, **params_line)  # 填充
            widths = np.abs(y.imag)
            y_mins.append(np.min(y.real - widths))
            y_maxs.append(np.max(y.real + widths))
        # # add colorbar
        # sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=1, vmax=8))
        # sm.set_array([])
        # cbar = self.fig.colorbar(sm, ax=self.ax)
        # cbar.set_label('$log_{10}(Q)$', rotation=270, labelpad=12)
        self.ax.set_xlim(self.x_vals.min(), self.x_vals.max())
        self.ax.set_ylim(np.nanmin(y_mins) * 0.98, np.nanmax(y_maxs) * 1.02)


def main(data_path):
    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={
            'xlabel': 'k (2$\pi$/P)', 'ylabel': 'f (c/P)', 'show_axis_labels': True, 'show_tick_labels': True,
            'ylim': (0.39, 0.48),
        },
    )
    config.figsize = (1.5, 3)
    config.tick_direction = 'in'
    plotter = MyScriptPlotter(config=config, data_path=data_path)
    plotter.run_full()

if __name__ == '__main__':
    pass
