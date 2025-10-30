# 脚本3示例：继承LinePlotter（填充多线变体）
import numpy as np

from core.plot_3D_params_space_plt import plot_1d_lines
from core.plot_workflow import PlotConfig, LinePlotter


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
        color_list = ['red', 'blue']
        for i, (x, y) in enumerate(zip(self.x_vals_list, self.y_vals_list)):
            params['default_color'] = color_list[i]
            # y = 10**y 恢复为指数数据
            y = np.power(10, y)
            # 将数据关于y轴镜像对称
            x_mirror = np.concatenate([-x[::-1], x])
            y_mirror = np.concatenate([y[::-1], y])
            self.plot_line(x_mirror, z1=y_mirror, **params)  # 填充

def main(data_path):
    config = PlotConfig(
        plot_params={
            'scale': 1,
        },
        annotations={
            'ylabel': 'f (c/P)', 'show_axis_labels': False, 'show_tick_labels': False,
            'xlim': (-0.1, 0.1), 'ylim': (1e1, 1e10),
            'y_log_scale': True,
        },
    )
    config.figsize = (6, 1.5)
    plotter = MyScript3Plotter(config=config, data_path=data_path)
    plotter.run_full()

if __name__ == '__main__':
    main(r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\eigens\2fold-kloss0-k_space-Qfactor.pkl")
    # main(r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\eigens\3fold-kloss0-k_space-Qfactor.pkl")
