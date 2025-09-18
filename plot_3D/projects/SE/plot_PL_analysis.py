# 使用示例：每个脚本一个自定义类（继承分类子类，重写prepare_data/plot）
# 脚本1示例：MyScript1Plotter（继承LinePlotter/PolarPlotter混合，用组合或多实例）
import numpy as np

from plot_3D.core.plot_workflow import PlotConfig, LinePlotter, PolarPlotter, HeatmapPlotter
from plot_3D.core.utils import load_lumerical_jsondata, structure_lumerical_jsondata

c_const = 299292458

# 修复后的脚本1类：MyScript1Plotter（针对plot_single_purcell.py）
class MyScript1Plotter(LinePlotter, PolarPlotter, HeatmapPlotter):  # 多继承：支持线+极坐标
    """脚本1自定义：重写prepare_data（加除零保护），plot分离4场景"""

    def prepare_data(self) -> None:
        """重写：手动提取Lumerical数据，加除零保护"""
        ref_path = self.data_path.replace('PL_Analysis.json', 'PL_Analysis_Ref.json')
        ref_data = load_lumerical_jsondata(ref_path)
        target_data = self.data
        self.freq = structure_lumerical_jsondata(ref_data, 'freq').ravel()/(c_const/1e-6)
        self.purcell_freq = np.linspace(self.freq[0], self.freq[-1], 1000)
        self.NA_list = structure_lumerical_jsondata(ref_data, 'NA_list').ravel().tolist()
        self.target_k_list = structure_lumerical_jsondata(target_data, 'k_list').ravel().tolist()
        print('k_list:', self.target_k_list)
        print('NA_list:', self.NA_list)
        self.target_farfield_power = structure_lumerical_jsondata(target_data, 'farfield_power_from_trans')
        self.target_purcell_factors = structure_lumerical_jsondata(target_data, 'purcell_factors')
        self.theta = structure_lumerical_jsondata(target_data, 'theta').ravel()
        self.target_E2_hyperdata = structure_lumerical_jsondata(target_data, 'E2_hyperdata')
        self.ref_integrate = structure_lumerical_jsondata(ref_data, 'ref_integrate_farfield_power')
        self.target_integrate = structure_lumerical_jsondata(target_data, 'integrate_farfield_power')

        self.target_integrate_PL_factor = self.target_integrate / (self.ref_integrate) / 3.5 / 3.5
        # 额外检查inf/nan
        if np.any(np.isnan(self.target_integrate_PL_factor)) or np.any(np.isinf(self.target_integrate_PL_factor)):
            print("Warning: target_integrate_PL_factor含NaN/Inf，可能影响绘图 ⚠️")
        print(f"脚本1数据准备：freq_shape={self.freq.shape}, k_list_len={len(self.target_k_list)} ✅")

    def plot(self) -> None:
        """抽象实现：用户手动调用具体场景"""
        pass  # 留空，强制用子方法

    def plot_single_line_purcell(self, twin=False) -> None:
        """场景A：单线图"""
        self.plot_line(self.purcell_freq, z1=self.target_purcell_factors[:, 0, 0], twin=twin)

    def plot_single_line_PL_factor(self, twin=False) -> None:
        """场景A：单线图"""
        # self.plot_line(self.freq.ravel(), z1=-2*self.target_farfield_power[:, 0, 0])
        self.plot_line(self.freq.ravel(), z1=self.target_integrate_PL_factor[::, 0, -1], twin=twin)

    def plot_multi_k_lines(self) -> None:
        """场景B：多线叠加"""
        x_vals_list, y_vals_list = [], []
        for i in range(len(self.target_k_list)):
            x_vals = np.linspace(self.freq[0], self.freq[-1], 1000)
            y_vals = self.target_purcell_factors[:, i, 0].ravel()
            x_vals_list.append(x_vals)
            y_vals_list.append(y_vals)
        for i, (x, y) in enumerate(zip(x_vals_list, y_vals_list)):
            self.plot_line(x, z1=y, index=i)

    def plot_multi_NA_lines(self) -> None:
        """场景B'：多线叠加"""
        x_vals_list, y_vals_list = [], []
        for j in range(len(self.NA_list)):
            x_vals = self.freq
            y_vals = self.target_integrate_PL_factor[:, 0, j].ravel()
            x_vals_list.append(x_vals)
            y_vals_list.append(y_vals)
        for i, (x, y) in enumerate(zip(x_vals_list, y_vals_list)):
            self.plot_line(x, z1=y, index=i)

    def plot_polar_lines(self) -> None:
        """场景C：极坐标多线（修复：直接用PolarPlotter骨架）"""
        # self.new_fig('polar')  # 直接polar投影
        x_vals_list, y_vals_list = [], []
        for i in range(len(self.freq.ravel()[160:-290:1])):
            x_vals = self.theta
            y_vals = self.target_E2_hyperdata[:, i, 0, 0].ravel()
            x_vals_list.append(x_vals)
            y_vals_list.append(y_vals)
        for x, y in zip(x_vals_list, y_vals_list):
            self.plot_polar(x, y, r_max=np.max(y_vals_list))  # 直接调用骨架
        print("极坐标图完成 📊")

    def plot_farfield_heatmaps(self) -> None:
        """场景：..."""
        z_vals_list = []
        self.freq = self.freq[160:-290:1]
        self.target_E2_hyperdata = self.target_E2_hyperdata[:, 160:-290:1, :, :]
        for i in range(len(self.freq.ravel()[::1])):
            z_vals = self.target_E2_hyperdata[50:-50, i, 0, 0].ravel()
            z_vals_list.append(z_vals)
        self.x_vals = self.theta[50:-50:1]
        self.y_vals = self.freq
        self.plot_heatmap(np.array(z_vals_list).T)  # 直接调用骨架
        print("极坐标图完成 📊")

    def plot_k_max_line(self, color1='red', color2='blue'):
        """场景D：参数-最大值线"""
        y_max_list = []
        y1_max_list = []
        for i in range(len(self.target_k_list)):
            y_vals = -2 * self.target_farfield_power[:, i, 0].ravel()
            y_max_list.append(np.max(y_vals))
            y1_vals = self.target_purcell_factors[:, i, 0].ravel()
            y1_max_list.append(np.max(y1_vals))
        self.plot_line(np.array(self.target_k_list), z1=np.array(y_max_list), default_color=color1)
        self.plot_line(np.array(self.target_k_list), z1=np.array(y1_max_list), default_color=color2)
        return self.target_k_list, (y_max_list, y1_max_list)

    def plot_NA_max_line(self) -> None:
        """场景E：参数-最大值线"""
        y_max_list = []
        for j in range(len(self.NA_list)):
            y_vals = self.target_integrate_PL_factor[:, 0, j].ravel()
            y_max_list.append(np.max(y_vals))
        self.plot_line(np.array(self.NA_list), z1=np.array(y_max_list))



if __name__ == '__main__':
    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma',
    #     },
    #     annotations={
    #         'ylim': (0, 1e-4)
    #     }
    # )
    # config.figsize = (2.5, 2.5)
    # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_NAs\PL_Analysis.json')
    # # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()  # 重写核心
    # plotter.new_fig('polar')
    # plotter.plot_polar_lines()  # 手动选场景
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存


    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma', 'default_color': 'k'
    #     },
    #     annotations={
    #         'ylim': (0, 2e2)
    #     }
    # )
    # config.figsize = (4, 4)
    # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()  # 重写核心
    # plotter.new_fig()
    # plotter.plot_NA_max_line()  # 手动选场景
    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma', 'default_color': 'gray'
    #     },
    #     annotations={
    #         'ylim': (0, 2e2)
    #     }
    # )
    # plotter.re_initialized(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()  # 重写核心
    # plotter.plot_NA_max_line()  # 手动选场景
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存


    config = PlotConfig(
        plot_params={
            'add_colorbar': True, 'cmap': 'magma', 'default_color': 'black'
        },
        annotations={
            'ylim': (0, 100)
        }
    )
    config.figsize = (4, 4)
    plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_ks\PL_Analysis.json')
    plotter.load_data()
    plotter.prepare_data()  # 重写核心
    plotter.new_fig()
    target_k_list_1, (y1_list_1, y2_list_1) = plotter.plot_k_max_line(color1='red', color2='gray')  # 手动选场景
    config = PlotConfig(
        plot_params={
            'add_colorbar': True, 'cmap': 'magma', 'default_color': 'gray'
        },
        annotations={
            'ylim': (0, 100)
        }
    )
    plotter.re_initialized(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_ks\PL_Analysis.json')
    plotter.load_data()
    plotter.prepare_data()  # 重写核心
    target_k_list_2, (y1_list_2, y2_list_2) = plotter.plot_k_max_line(color1='blue', color2='gray')  # 手动选场景
    plotter.add_annotations()  # 注解
    plotter.save_and_show()  # 保存

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma',
    #     },
    #     annotations={
    #         # 'ylim': (0, 1e2)
    #     }
    # )
    # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\sweep_NA\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()  # 重写核心
    # plotter.new_fig()
    # plotter.plot_k_max_line()  # 手动选场景
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma',
    #     },
    #     annotations={
    #         'ylim': (0, 1e2)
    #     }
    # )
    # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\sweep_NA\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()  # 重写核心
    # plotter.new_fig()
    # plotter.plot_multi_NA_lines()  # 手动选场景
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存


    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma', 'default_color': 'black'
    #     },
    #     annotations={
    #         'ylim': (0, 40)
    #     }
    # )
    # config.figsize = (2, 4)
    # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()  # 重写核心
    # plotter.new_fig()
    # plotter.plot_single_line_PL_factor()  # 手动选场景
    # plotter.add_annotations()  # 注解
    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma', 'default_color': 'gray'
    #     },
    #     annotations={
    #         'ylim': (0, 40)
    #     }
    # )
    # plotter.re_initialized(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()  # 重写核心
    # plotter.plot_single_line_PL_factor(twin=True)  # 手动选场景
    # plotter.add_twin_annotations()  # 注解
    # plotter.save_and_show()  # 保存


    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma', 'default_color': 'black'
    #     },
    #     annotations={
    #         'ylim': (0, 40)
    #     }
    # )
    # config.figsize = (2, 4)
    # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()  # 重写核心
    # plotter.new_fig()
    # plotter.plot_single_line_purcell()  # 手动选场景
    # plotter.add_annotations()  # 注解
    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma', 'default_color': 'gray'
    #     },
    #     annotations={
    #         'ylim': (0, 40)
    #     }
    # )
    # plotter.re_initialized(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()  # 重写核心
    # plotter.plot_single_line_purcell(twin=True)  # 手动选场景
    # plotter.add_twin_annotations()  # 注解
    # plotter.save_and_show()  # 保存


    # plotter.re_initialized(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\sweep_ks\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()  # 重写核心
    #
    # # plotter.plot_single_line(2)  # 手动选场景
