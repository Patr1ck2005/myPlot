# 使用示例：每个脚本一个自定义类（继承分类子类，重写prepare_data/plot）
# 脚本1示例：MyScript1Plotter（继承LinePlotter/PolarPlotter混合，用组合或多实例）
from typing import Dict, Any

import numpy as np

from plot_3D.core.plot_workflow import PlotConfig, LinePlotter, PolarPlotter, HeatmapPlotter, ScatterPlotter
from plot_3D.core.utils import load_lumerical_jsondata, structure_lumerical_jsondata

c_const = 299292458


# 脚本1类：MyScript1Plotter（多继承，支持所有）
class MyScript1Plotter(ScatterPlotter, LinePlotter, PolarPlotter, HeatmapPlotter):
    """脚本1自定义：prepare_data手动重写；compute_xxx返回数据，便于main后处理/输出"""

    def prepare_data(self) -> None:
        """重写：手动提取Lumerical数据"""
        ref_path = self.data_path.replace('PL_Analysis.json', 'PL_Analysis_Ref.json')
        ref_data = load_lumerical_jsondata(ref_path)
        target_data = self.raw_dataset
        self.freq = structure_lumerical_jsondata(ref_data, 'freq').ravel() / (c_const / 1e-6)
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
        if np.any(np.isnan(self.target_integrate_PL_factor)) or np.any(np.isinf(self.target_integrate_PL_factor)):
            print("Warning: target_integrate_PL_factor含NaN/Inf，可能影响绘图 ⚠️")
        print(f"脚本1数据准备：freq_shape={self.freq.shape}, k_list_len={len(self.target_k_list)} ✅")

    def plot(self) -> None:
        """抽象：留空，用户在main手动调用绘图"""
        pass

    def compute_single_line_purcell(self) -> Dict[str, Any]:
        """计算单线purcell数据，返回dict便于输出/后处理"""
        x = self.purcell_freq
        y = self.target_purcell_factors[:, 0, 0]
        print(f"计算single_line_purcell: x_shape={x.shape}, y_shape={y.shape} 📊")
        return {'x': x, 'y': y}

    def compute_single_line_PL_factor(self) -> Dict[str, Any]:
        """计算单线PL_factor数据"""
        x = self.freq.ravel()
        y = self.target_integrate_PL_factor[:, 0, -1]
        print(f"计算single_line_PL_factor: x_shape={x.shape}, y_shape={y.shape} 📊")
        return {'x': x, 'y': y}

    def compute_multi_k_lines(self, mode=1) -> Dict[str, Any]:
        """计算多k线数据，返回list dict"""
        x_list, y_list = [], []
        for i in range(len(self.target_k_list)):
            x = np.linspace(self.freq[0], self.freq[-1], 1000)
            if mode == 1:
                y = self.target_purcell_factors[:, i, 0].ravel()
            elif mode == 2:
                y = self.target_integrate_PL_factor[:, i, 0].ravel()
            else:
                y = -2 * self.target_farfield_power[:, i, 0].ravel()
            x_list.append(x)
            y_list.append(y)
        print(f"计算multi_k_lines: lists_len={len(x_list)} 📊")
        return {'x': x_list[0], 'y_list': y_list, 'k_list': self.target_k_list}

    def compute_multi_NA_lines(self) -> Dict[str, Any]:
        """计算多NA线数据"""
        x_list, y_list = [], []
        for j in range(len(self.NA_list)):
            x = self.freq
            y = self.target_integrate_PL_factor[:, 0, j].ravel()
            x_list.append(x)
            y_list.append(y)
        print(f"计算multi_NA_lines: lists_len={len(x_list)} 📊")
        return {'x_list': x_list, 'y_list': y_list}

    def compute_polar_lines(self, k_index=0) -> Dict[str, Any]:
        """计算极坐标数据"""
        x_list, y_list = [], []
        for i in range(len(self.freq.ravel()[::1])):
            x = self.theta
            y = self.target_E2_hyperdata[:, i, k_index, 0].ravel()
            x_list.append(x)
            y_list.append(y)
        print(f"计算polar_lines: lists_len={len(x_list)} 📊")
        return {'theta_list': x_list, 'radial_list': y_list}

    def compute_farfield_heatmaps(self) -> Dict[str, Any]:
        """计算远场热图数据"""
        self.freq = self.freq[160:-290:1]
        self.target_E2_hyperdata = self.target_E2_hyperdata[:, 160:-290:1, :, :]
        z_list = []
        for i in range(len(self.freq.ravel()[::1])):
            z = self.target_E2_hyperdata[50:-50, i, 0, 0].ravel()
            z_list.append(z)
        x = self.theta[50:-50:1]
        y = self.freq
        Z = np.array(z_list).T
        print(f"计算farfield_heatmaps: Z_shape={Z.shape} 📊")
        return {'x': x, 'y': y, 'Z': Z}

    def compute_k_max_line(self) -> Dict[str, Any]:
        """计算k最大值线数据，返回dict"""
        y_max = []
        purcell_max = []
        for i in range(len(self.target_k_list)):
            # y_vals = -2 * self.target_farfield_power[:, i, 0].ravel()
            y_vals = self.target_integrate_PL_factor[:, i, 0].ravel()
            y_max.append(np.max(y_vals))
            purcell_vals = self.target_purcell_factors[:, i, 0].ravel()
            purcell_max.append(np.max(purcell_vals))
        k_array = np.array(self.target_k_list)
        print(f"计算k_max_line: k_len={len(k_array)} 📊")
        return {'k_array': k_array, 'y_max': np.array(y_max), 'purcell_max': np.array(purcell_max)}

    def compute_NA_max_line(self) -> Dict[str, Any]:
        """计算NA最大值线数据"""
        y_max = []
        for j in range(len(self.NA_list)):
            y_vals = self.target_integrate_PL_factor[:, 0, j].ravel()
            y_max.append(np.max(y_vals))
        na_list = np.array(self.NA_list)
        print(f"计算NA_max_line: na_len={len(na_list)} 📊")
        return {'na_list': na_list, 'y_max': np.array(y_max)}

class SimpleScriptPlotter(ScatterPlotter, LinePlotter, PolarPlotter, HeatmapPlotter):
    """脚本1自定义：prepare_data手动重写；compute_xxx返回数据，便于main后处理/输出"""

    def prepare_data(self) -> None:
        """重写：手动提取Lumerical数据"""
        target_data = self.raw_dataset
        self.freq = structure_lumerical_jsondata(target_data, 'freq').ravel() / (c_const / 1e-6)
        self.purcell_freq = np.linspace(self.freq[0], self.freq[-1], 1000)
        self.target_k_list = structure_lumerical_jsondata(target_data, 'para_list').ravel().tolist()
        print('para_list:', self.target_k_list)
        self.target_farfield_power = structure_lumerical_jsondata(target_data, 'farfield_power_from_trans')
        self.target_purcell_factors = structure_lumerical_jsondata(target_data, 'purcell_factors')
        self.target_E2_hyperdata = structure_lumerical_jsondata(target_data, 'E2_hyperdata')
        self.target_integrate = structure_lumerical_jsondata(target_data, 'integrate_farfield_power')
        print(f"脚本1数据准备：freq_shape={self.freq.shape}, k_list_len={len(self.target_k_list)} ✅")

    def plot(self) -> None:
        """抽象：留空，用户在main手动调用绘图"""
        pass

    def compute_single_line_purcell(self) -> Dict[str, Any]:
        """计算单线purcell数据，返回dict便于输出/后处理"""
        x = self.purcell_freq
        y = self.target_purcell_factors[:, 0, 0]
        print(f"计算single_line_purcell: x_shape={x.shape}, y_shape={y.shape} 📊")
        return {'x': x, 'y': y}

    def compute_multi_k_lines(self, mode=1) -> Dict[str, Any]:
        """计算多k线数据，返回list dict"""
        x_list, y_list = [], []
        for i in range(len(self.target_k_list)):
            x = np.linspace(self.freq[0], self.freq[-1], 1000)
            if mode == 1:
                y = self.target_purcell_factors[:, i, 0].ravel()
            else:
                y = -2 * self.target_farfield_power[:, i, 0].ravel()
            x_list.append(x)
            y_list.append(y)
        print(f"计算multi_k_lines: lists_len={len(x_list)} 📊")
        return {'x': x_list[0], 'y_list': y_list, 'k_list': self.target_k_list}

    def compute_polar_lines(self, k_index=0) -> Dict[str, Any]:
        """计算极坐标数据"""
        x_list, y_list = [], []
        for i in range(len(self.freq.ravel()[::1])):
            x = self.theta
            y = self.target_E2_hyperdata[:, i, k_index, 0].ravel()
            x_list.append(x)
            y_list.append(y)
        print(f"计算polar_lines: lists_len={len(x_list)} 📊")
        return {'theta_list': x_list, 'radial_list': y_list}

    def compute_farfield_heatmaps(self) -> Dict[str, Any]:
        """计算远场热图数据"""
        self.freq = self.freq[160:-290:1]
        self.target_E2_hyperdata = self.target_E2_hyperdata[:, 160:-290:1, :, :]
        z_list = []
        for i in range(len(self.freq.ravel()[::1])):
            z = self.target_E2_hyperdata[50:-50, i, 0, 0].ravel()
            z_list.append(z)
        x = self.theta[50:-50:1]
        y = self.freq
        Z = np.array(z_list).T
        print(f"计算farfield_heatmaps: Z_shape={Z.shape} 📊")
        return {'x': x, 'y': y, 'Z': Z}

    def compute_k_max_line(self) -> Dict[str, Any]:
        """计算k最大值线数据，返回dict"""
        y_max = []
        purcell_max = []
        for i in range(len(self.target_k_list)):
            y_vals = -2 * self.target_farfield_power[:, i, 0].ravel()
            y_max.append(np.max(y_vals))
            purcell_vals = self.target_purcell_factors[:, i, 0].ravel()
            purcell_max.append(np.max(purcell_vals))
        k_array = np.array(self.target_k_list)
        print(f"计算k_max_line: k_len={len(k_array)} 📊")
        return {'k_array': k_array, 'y_max': np.array(y_max), 'purcell_max': np.array(purcell_max)}


if __name__ == '__main__':
    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma',
    #     },
    #     annotations={
    #         # 'ylim': (0, 0.020)
    #         # 'ylim': (0, 0.005)
    #         'ylim': (0, 1)
    #     }
    # )
    # config.figsize = (2, 2)
    # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_NAs\PL_Analysis.json')
    # # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data() 
    # plot_dataset = plotter.compute_polar_lines(k_index=0)  # 手动选场景
    # theta = np.array(plot_dataset['theta_list'])
    # radial = np.array(plot_dataset['radial_list'])
    # plotter.new_fig('polar')
    # plotter.plot_polar(theta=theta[0]*10, radial=radial[187]/np.max(radial[187]), default_color='k')  # for lowQ-BIC
    # # plotter.plot_polar(theta=theta[0]*10, radial=radial[332]/np.max(radial[332]), default_color='red')  # for highQ-BIC
    # # plotter.re_initialized(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-QGM\sweep_NAs\PL_Analysis.json')
    # # plotter.load_data()
    # # plotter.prepare_data() 
    # # plot_dataset = plotter.compute_polar_lines(k_index=0)  # 手动选场景
    # # theta = np.array(plot_dataset['theta_list'])
    # # radial = np.array(plot_dataset['radial_list'])
    # # plotter.plot_polar(theta=theta[0]*10, radial=radial[343]/np.max(radial[343]), default_color='blue')  # for highQ-QGM
    # # plotter.plot_polar(theta=theta[0], radial=radial[187]/np.max(radial[187]), default_color='red')
    # # plot_dataset_1 = plotter.compute_polar_lines(k_index=1)  # 手动选场景
    # # theta = np.array(plot_dataset_1['theta_list'])
    # # radial = np.array(plot_dataset_1['radial_list'])
    # # plotter.new_fig('polar')
    # # plotter.plot_polar(theta=theta[0], radial=radial[187]/np.max(radial[187]), default_color='blue')
    # # plot_dataset_1 = plotter.compute_polar_lines(k_index=9)  # 手动选场景
    # # theta = np.array(plot_dataset_1['theta_list'])
    # # radial = np.array(plot_dataset_1['radial_list'])
    # # # plotter.new_fig('polar')
    # # plotter.plot_polar(theta=theta[0], radial=radial[187]/np.max(radial[187]), default_color='k')
    # # plotter.new_fig()
    # # plotter.plot_heatmap(radial)
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': False, 'cmap': 'magma', 'default_color': 'black',
    #         'global_color_vmin': 0, 'global_color_vmax': 75
    #     },
    #     annotations={
    #         'xlim': (-1e-3, 1e-2+1e-3), 'ylim': (0, 100), 'add_grid': True
    #     }
    # )
    # config.figsize = (4, 3)
    # plotter = MyScript1Plotter(config=config,
    #                            data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_ks\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data() 
    # plotter.new_fig()
    # plot_dataset_1 = plotter.compute_k_max_line()  # 手动选场景
    # plotter.plot_line(plot_dataset_1['k_array'], plot_dataset_1['y_max'], default_color='k', default_linestyle='-')
    # plotter.plot_scatter(
    #     plot_dataset_1['k_array'], plot_dataset_1['y_max'], default_color='k', marker='o',
    #     enable_dynamic_color=True, zorder=100, alpha=1
    # )
    # plotter.plot_line(
    #     plot_dataset_1['k_array'], plot_dataset_1['purcell_max'], default_color='k', default_linestyle='--',
    # )
    # plotter.plot_scatter(
    #     plot_dataset_1['k_array'], plot_dataset_1['purcell_max'], default_color='k', marker='o',
    #     enable_dynamic_color=True, zorder=100, alpha=1
    # )
    # plotter.re_initialized(config=config,
    #                        data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_ks\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data() 
    # plot_dataset_2 = plotter.compute_k_max_line()  # 手动选场景
    # plotter.plot_line(plot_dataset_2['k_array'], plot_dataset_2['y_max'], default_color='blue', default_linestyle='-')
    # plotter.plot_scatter(
    #     plot_dataset_2['k_array'], plot_dataset_2['y_max'], default_color='k', marker='D',
    #     enable_dynamic_color=True, zorder=100, alpha=1
    # )
    # plotter.plot_line(plot_dataset_2['k_array'], plot_dataset_2['purcell_max'], default_color='blue',
    #                   default_linestyle='--')
    # plotter.plot_scatter(
    #     plot_dataset_2['k_array'], plot_dataset_2['purcell_max'], default_color='k', marker='D',
    #     enable_dynamic_color=True, zorder=100, alpha=1
    # )
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': False, 'cmap': 'magma',
    #     },
    #     annotations={
    #         'xlim': (-0.1, 1.1), 'ylim': (0, 2e2), 'add_grid': True,
    #     }
    # )
    # config.figsize = (4, 3)
    # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data() 
    # plotter.new_fig()
    # plot_dataset_1 = plotter.compute_NA_max_line()  # 手动选场景
    # plotter.plot_line(np.array(plot_dataset_1['na_list']), np.array(plot_dataset_1['y_max']), default_color='k')
    # plotter.plot_scatter(
    #     np.array(plot_dataset_1['na_list']), np.array(plot_dataset_1['y_max']),
    #     # default_color='k', enable_dynamic_color=True, cmap='magma', alpha=1, global_color_vmin=0, global_color_vmax=2e2,
    #     default_color='k', enable_dynamic_color=False, alpha=1, global_color_vmin=0, global_color_vmax=2e2,
    #     marker='o', zorder=999
    # )
    # plotter.re_initialized(config=config,
    #                        data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data() 
    # plot_dataset_2 = plotter.compute_NA_max_line()  # 手动选场景
    # plotter.plot_line(np.array(plot_dataset_2['na_list']), np.array(plot_dataset_2['y_max']), default_color='k')
    # plotter.plot_scatter(
    #     np.array(plot_dataset_2['na_list']), np.array(plot_dataset_2['y_max']),
    #     default_color='orange', enable_dynamic_color=False, alpha=1, global_color_vmin=0, global_color_vmax=2e2,
    #     marker='D', zorder=999
    # )
    # plotter.re_initialized(config=config,
    #                        data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-QGM\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data() 
    # plot_dataset_3 = plotter.compute_NA_max_line()  # 手动选场景
    # plotter.plot_line(np.array(plot_dataset_3['na_list']), np.array(plot_dataset_3['y_max']), default_color='k')
    # plotter.plot_scatter(
    #     np.array(plot_dataset_3['na_list']), np.array(plot_dataset_3['y_max']),
    #     default_color='blue', enable_dynamic_color=False, cmap='magma', alpha=1, global_color_vmin=0, global_color_vmax=2e2,
    #     marker='p', zorder=999
    # )
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': True, 'cmap': 'magma'
    #     },
    #     annotations={
    #         'ylim': (0, 40)
    #     }
    # )
    # config.figsize = (2, 4)
    # plotter = MyScript1Plotter(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()
    # plotter.new_fig()
    # # plot_dataset_1 = plotter.compute_single_line_purcell()  # 手动选场景
    # plot_dataset_1 = plotter.compute_single_line_PL_factor()  # 手动选场景
    # plotter.plot_line(x=plot_dataset_1['x'], z1=plot_dataset_1['y'], default_color='k', default_linestyle='-')
    # plotter.add_annotations()  # 注解
    # plotter.re_initialized(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()
    # # plot_dataset_2 = plotter.compute_single_line_purcell()  # 手动选场景
    # plot_dataset_2 = plotter.compute_single_line_PL_factor()  # 手动选场景
    # plotter.plot_line(
    #     x=plot_dataset_2['x'], z1=plot_dataset_2['y'],
    #     twin=True,
    #     default_color='red', default_linestyle='-'
    # )
    # plotter.re_initialized(config=config, data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-QGM\sweep_NAs\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()
    # # plot_dataset_3 = plotter.compute_single_line_purcell()  # 手动选场景
    # plot_dataset_3 = plotter.compute_single_line_PL_factor()  # 手动选场景
    # plotter.plot_line(
    #     x=plot_dataset_3['x'], z1=plot_dataset_3['y'],
    #     default_color='blue', default_linestyle='-'
    # )
    # plotter.add_twin_annotations()  # 注解
    # plotter.save_and_show()  # 保存

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': False, 'cmap': 'magma', 'default_color': 'black',
    #         'global_color_vmin': 0, 'global_color_vmax': 75
    #     },
    #     annotations={
    #         # 'xlim': (-1e-3, 1e-2+1e-3), 'ylim': (0, 100)
    #         'show_axis_labels': True,
    #         'show_tick_labels': True,
    #     }
    # )
    # config.figsize = (4, 3)
    # plotter = MyScript1Plotter(config=config,
    #                            data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_ks\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data() 
    # plotter.new_fig()
    # plot_dataset_1 = plotter.compute_multi_k_lines()  # 手动选场景
    # plotter.config.plot_params = {
    #     'add_colorbar': True, 'cmap': 'rainbow',
    #     'title': False,
    # }
    # plotter.plot_multiline_2d(
    #     x_vals=np.array(plot_dataset_1['x']),
    #     y_vals=np.array(plot_dataset_1['k_list']),
    #     Z=np.array(plot_dataset_1['y_list']).T,
    # )
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存


    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': False, 'cmap': 'magma', 'default_color': 'black',
    #     },
    #     annotations={
    #         # 'xlim': (-1e-3, 1e-2+1e-3), 'ylim': (0, 100), 'add_grid': True
    #         'xlim': (0.50, 0.61), 'ylim': (3e-1, 40),
    #         # 'xlim': (0.48, 0.58), 'ylim': (0, 5)
    #         'y_log_scale': True,
    #     }
    # )
    # config.figsize = (2, 3)
    # plotter = MyScript1Plotter(config=config,
    #                            # data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\QGM\PL_Analysis.json')
    #                            data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\BIC\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()
    # plotter.new_fig()
    # plot_dataset_1 = plotter.compute_multi_k_lines(mode=1)  # 手动选场景
    # plotter.config.plot_params = {
    #     # 'add_colorbar': True, 'cmap': 'magma_r',
    #     'add_colorbar': False, 'cmap': 'viridis_r', 'default_color': 'gray',
    #     'title': False, 'alpha': 0.5,
    # }
    # plotter.plot_multiline_2d(
    #     x_vals=np.array(plot_dataset_1['x']),
    #     y_vals=np.array(plot_dataset_1['k_list']),
    #     Z=np.array(plot_dataset_1['y_list']).T,
    #
    # )
    # plot_dataset_1 = plotter.compute_multi_k_lines(mode=2)  # 手动选场景
    # plotter.config.plot_params = {
    #     # 'add_colorbar': True, 'cmap': 'magma_r',
    #     # 'add_colorbar': False, 'cmap': 'Blues_r',
    #     'add_colorbar': False, 'cmap': 'Reds_r',
    #     'title': False,
    #     'global_color_vmin': 1, 'global_color_vmax': 5,
    # }
    # plotter.plot_multiline_2d(
    #     x_vals=np.array(plot_dataset_1['x']),
    #     y_vals=np.array(plot_dataset_1['k_list']),
    #     Z=np.array(plot_dataset_1['y_list']).T,
    #
    # )
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': False, 'cmap': 'magma', 'default_color': 'black',
    #     },
    #     annotations={
    #         'xlim': (0, 400), 'ylim': (0, 10),
    #         'y_log_scale': False,
    #     }
    # )
    # config.figsize = (2, 4)
    # plotter = SimpleScriptPlotter(config=config,
    #                            data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_nx\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()
    # plotter.new_fig()
    # plot_dataset_1 = plotter.compute_k_max_line()  # 预处理最大值
    # plotter.plot_line(plot_dataset_1['k_array'], plot_dataset_1['purcell_max'], default_color='k', default_linestyle='-')
    # plotter.plot_scatter(x=plot_dataset_1['k_array'], z1=plot_dataset_1['purcell_max'], default_color='red', marker='o',zorder=100, alpha=1)
    # plotter.re_initialized(config=config,
    #                            data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-QGM\sweep_nx\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()
    # plot_dataset_2 = plotter.compute_k_max_line()  # 预处理最大值
    # plotter.plot_line(plot_dataset_2['k_array'], plot_dataset_2['purcell_max'], default_color='k', default_linestyle='-')
    # plotter.plot_scatter(x=plot_dataset_2['k_array'], z1=plot_dataset_2['purcell_max'], default_color='blue', marker='o',zorder=100, alpha=1)
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': False, 'cmap': 'magma', 'default_color': 'black',
    #     },
    #     annotations={
    #         'xlim': (0, 200), 'ylim': (0, 50),
    #         'y_log_scale': False,
    #     }
    # )
    # config.figsize = (2, 4)
    # plotter = SimpleScriptPlotter(config=config,
    #                            data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_nx\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()
    # plotter.new_fig()
    # plot_dataset_1 = plotter.compute_k_max_line()  # 预处理最大值
    # plotter.plot_line(plot_dataset_1['k_array'], plot_dataset_1['purcell_max'], default_color='k', default_linestyle='-')
    # plotter.plot_scatter(x=plot_dataset_1['k_array'], z1=plot_dataset_1['purcell_max'], default_color='k', marker='o',zorder=100, alpha=1)
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存

    # config = PlotConfig(
    #     plot_params={
    #         'add_colorbar': False, 'cmap': 'magma', 'default_color': 'black',
    #     },
    #     annotations={
    #         'ylim': (0, 10),
    #         'y_log_scale': False,
    #     }
    # )
    # config.figsize = (3, 1.5)
    # plotter = SimpleScriptPlotter(config=config,
    #                            data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-BIC\sweep_nx\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()
    # plotter.new_fig()
    # plot_dataset_1 = plotter.compute_multi_k_lines(mode=1)  # 手动选场景
    # # 选择一条来绘制
    # plotter.plot_line(
    #     x=np.array(plot_dataset_1['x']),
    #     z1=np.array(plot_dataset_1['y_list'])[2],  # 选择
    #     default_color='red', default_linestyle='-'
    # )
    # plotter.re_initialized(config=config,
    #                            data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\highQ-QGM\sweep_nx\PL_Analysis.json')
    # plotter.load_data()
    # plotter.prepare_data()
    # plot_dataset_2 = plotter.compute_multi_k_lines(mode=1)  # 手动选场景
    # # 选择一条来绘制
    # plotter.plot_line(
    #     x=np.array(plot_dataset_2['x']),
    #     z1=np.array(plot_dataset_2['y_list'])[2],  # 选择
    #     default_color='blue', default_linestyle='-'
    # )
    # plotter.add_annotations()  # 注解
    # plotter.save_and_show()  # 保存

    config = PlotConfig(
        plot_params={
            'add_colorbar': False, 'cmap': 'magma', 'default_color': 'black',
        },
        annotations={
            'ylim': (0, 50),
            'y_log_scale': False,
        }
    )
    config.figsize = (3, 1.5)
    plotter = SimpleScriptPlotter(config=config,
                               data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\SE\data\lowQ-BIC\sweep_nx\PL_Analysis.json')
    plotter.load_data()
    plotter.prepare_data()
    plotter.new_fig()
    plot_dataset_1 = plotter.compute_multi_k_lines(mode=1)  # 手动选场景
    # 选择一条来绘制
    plotter.plot_line(
        x=np.array(plot_dataset_1['x']),
        z1=np.array(plot_dataset_1['y_list'])[0],  # 选择
        default_color='k', default_linestyle='-'
    )
    plotter.add_annotations()  # 注解
    plotter.save_and_show()  # 保存


