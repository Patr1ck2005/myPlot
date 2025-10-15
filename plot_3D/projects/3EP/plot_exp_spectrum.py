import numpy as np
import pandas as pd
import os
from pathlib import Path

from plot_3D.core.plot_workflow import HeatmapPlotter, PlotConfig


c_const = 299792458  # 光速 (m/s)

class CustomPlotter(HeatmapPlotter):
    def load_data(self) -> None:
        self.wavelength = pd.read_csv(
            self.data_path,
            sep=',',  # 假设分隔符是逗号；如果实际是制表符，改为 '\t'
            skiprows=3,  # 跳过前 3 行（元信息和空行）
            nrows=1,
            header=None,
            usecols=range(1, 3039 + 2),
        )
        print(self.wavelength)
        self.raw_dataset = pd.read_csv(
            self.data_path,
            sep=',',  # 假设分隔符是逗号；如果实际是制表符，改为 '\t'
            skiprows=4,  # 跳过前 3 行（元信息和空行）
            nrows=511 + 1,
            header=None,
            usecols=range(1, 3039 + 2),
        )  # 使用第一列作为索引
        print(self.raw_dataset)

    def prepare_data(self) -> None:
        self.Z1 = self.raw_dataset.to_numpy()
        self.y_vals = self.wavelength.values.flatten()
        # self.x_vals = self.raw_dataset.index.values
        # self.x_vals = np.linspace(-50, 50, len(self.raw_dataset.columns))
        self.x_vals = np.linspace(-50, 50, len(self.raw_dataset.index.values))

    def plot(self) -> None:  # 重写：调用骨架
        self.plot_heatmap(self.Z1, self.x_vals, self.y_vals, )
        # 导出imshow中纯净的绘图数据
        im_data = self.ax.images[0].get_array().data  # 获取imshow的纯净数据
        # 筛选范围
        # 波长范围 1000-1300nm
        y_mask = (self.y_vals >= 1000) & (self.y_vals <= 1300)
        im_data = im_data[y_mask, :]
        # 角度范围 -15 到 15 度（本来就是这个范围）
        x_mask = (self.x_vals >= -15) & (self.x_vals <= 15)
        im_data = im_data[:, x_mask]
        # 保存为 pkl 文件
        save_dict = {
            'x_vals': np.sin(np.deg2rad(self.x_vals[x_mask])),  # 转为NA
            'y_vals': c_const/self.y_vals[y_mask]/(c_const/991),  # 转为归一化频率 (c/P) P=991nm
            'subs': [im_data],
        }
        # print(c_const/self.y_vals/(c_const/991))
        save_path = os.path.join(os.path.dirname(self.data_path), 'im_data.pkl')
        pd.to_pickle(save_dict, save_path)
        print(f"纯净的绘图数据已保存到 {save_path}")



# 新增函数：批量处理 data 目录下的所有 CSV 文件
def batch_plot(data_dir: str, batch_mode: bool = True) -> None:
    """
    批量绘图函数。
    - 如果 batch_mode=True，遍历 data_dir 下的所有 CSV 文件，分别绘图并保存到原目录。
    - 标题设置为文件名（去扩展名）。
    - 保存图片名为 文件名（去扩展名）.png，到 CSV 所在的目录。
    """
    if not batch_mode:
        print("批量模式关闭，使用单文件模式。")
        return

    # 遍历目录下的所有 CSV 文件
    for file_name in os.listdir(data_dir):
        if file_name.lower().endswith('.csv'):
            full_path = os.path.join(data_dir, file_name)
            print(f"处理文件：{full_path}")

            # 获取文件名（去扩展名）作为标题和保存名
            base_name = Path(file_name).stem  # 如 'MZH-3EP-Y-pol-220Dose-Calc'

            # 创建 PlotConfig，设置标题为文件名
            config = PlotConfig(
                plot_params={
                    'add_colorbar': True, 'cmap': 'gray',
                    'title': False, 'global_color_vmin': 0, 'global_color_vmax': 1
                },
                annotations={
                    'xlabel': r'$\theta$', 'ylabel': r'$\lambda (nm)$',
                    'xlim': (-50, 50),
                    'ylim': (1000, 1400),
                    'title': base_name,  # 设置标题为文件名
                    'show_axis_labels': True,
                    'show_tick_labels': True,
                }
            )
            config.show = False

            # 创建 plotter 实例
            plotter = CustomPlotter(
                config=config,
                data_path=full_path  # 当前 CSV 文件路径
            )

            # 执行绘图流程
            plotter.load_data()
            plotter.prepare_data()
            plotter.new_fig()
            plotter.plot()
            plotter.add_annotations()
            plotter.ax.invert_yaxis()

            # 保存到原目录，使用文件名作为 custom_name
            save_dir = os.path.dirname(full_path)+'\\'+base_name  # CSV 所在的目录
            plotter.save_and_show(save=True, custom_abs_path=save_dir, save_type='png')

    print("批量绘图完成！🎉")


if __name__ == '__main__':
    # 单文件模式（示例）
    single_config = PlotConfig(
        plot_params={
            'add_colorbar': True, 'cmap': 'magma',
            'title': False, 'global_color_vmin': 0, 'global_color_vmax': 1
        },
        annotations={
            'xlabel': r'$\theta$', 'ylabel': r'$\lambda (nm)$',
            'xlim': (-15, 15),
            'ylim': (1000, 1300),
            'title': 'test',
            'show_axis_labels': True,
            'show_tick_labels': True,
        }
    )
    single_plotter = CustomPlotter(
        config=single_config,
        data_path=r'D:\DELL\Documents\myPlots\plot_3D\projects\3EP\data\MZH-3EP-X-pol-260Dose'
                  r'-Calc\MZH-3EP-Y-pol-200Dose-Calc.csv',
    )
    # 单文件执行（注释掉以切换）
    single_plotter.load_data()
    single_plotter.prepare_data()
    single_plotter.new_fig()
    single_plotter.plot()
    single_plotter.add_annotations()
    single_plotter.ax.invert_yaxis()
    single_plotter.save_and_show(save=True, custom_name='test', custom_abs_path=None)

    # # 批量模式：调用 batch_plot，传入 data 目录路径，设置 batch_mode=True
    # data_dir = r'D:\DELL\Documents\myPlots\plot_3D\projects\3EP\data\MZH-3EP-X-pol-260Dose-Calc'  # 替换为你的 data 目录路径
    # batch_plot(data_dir, batch_mode=True)  # 设置为 False 以关闭批量模式
