import numpy as np
import pandas as pd
import os
from pathlib import Path

import scipy
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from core.plot_workflow import HeatmapPlotter, PlotConfig

c_const = 299792458  # 光速 (m/s)


class CustomPlotter(HeatmapPlotter):
    def load_data(self) -> None:
        self.wavelength = pd.read_csv(
            self.data_path,
            sep=',',  # 假设分隔符是逗号；如果实际是制表符，改为 '\t'
            skiprows=0,
            nrows=1,
            header=None,
        )
        self.wavelength = self.wavelength.iloc[:, 1:]  # 跳过第一列
        print(self.wavelength)
        self.raw_dataset = pd.read_csv(
            self.data_path,
            sep=',',  # 假设分隔符是逗号；如果实际是制表符，改为 '\t'
            skiprows=1,
            header=None,
        )  # 使用第一列作为索引
        self.raw_dataset = self.raw_dataset.iloc[:, 1:]  # 跳过第一列
        print(self.raw_dataset)

    def prepare_data(self) -> None:
        self.Z1 = self.raw_dataset.to_numpy()
        self.y_vals = self.wavelength.values.flatten()
        # self.x_vals = self.raw_dataset.index.values
        # self.x_vals = np.linspace(-50, 50, len(self.raw_dataset.columns))
        self.x_vals = np.linspace(-49.5, 50, len(self.raw_dataset.index.values))

    def plot(self) -> None:  # 重写：调用骨架
        self.plot_heatmap(self.Z1, self.x_vals, self.y_vals, )
        # 导出imshow中纯净的绘图数据
        im_data = self.Z1  # 获取imshow的纯净数据
        # 筛选范围
        # 波长范围 1000-1300nm
        y_mask = (self.y_vals >= 1000) & (self.y_vals <= 1500)
        # y_mask = (self.y_vals >= 0)
        im_data = im_data[:, y_mask]
        # 角度范围 -15 到 15 度（本来就是这个范围）
        # x_mask = (self.x_vals >= -15) & (self.x_vals <= 15)
        x_mask = (self.x_vals >= -30) & (self.x_vals <= 30)
        # x_mask = (self.x_vals >= -100)
        im_data = im_data[x_mask, :]
        # # clip 0-1
        # im_data = np.clip(im_data, 0, 1)
        # renorm to 0-1
        # 使用 SG 平滑后的最大值进行归一化, 沿着 y 轴平滑
        smoothed_data = scipy.signal.savgol_filter(im_data, window_length=11*5, polyorder=3, axis=1)
        # # 随便绘制一条曲线看效果
        # plt.figure(figsize=(6, 4))
        # plt.plot(self.y_vals[y_mask], smoothed_data[40, :])
        # plt.scatter(self.y_vals[y_mask], im_data[40, :], s=1, color='red')
        # plt.xlabel('Wavelength (nm)')
        # plt.ylabel('Smoothed Intensity (a.u.)')
        # plt.title('Smoothed Spectrum at Angle Index 15')
        # plt.show()
        im_data /= np.max(smoothed_data)
        # clip 0-1
        im_data = np.clip(im_data, 0, 1)
        # 保存为 pkl 文件
        save_dict = {
            'x_vals': np.deg2rad(self.x_vals[x_mask]),  # 转为NA
            # 'y_vals': c_const/self.y_vals[y_mask]/(c_const/894),  # 转为归一化频率 (c/P) P=894nm
            'y_vals': self.y_vals[y_mask] / 894,  # 转为归一化波长 (P) P=894nm
            'subs': [im_data],
        }
        # # plt预览
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(6, 4))
        # plt.imshow(
        #     save_dict['subs'][0].T, extent=(save_dict['x_vals'][0], save_dict['x_vals'][-1],
        #                                     save_dict['y_vals'][0], save_dict['y_vals'][-1]), aspect='auto',
        #     cmap='gray',
        #     origin='lower',
        # )
        # plt.colorbar(label='Intensity (a.u.)')
        # plt.xlabel(r'$\theta$ (rad)')
        # plt.ylabel(r'$\lambda$ (P)')
        # plt.title('Filtered Experimental Spectrum')
        # plt.show()
        # # 实验性功能: 重新映射到 归一化频率 和 k 空间
        # 测试 wavelength_angle_to_freq_k_space 函数
        # 示例数据
        wavelengths = self.y_vals[y_mask]  # 波长
        angles = self.x_vals[x_mask]  # 角度
        P = 894  # 周期
        # 调用函数
        f_grid, k_grid, Z_interp, _, _ = wavelength_angle_to_norm_freq_k_space(wavelengths, angles, im_data.T, P)
        # 可视化结果
        plt.figure(figsize=(6, 4))
        plt.imshow(
            Z_interp.T, extent=(k_grid[0, 0], k_grid[-1, 0], f_grid[0, 0], f_grid[0, -1]), aspect='auto',
            cmap='gray',
            origin='lower',
        )
        plt.colorbar(label='Z')
        plt.xlabel('k-space (sin(θ) * P/λ)')
        plt.ylabel('Normalized Frequency (P/λ)')
        plt.title('Interpolated Z in Normalized Frequency and k-space')
        plt.show()
        # 保存结果
        save_dict = {
            'x_vals': k_grid[:, 0],  # k_space
            'y_vals': f_grid[0, :],  # norm_freq
            'subs': [Z_interp],
        }
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
                    'xlim': (-15, 15),
                    'ylim': (1100, 1400),
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
            plotter.new_2d_fig()
            plotter.plot()
            plotter.add_annotations()
            plotter.ax.invert_yaxis()

            # 保存到原目录，使用文件名作为 custom_name
            save_dir = os.path.dirname(full_path) + '\\' + base_name  # CSV 所在的目录
            plotter.save_and_show(save=True, custom_abs_path=save_dir, save_type='png')

    print("批量绘图完成！🎉")


import numpy as np
from scipy.interpolate import RegularGridInterpolator


def wavelength_angle_to_norm_freq_k_space(wavelengths, angles, Z, P):
    """
    转换波长和角度到归一化频率和k空间，在形成的梯形区域中找到最大的内接矩形区域，
    然后重新生成均匀网格进行采样（插值）。

    :param wavelengths: 波长数组 (nm), 一维
    :param angles: 角度数组 (degrees), 一维
    :param Z: 与波长和角度对应的物理量，二维数组 (len(wavelengths), len(angles)) 或反之
    :param P: 周期 (nm)
    :return: 插值后的 f_grid, k_grid, Z_interp, 以及原始 norm_freq, k_space
    """

    # 确保输入为numpy数组
    wavelengths = np.array(wavelengths)
    angles = np.array(angles)
    Z = np.array(Z)

    # 检查并调整 Z 形状：确保第一维对应 wavelengths，第二维对应 angles
    len_w = len(wavelengths)
    len_a = len(angles)
    if Z.shape == (len_a, len_w):
        Z = Z.T  # 转置以匹配 (wavelengths, angles)
    elif Z.shape != (len_w, len_a):
        raise ValueError(f"Z 形状 {Z.shape} 必须为 ({len_w}, {len_a}) 或 ({len_a}, {len_w})")

    # 生成网格
    angles_grid, wavelengths_grid = np.meshgrid(angles, wavelengths)  # 形状 (len_w, len_a)

    # 计算归一化频率和 k 空间
    norm_freq = P / wavelengths_grid
    k_space = np.sin(np.deg2rad(angles_grid)) * norm_freq

    # 确定梯形边界
    f_min = np.min(norm_freq)
    f_max = np.max(norm_freq)
    sin_theta_max = np.sin(np.deg2rad(np.max(angles)))
    sin_theta_min = np.sin(np.deg2rad(np.min(angles)))

    # 步骤2: 计算最大的内接矩形
    if f_max / 2 >= f_min:
        f_rect_min = f_max / 2
    else:
        f_rect_min = f_min
    f_rect_max = f_max
    k_rect_min = sin_theta_min * f_rect_min
    k_rect_max = sin_theta_max * f_rect_min

    # 步骤3: 生成均匀网格
    num_f = len(wavelengths)
    num_k = len(angles)
    f_grid_1d = np.linspace(f_rect_min, f_rect_max, num_f)
    k_grid_1d = np.linspace(k_rect_min, k_rect_max, num_k)
    f_grid, k_grid = np.meshgrid(f_grid_1d, k_grid_1d)  # 形状 (num_k, num_f)

    # 步骤4: 创建 λ-θ 空间的插值器（假设 wavelengths 和 angles 已递增排序）
    interpolator = RegularGridInterpolator((wavelengths, angles), Z, method='linear', bounds_error=False,
                                           fill_value=np.nan)

    # 对于每个网格点，反解 λ 和 θ
    lambda_grid = P / f_grid
    with np.errstate(invalid='ignore'):  # 忽略潜在的无效值
        theta_grid = np.rad2deg(np.arcsin(k_grid / f_grid))

    # 准备插值点
    points = np.stack((lambda_grid.flatten(), theta_grid.flatten()), axis=-1)
    Z_interp_flat = interpolator(points)
    Z_interp = Z_interp_flat.reshape(f_grid.shape)

    # fig, ax = plt.subplots(figsize=(6, 4))
    # # 使用 pcolormesh 绘制变形后的数据（梯形区域）
    # pcm = ax.pcolormesh(norm_freq, k_space, Z, cmap='viridis', shading='auto')
    # ax.set_xlabel('Normalized Frequency (P/λ)')
    # ax.set_ylabel('k-space (sin(θ) * P/λ)')
    # ax.set_title('Deformed Data (pcolormesh)')
    # fig.colorbar(pcm, ax=ax, label='Z')
    # plt.show()

    return f_grid, k_grid, Z_interp, norm_freq, k_space


if __name__ == '__main__':
    # 示例数据
    wavelengths = np.linspace(400, 800, 50)  # 波长从400nm到800nm
    angles = np.linspace(0, 60, 30)  # 角度从0°到60°
    Z = np.random.rand(50, 30)  # 物理量，形状为(50, 30)，对应(wavelengths, angles)
    P = 500  # 周期500nm

    # 调用函数
    f_grid, k_grid, Z_interp, norm_freq, k_space = wavelength_angle_to_norm_freq_k_space(wavelengths, angles, Z, P)

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 使用 pcolormesh 绘制变形后的数据（梯形区域）
    pcm = ax1.pcolormesh(norm_freq, k_space, Z, cmap='viridis', shading='auto')
    ax1.set_xlabel('Normalized Frequency (P/λ)')
    ax1.set_ylabel('k-space (sin(θ) * P/λ)')
    ax1.set_title('Deformed Data (pcolormesh)')
    fig.colorbar(pcm, ax=ax1, label='Z')

    # 使用 imshow 绘制插值后的方形数据
    im = ax2.imshow(Z_interp, origin='lower', cmap='viridis',
                    extent=[f_grid.min(), f_grid.max(), k_grid.min(), k_grid.max()])
    ax2.set_xlabel('Normalized Frequency (P/λ)')
    ax2.set_ylabel('k-space (sin(θ) * P/λ)')
    ax2.set_title('Interpolated Square Data (imshow)')
    fig.colorbar(im, ax=ax2, label='Z')

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

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
        data_path=r"D:\DELL\Documents\myPlots\projects\EP3\data\20251127角分辨反射数据\sample1125-y-pol-dose225(15)  36_processed.csv",
    )
    # 单文件执行（注释掉以切换）
    single_plotter.load_data()
    single_plotter.prepare_data()
    single_plotter.new_2d_fig()
    single_plotter.plot()
    single_plotter.add_annotations()
    single_plotter.ax.invert_yaxis()
    single_plotter.save_and_show(save=True, custom_name='test', custom_abs_path=None)

    # # 批量模式：调用 batch_plot，传入 data 目录路径，设置 batch_mode=True
    # data_dir = r'D:\DELL\Documents\myPlots\projects\EP3\data\20251127角分辨反射数据'  # 替换为你的 data 目录路径
    # batch_plot(data_dir, batch_mode=True)  # 设置为 False 以关闭批量模式
