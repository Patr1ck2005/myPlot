import os
import numpy as np
import matplotlib.pyplot as plt


def load_and_visualize_ftir_spectra(directory, prefix, start_num, end_num, angles,
                                    file_ext='.dpt', wavenumber_col=0, spectrum_col=1,
                                    save_plots=False, plot_dir=None):
    """
    加载FTIR谱数据并可视化。

    参数:
    - directory: str, 数据文件所在的目录路径。
    - prefix: str, 文件名前缀，如 'Sample name.'。
    - start_num: int, 起始编号，如 5897。
    - end_num: int, 结束编号，如 5912。
    - angles: list of float/int, 角度列表，必须与文件数量匹配，如 [0, 1, 2, ..., 15]。
    - file_ext: str, 文件后缀，默认 '.dpt'。
    - wavenumber_col: int, 波数列索引，默认 0。
    - spectrum_col: int, 光谱列索引，默认 1。
    - save_plots: bool, 是否保存图像，默认 False。
    - plot_dir: str or None, 保存图像的目录，如果 save_plots=True，则必须提供。

    返回:
    - data_dict: dict, 键为角度，值为 (wavenumbers, spectrums) 的元组。
    """
    # 检查角度列表长度是否匹配文件数量
    num_files = end_num - start_num + 1
    if len(angles) != num_files:
        raise ValueError(f"角度列表长度 {len(angles)} 不匹配文件数量 {num_files}。")

    # 初始化数据存储
    data_dict = {}
    all_spectrums = []  # 用于heatmap的2D数组
    common_wavenumbers = None  # 假设所有文件有相同的波数网格

    # 按顺序加载文件
    for i, num in enumerate(range(start_num, end_num + 1)):
        angle = angles[i]
        filename = f"{prefix}{num}{file_ext}"
        filepath = os.path.join(directory, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件 {filepath} 不存在。")

        # 读取数据：假设两列，制表符分隔
        data = np.loadtxt(filepath, usecols=(wavenumber_col, spectrum_col))
        wavenumbers = data[:, 0]
        spectrums = data[:, 1]

        data_dict[angle] = (wavenumbers, spectrums)

        # 为heatmap准备：检查波数是否一致
        if common_wavenumbers is None:
            common_wavenumbers = wavenumbers
        elif not np.array_equal(common_wavenumbers, wavenumbers):
            raise ValueError(f"文件 {filename} 的波数网格与前文件不一致，无法直接创建heatmap。需插值处理。")

        all_spectrums.append(spectrums)

    # 转换为2D数组用于heatmap：行=角度，列=波数
    all_spectrums = np.array(all_spectrums)

    # 可视化1: 不同角度的曲线图
    cmap_lines = plt.get_cmap('viridis', num_files)
    color_list = [cmap_lines(i) for i in range(num_files)]

    plt.figure(figsize=(6, 4))
    for i, (angle, (wavenumbers, spectrums)) in enumerate(data_dict.items()):
        plt.plot(wavenumbers, spectrums, label=f'Angle: {angle}°', color=color_list[i])
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('spectrum')
    plt.title('FTIR spectrum Spectra at Different Angles')
    plt.legend()
    plt.grid(True)
    if save_plots:
        curve_path = os.path.join(plot_dir, 'ftir_curves.png')
        plt.savefig(curve_path)
        print(f"曲线图已保存到 {curve_path}")
    plt.show()

    # 可视化2: 二维heatmap
    # 转换成THz频率
    common_y = common_wavenumbers * 0.03  # cm⁻¹ to THz
    plt.figure(figsize=(5, 6))
    # 纵轴波数，横轴角度(0-15)
    plt.imshow(
        all_spectrums.T, aspect='auto',
        extent=[min(angles), max(angles), common_y[0], common_y[-1]],
        origin='lower', cmap='gray',
        interpolation='none',
        vmin=0.6, vmax=1,
    )
    plt.colorbar(label='spectrum')
    plt.xlabel('Angle (°)')
    # plt.ylabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Frequency (THz)')
    plt.ylim((80, 95))
    plt.gca().invert_yaxis()
    if save_plots:
        heatmap_path = os.path.join(plot_dir, 'ftir_heatmap.png')
        plt.savefig(heatmap_path)
        print(f"Heatmap已保存到 {heatmap_path}")
    plt.show()

    return data_dict


# # 示例调用（根据你的数据，替换为实际值）
# directory = r'D:\DELL\Documents\myPlots\plot_3D\projects\superUGR\data\exp\25.10.13 KEREN\1'
# prefix = 'Sample name.'
# start_num = 5897
# end_num = 5912

# angles = list(range(0, 16))  # 0到15，共16个
# data = load_and_visualize_ftir_spectra(directory, prefix, start_num, end_num, angles, save_plots=True, plot_dir='./')

# directory = r'D:\DELL\Documents\myPlots\plot_3D\projects\superUGR\data\exp\25.10.13 KEREN\2'
# prefix = 'Sample name.'
# start_num = 5913
# end_num = 5913 + 15
# data = load_and_visualize_ftir_spectra(directory, prefix, start_num, end_num, angles, save_plots=True, plot_dir='./')

# 示例调用（根据你的数据，替换为实际值）

# prefix = 'Sample name.'
# directory = r'D:\DELL\Documents\myPlots\plot_3D\projects\superUGR\data\exp\25.10.20 KEREN\1'
# start_num = 6028
# end_num = 6043

# directory = r'D:\DELL\Documents\myPlots\plot_3D\projects\superUGR\data\exp\25.10.20 KEREN\2'
# start_num = 6044
# end_num = 6059

prefix = ''
directory = r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.8 KEREN\26.1.8\Sair'
start_num = 0
end_num = 15

angles = list(range(0, 16))  # 0到15，共16个
data = load_and_visualize_ftir_spectra(directory, prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt')
