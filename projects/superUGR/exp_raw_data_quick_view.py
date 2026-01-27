import os
import numpy as np
import matplotlib.pyplot as plt


import os
import numpy as np
import matplotlib.pyplot as plt

fs = 9
plt.rcParams.update({"font.size": fs})

def load_and_visualize_ftir_spectra(directory, prefix, start_num, end_num, angles,
                                    file_ext='.dpt', wavenumber_col=0, spectrum_col=1,
                                    cmap='RdBu',
                                    save_plots=False, plot_dir=None,
                                    ref_filename=None, ref_usecols=None,
                                    bg_filename=None, bg_usecols=None,
                                    interpolate_ref=False, eps=0):
    """
    加载FTIR谱数据并可视化，并可选用参考谱做分母归一化/比值计算。

    返回:
    - data_dict: dict, 键为角度，值为 (wavenumbers, spectrums_processed) 的元组。
    """
    num_files = end_num - start_num + 1
    if len(angles) != num_files:
        raise ValueError(f"角度列表长度 {len(angles)} 不匹配文件数量 {num_files}。")

    if save_plots and not plot_dir:
        raise ValueError("save_plots=True 时必须提供 plot_dir。")

    # -------- 1) 读取参考谱（如果提供）--------
    ref_w = None
    ref_s = None
    if ref_filename is not None:
        # 允许用户传完整路径；否则默认在 directory 下
        ref_path = ref_filename if os.path.isabs(ref_filename) else os.path.join(directory, ref_filename)
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"参考谱文件 {ref_path} 不存在。")

        if ref_usecols is None:
            ref_usecols = (wavenumber_col, spectrum_col)

        ref_data = np.loadtxt(ref_path, usecols=ref_usecols)
        ref_w = ref_data[:, 0]
        ref_s = ref_data[:, 1]

    # -------- 1) 读取背景谱（如果提供）--------
    bg_w = None
    bg_s = None
    if bg_filename is not None:
        # 允许用户传完整路径；否则默认在 directory 下
        bg_path = bg_filename if os.path.isabs(bg_filename) else os.path.join(directory, bg_filename)
        if not os.path.exists(bg_path):
            raise FileNotFoundError(f"参考谱文件 {bg_path} 不存在。")

        if bg_usecols is None:
            bg_usecols = (wavenumber_col, spectrum_col)

        bg_data = np.loadtxt(bg_path, usecols=bg_usecols)
        bg_w = bg_data[:, 0]
        bg_s = bg_data[:, 1]

    # -------- 2) 加载样品谱 --------
    data_dict = {}
    all_spectrums = []
    common_wavenumbers = None

    for i, num in enumerate(range(start_num, end_num + 1)):
        angle = angles[i]
        filename = f"{prefix}{num}{file_ext}"
        filepath = os.path.join(directory, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件 {filepath} 不存在。")

        data = np.loadtxt(filepath, usecols=(wavenumber_col, spectrum_col))
        wavenumbers = data[:, 0]
        spectrums = data[:, 1]

        # 统一波数网格检查（样品之间）
        if common_wavenumbers is None:
            common_wavenumbers = wavenumbers
        elif not np.array_equal(common_wavenumbers, wavenumbers):
            raise ValueError(
                f"文件 {filename} 的波数网格与前文件不一致，无法直接创建heatmap。需插值处理。"
            )

        # -------- 3) 减去背景谱（如果提供）--------
        if bg_filename is not None:
            if np.array_equal(bg_w, wavenumbers):
                bg_s = bg_s
            else:
                if not interpolate_ref:
                    raise ValueError(
                        f"参考谱波数网格与样品不一致（ref={len(ref_w)}点, sample={len(wavenumbers)}点）。"
                        f"如需自动插值，请设置 interpolate_ref=True。"
                    )
                # 将参考谱插值到样品波数网格
                bg_s = np.interp(wavenumbers, bg_w, bg_s)
        else:
            bg_s = 0.0

        # -------- 3) 用参考谱做分母（如果提供）--------
        if ref_filename is not None:
            if np.array_equal(ref_w, wavenumbers):
                denom = ref_s
            else:
                if not interpolate_ref:
                    raise ValueError(
                        f"参考谱波数网格与样品不一致（ref={len(ref_w)}点, sample={len(wavenumbers)}点）。"
                        f"如需自动插值，请设置 interpolate_ref=True。"
                    )
                # 将参考谱插值到样品波数网格
                denom = np.interp(wavenumbers, ref_w, ref_s)
        else:
            denom = 1.0

        # spectrums = (spectrums/(1-3.37e-2) - bg_s) / (denom - bg_s)
        spectrums = (spectrums/(1) - bg_s) / (denom - bg_s)
        # spectrums = (spectrums) / (denom)

        data_dict[angle] = (wavenumbers, spectrums)
        all_spectrums.append(spectrums)

    all_spectrums = np.array(all_spectrums)

    # -------- 4) 可视化1：曲线图 --------
    cmap_lines = plt.get_cmap('viridis', num_files)
    color_list = [cmap_lines(i) for i in range(num_files)]

    plt.figure(figsize=(4, 3))
    for i, (angle, (wavenumbers, spectrums)) in enumerate(data_dict.items()):
        plt.plot(wavenumbers, spectrums, label=f'Angle: {angle}°', color=color_list[i])

    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('spectrum' if ref_filename is None else 'spectrum / reference')
    plt.title('FTIR Spectra at Different Angles' + ('' if ref_filename is None else ' (Divided by Reference)'))
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    if save_plots:
        curve_path = os.path.join(plot_dir, 'ftir_curves.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        print(f"曲线图已保存到 {curve_path}")
    plt.show()

    # -------- 5) 可视化2：heatmap --------
    common_y = common_wavenumbers * 0.03  # cm⁻¹ -> THz

    # # 采用对称的angle范围, 手动增添对称数据
    # supp_angles = [-a for a in angles[::-1]]
    # supp_spectrums = all_spectrums[::-1, :]
    # angles = supp_angles + angles
    # all_spectrums = np.vstack((supp_spectrums, all_spectrums))

    plt.figure(figsize=(3, 4))
    plt.imshow(
        all_spectrums.T, aspect='auto',
        extent=[min(angles), max(angles), common_y[0], common_y[-1]],
        origin='lower',
        cmap=cmap,
        interpolation='none',
        # vmin/vmax 你可以根据“除以参考谱后的范围”重新调整
        vmin=0, vmax=1,
    )
    plt.colorbar(label=('spectrum' if ref_filename is None else 'spectrum / reference'), pad=0.25)
    plt.xlabel('Angle (°)')
    plt.ylabel('Frequency (THz)')
    plt.ylim((70, 90))
    # 添加一个次坐标轴显示波长
    ax = plt.gca()
    secax = ax.secondary_yaxis('right', functions=(lambda x: 300 / x, lambda x: 300 / x))
    secax.set_ylabel('Wavelength (μm)')

    # plt.ylabel('Wavelength (μm)')
    # plt.ylim((5, 4.2))
    # plt.gca().invert_yaxis()

    if save_plots:
        heatmap_path = os.path.join(plot_dir, 'ftir_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap已保存到 {heatmap_path}")
    plt.savefig('temp.svg', dpi=300, bbox_inches='tight', transparent=True)
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

# # EXP @2026.1.8 KEREN Sair with reference spectrum
# prefix = ''
# directory = r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.8 KEREN\26.1.8\Sair'
# start_num = -6
# end_num = 15
#
# angles = list(range(start_num, end_num+1))
# data = load_and_visualize_ftir_spectra(
#     directory, prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
#     ref_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.8 KEREN\26.1.8\Sair\BALCKBODY.dpt",
#     bg_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.9 KEREN\air.dpt"
# )

# # EXP @2026.1.8 KEREN Pair with reference spectrum
# prefix = ''
# directory = r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.8 KEREN\26.1.8\Pair'
# start_num = -5
# end_num = 15
#
# angles = list(range(start_num, end_num+1))
# data = load_and_visualize_ftir_spectra(
#     directory, prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
#     ref_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.8 KEREN\26.1.8\Pair\BLACKBODY.dpt",
#     bg_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.9 KEREN\air.dpt"
# )

# # EXP @2026.1.9 KEREN S-back with reference spectrum
# prefix = ''
# directory = r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.9 KEREN\Sback'
# start_num = -3
# end_num = 15
#
# angles = list(range(start_num, end_num+1))
# data = load_and_visualize_ftir_spectra(
#     directory, prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
#     ref_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.8 KEREN\26.1.8\Pair\BLACKBODY.dpt",
#     bg_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.9 KEREN\air.dpt"
# )

# # EXP @2026.1.23 KEREN S-forw with reference spectrum
# prefix = 'Sample name.'
# start_num = 6932
# end_num = 6950
#
# angles = list(range(start_num, end_num+1))
# data = load_and_visualize_ftir_spectra(
#     r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\zheng-s-2',
#     prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
#     ref_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\heiti_beijing\blackbody.dpt",
#     bg_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\heiti_beijing\beijing.dpt"
# )
#
# # EXP @2026.1.23 KEREN P-forw with reference spectrum
# prefix = 'Sample name.'
# start_num = 6911
# end_num = 6931
#
# angles = list(range(start_num, end_num+1))
# data = load_and_visualize_ftir_spectra(
#     r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\zheng-p',
#     prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
#     ref_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\heiti_beijing\blackbody.dpt",
#     bg_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\heiti_beijing\beijing.dpt"
# )
#
# # EXP @2026.1.23 KEREN S-back with reference spectrum
# prefix = 'Sample name.'
# start_num = 6964
# end_num = 6984
#
# angles = list(range(start_num, end_num+1))
# data = load_and_visualize_ftir_spectra(
#     r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\bei-s',
#     prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
#     ref_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\heiti_beijing\blackbody.dpt",
#     bg_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\heiti_beijing\beijing.dpt"
# )

# # EXP @2026.1.23 KEREN P-back with reference spectrum
# prefix = 'Sample name.'
# start_num = 6991
# end_num = 7006
#
# angles = list(range(start_num, end_num+1))
# data = load_and_visualize_ftir_spectra(
#     r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\bei-p',
#     prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
#     ref_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\heiti_beijing\blackbody.dpt",
#     bg_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\heiti_beijing\beijing.dpt"
# )

# # EXP @2026.1.23 KEREN P-back with reference spectrum
# prefix = 'Sample name.'
# start_num = 6894
# end_num = 6910
#
# angles = list(range(start_num, end_num+1))
# data = load_and_visualize_ftir_spectra(
#     r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\zheng-s',
#     prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
#     ref_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\heiti_beijing\blackbody.dpt",
#     bg_filename=r"D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.23 KEREN\heiti_beijing\beijing.dpt"
# )

# EXP @2026.1.23 KEREN s-forw with reference spectrum
prefix = 'Sample name.'
start_num = 7191
end_num = 7206

angles = list(range(start_num, end_num+1))
data = load_and_visualize_ftir_spectra(
    r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.26\S ZHENG zhengxu(-5-15)',
    prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
    cmap='Blues_r'
)

# # EXP @2026.1.23 KEREN s-back with reference spectrum
# prefix = 'Sample name.'
# start_num = 7208
# end_num = 7208+20
#
# angles = list(range(start_num, end_num+1))
# data = load_and_visualize_ftir_spectra(
#     r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.26\S BEI daoxu(15--5)',
#     prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
#     cmap='Blues_r'
# )

# EXP @2026.1.23 KEREN P-forw with reference spectrum
prefix = 'Sample name.'
start_num = 7132
end_num = 7132+20

angles = list(range(start_num, end_num+1))
data = load_and_visualize_ftir_spectra(
    r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.26\P ZHENG zhengxu(-5-15)',
    prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
    cmap='Blues_r'
)

# # EXP @2026.1.23 KEREN P-back with reference spectrum
# prefix = 'Sample name.'
# start_num = 7159
# end_num = 7159+20
#
# angles = list(range(start_num, end_num+1))
# data = load_and_visualize_ftir_spectra(
#     r'D:\DELL\Documents\myPlots\projects\superUGR\data\exp\26.1.26\P BEI daoxu(15--5)',
#     prefix, start_num, end_num, angles, save_plots=True, plot_dir='./', file_ext='.dpt',
#     cmap='Blues_r'
# )
