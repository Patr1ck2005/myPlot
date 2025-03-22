import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def plot_band_surfaces(data):
    # 准备绘图
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 提取频率并归一化
    frequencies = [complex(d[3].replace('i', 'j')) for d in data]  # 提取复数频率
    freq_real = [freq.real for freq in frequencies]  # 实部用于颜色映射
    freq_imag = [freq.imag for freq in frequencies]  # 虚部用于透明度

    freq_min, freq_max = 90, 120
    freq_length = freq_max-freq_min

    freq_im_min = 0.5
    freq_im_max = 1.4
    freq_im_length = freq_im_max-freq_im_min

    # 创建颜色映射对象
    refreq_norm = plt.Normalize(vmin=freq_min, vmax=freq_max)
    imfreq_norm = plt.Normalize(vmin=freq_im_min, vmax=freq_im_max)
    cmap = plt.get_cmap('twilight')

    # 选择坐标和频率数据进行绘制
    for i, d in enumerate(data):
        buffer, a, k1, freq, freq_real, Q, S_air_prop = d
        if k1 != 0.00:
            print('k1 skip')
            continue
        if Q < 30:
            print('Q skip')
            continue
        # elif S_air_prop > 10:
        #     print('S_air_prop skip')
        freq_complex = complex(freq.replace('i', 'j'))
        freq_re = freq_complex.real
        freq_im = freq_complex.imag
        # .98~136
        # if freq_re < 98 or freq_re > 136:
        if freq_re < freq_min or freq_re > freq_max:
            print('freq skip')
            continue

        edge = False

        # z = freq_re
        z = freq_im

        # 频率的实部和虚部影响颜色和透明度
        # color = cmap(refreq_norm(freq_re))
        color = cmap(imfreq_norm(freq_im))
        # alpha = np.clip(30/Q/2, 0, 1)  # 透明度由虚部控制
        alpha = 1  # 透明度由虚部控制

        # 绘制曲面
        if edge:
            ax.scatter(buffer, a, z, color=color, alpha=alpha, s=10, edgecolors='black', linewidths=2)  # 使用散点绘图
        else:
            ax.scatter(buffer, a, z, color=color, alpha=alpha, s=10)

    # 创建X和Z的网格数据
    y = np.linspace(0, 0.16, 10)  # X的范围
    z = np.linspace(0, 1.4, 10)  # Z的范围
    y, z = np.meshgrid(y, z)  # 生成网格

    # 计算对应的x值，y=2
    x = np.full_like(y, 1400)

    # 绘制平面
    ax.plot_surface(x, y, z, color='white', alpha=0.5)

    # # 设置图形格式
    # # ax.set_xlabel('kx (2π/P)', labelpad=12)
    # # ax.set_ylabel('ky (2π/P)', labelpad=12)
    # # ax.set_zlabel('Frequency (THz)', labelpad=30)
    # # ax.grid(False)
    # ax.set_xlim(-0.15, 0.15)
    # ax.set_ylim(-0.15, 0.15)
    # ax.set_zlim(freq_min-.05*freq_length, freq_max+.1*freq_length)
    # ax.set_xticks([-0.1, 0, 0.1])
    # ax.set_yticks([-0.1, 0, 0.1])
    # ax.set_zticks([100, 120, 140])
    # ax.set_xticklabels([-0.1, 0, 0.1])
    # ax.set_yticklabels([-0.1, 0, 0.1])
    # ax.set_zticklabels([1800, 1500, 1300])
    # 调整刻度位置
    ax.tick_params(axis='x', pad=1, length=15)
    ax.tick_params(axis='y', pad=1, length=15)
    ax.tick_params(axis='z', pad=15, length=15)
    ax.set_box_aspect([1, 1, 2])  # x, y, z 轴的比例
    # ax.view_init(elev=30, azim=-60+90+75)
    ax.view_init(elev=30, azim=-60+90+120)

    # # 添加颜色条
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # 空数组用于生成颜色条
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label('Frequency (THz)')

    plt.savefig(f'./rsl/3D-{cmap.name}-{"EP_band"}.png', dpi=500, bbox_inches='tight', pad_inches=0.3, transparent=True)
    plt.savefig(f'./rsl/3D-{cmap.name}-{"EP_band"}.svg', bbox_inches='tight', pad_inches=0.3, transparent=True)
    # plt.show()

# 载入数据并调用函数
data = pd.read_csv('./data/EP_band.csv', sep='\t').to_numpy()
# data = pd.read_csv('expanded_VBG-final_design.csv', sep='\t').to_numpy()
plot_band_surfaces(data)
