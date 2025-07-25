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
    frequencies = [complex(d[2].replace('i', 'j')) for d in data]  # 提取复数频率
    freq_real = [freq.real for freq in frequencies]  # 实部用于颜色映射
    freq_imag = [freq.imag for freq in frequencies]  # 虚部用于透明度

    freq_min, freq_max = 98+4, 136+4
    freq_length = freq_max-freq_min

    # 创建颜色映射对象
    norm = plt.Normalize(vmin=freq_min, vmax=freq_max)
    cmap = plt.get_cmap('twilight')

    # 选择坐标和频率数据进行绘制
    for i, d in enumerate(data):
        m1, m2, freq, tanchi, phi, Q, S_air_prop, rank = d
        if Q < 7:
            print('Q skip')
            continue
        elif S_air_prop > 10:
            print('S_air_prop skip')
        freq_complex = complex(freq.replace('i', 'j'))
        freq_re = freq_complex.real
        freq_im = freq_complex.imag
        # .98~136
        # if freq_re < 98 or freq_re > 136:
        if freq_re < freq_min or freq_re > freq_max:
            print('freq skip')
            continue

        edge = False
        phi = np.arctan2(m2, m1)
        if 0 < phi < 3*np.pi/4:
            print('phi skip')
            continue
        elif phi == 0 or phi == 3*np.pi/4:
            print('phi skip')
            edge = True

        # 频率的实部和虚部影响颜色和透明度
        color = cmap(norm(freq_re))
        alpha = np.clip(30/Q/2, 0, 1)  # 透明度由虚部控制

        # 绘制曲面
        if edge:
            ax.scatter(m1, m2, freq_re, color=color, alpha=alpha, s=10, edgecolors='black', linewidths=2)  # 使用散点绘图
        else:
            ax.scatter(m1, m2, freq_re, color=color, alpha=alpha, s=10)

    # 设置图形格式
    # ax.set_xlabel('kx (2π/P)', labelpad=12)
    # ax.set_ylabel('ky (2π/P)', labelpad=12)
    # ax.set_zlabel('Frequency (THz)', labelpad=30)
    # ax.grid(False)
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_zlim(freq_min-.05*freq_length, freq_max+.1*freq_length)
    ax.set_xticks([-0.1, 0, 0.1])
    ax.set_yticks([-0.1, 0, 0.1])
    ax.set_zticks([100, 120, 140])
    ax.set_xticklabels([-0.1, 0, 0.1])
    ax.set_yticklabels([-0.1, 0, 0.1])
    # ax.set_zticklabels([1800, 1500, 1300])
    ax.set_zticklabels([1800, 1500, 1300])
    # 调整刻度位置
    ax.tick_params(axis='x', pad=1, length=15)
    ax.tick_params(axis='y', pad=1, length=15)
    ax.tick_params(axis='z', pad=15, length=15)
    ax.set_box_aspect([1, 1, 2])  # x, y, z 轴的比例
    ax.view_init(elev=30, azim=-60+90)

    # # 添加颜色条
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # 空数组用于生成颜色条
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label('Frequency (THz)')

    # plt.savefig(f'./rsl/band3D-{cmap.name}-{"comparison_design"}.png', dpi=500, bbox_inches='tight', pad_inches=0.3, transparent=True)
    # plt.savefig(f'./rsl/band3D-{cmap.name}-{"comparison_design"}.svg', bbox_inches='tight', pad_inches=0.3, transparent=True)
    plt.savefig(f'./rsl/band3D-{cmap.name}-{"final_design"}.png', dpi=500, bbox_inches='tight', pad_inches=0.3, transparent=True)
    plt.savefig(f'./rsl/band3D-{cmap.name}-{"final_design"}.svg', bbox_inches='tight', pad_inches=0.3, transparent=True)
    # plt.show()

# 载入数据并调用函数
# data = pd.read_csv('./data/expanded_VBG-comparison_design-0.12.csv', sep='\t').to_numpy()
data = pd.read_csv('./data/expanded_VBG-final_design.csv', sep='\t').to_numpy()
plot_band_surfaces(data)
