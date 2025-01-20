import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def plot_band_surfaces(data):
    # 准备绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取频率并归一化
    frequencies = [complex(d[2].replace('i', 'j')) for d in data]  # 提取复数频率
    freq_real = [freq.real for freq in frequencies]  # 实部用于颜色映射
    freq_imag = [freq.imag for freq in frequencies]  # 虚部用于透明度

    freq_min, freq_max = min(freq_real), max(freq_real)

    # 创建颜色映射对象
    norm = plt.Normalize(vmin=freq_min, vmax=freq_max)
    cmap = plt.get_cmap('twilight')

    # 选择坐标和频率数据进行绘制
    for i, d in enumerate(data):
        m1, m2, freq, tanchi, phi, Q, rank = d
        if Q < 30:
            print('skip')
            continue
        freq_complex = complex(freq.replace('i', 'j'))
        freq_re = freq_complex.real
        freq_im = freq_complex.imag

        # 频率的实部和虚部影响颜色和透明度
        color = cmap(norm(freq_re))
        alpha = np.clip(30/Q, 0, 1)  # 透明度由虚部控制

        # 绘制曲面
        ax.scatter(m1, m2, freq_re, color=color, alpha=alpha, s=10)  # 使用散点绘图

    # 设置图形格式
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Frequency (THz)')
    ax.set_title('Band Structure Visualization')
    ax.set_box_aspect([1, 1, 2])  # x, y, z 轴的比例

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 空数组用于生成颜色条
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Frequency (THz)')

    plt.tight_layout()
    plt.show()

# 载入数据并调用函数
# data = pd.read_csv('sorted_VBG-final_design.csv', sep='\t').to_numpy()
data = pd.read_csv('expanded_VBG-final_design.csv', sep='\t').to_numpy()
plot_band_surfaces(data)
