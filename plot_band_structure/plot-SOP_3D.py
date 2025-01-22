import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 数据
# data = pd.read_csv('sorted_VBG-II-shrink_type-1stFP', sep='\t').to_numpy()
# data = pd.read_csv('sorted_VBG-II-shrink_type-1stFP-meshed', sep='\t').to_numpy()
# data = pd.read_csv('sorted_VBG_III_s1mple', sep='\t').to_numpy()
# data = pd.read_csv('sorted_VBG-II-shrink_type-1stFP-uncoupled-meshed', sep='\t').to_numpy()
data = pd.read_csv('sorted_VBG-final_design.csv', sep='\t').to_numpy()

# 准备绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取频率并归一化
frequencies = [complex(d[2].replace('i', 'j')).real for d in data]  # 提取频率 (仅实部用于颜色映射)
freq_min, freq_max = min(frequencies), max(frequencies)

# 创建颜色映射对象
norm = plt.Normalize(vmin=freq_min, vmax=freq_max)
cmap = plt.get_cmap('twilight')

for d in data:
    # m1, m2, freq, tanchi, phi, rank = d
    m1, m2, freq, tanchi, phi, _, rank = d
    # m1, m2: momentum space coordinate
    # freq: complex frequency (THz)
    # tanchi, phi: polarization parameter (elliptical polarized)
    # rank: rank of mode frequency

    freq_re = complex(freq.replace('i', 'j')).real
    freq_im = complex(freq.replace('i', 'j')).imag

    # 计算椭圆的长轴和短轴
    major_axis = freq_im/2000+0.0001
    minor_axis = major_axis * np.tan(tanchi)

    # 椭圆的角度
    theta = np.linspace(0, 2 * np.pi, 400)

    # 椭圆参数方程
    x = major_axis * np.cos(theta)
    y = minor_axis * np.sin(theta)

    # 旋转椭圆
    rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)],
                                [np.sin(phi), np.cos(phi)]])
    ellipse = np.dot(rotation_matrix, np.array([x, y]))

    color = cmap(norm(freq_re))
    if rank == 1:
        color = 'gold'
        alpha = 0.1
    elif rank == 2:
        color = 'green'
        alpha = 0.1
    elif rank == 3:  # imp
        color = 'blue'
        alpha = 0.9
    elif rank == 4:  # imp
        color = 'red'
        alpha = 0.9
    elif rank == 5:
        color = 'purple'
        alpha = 0.1
    elif rank == 6:
        color = 'orange'
        alpha = 0.1
    # 绘制椭圆
    ax.plot(m1 + ellipse[0], m2 + ellipse[1], zs=freq_re, color=color, linewidth=3, alpha=alpha)
    # ax.plot(-m1 + ellipse[0], m2 + ellipse[1], zs=freq, color=color)
    # ax.plot(-m1 + ellipse[0], -m2 + ellipse[1], zs=freq, color=color)
    # ax.plot(m1 + ellipse[0], -m2 + ellipse[1], zs=freq, color=color)
    # ax.plot(m2 + ellipse[1], m1 + ellipse[0], zs=freq, color=color)
    # ax.plot(-m2 + ellipse[1], m1 + ellipse[0], zs=freq, color=color)
    # ax.plot(-m2 + ellipse[1], -m1 + ellipse[0], zs=freq, color=color)
    # ax.plot(m2 + ellipse[1], -m1 + ellipse[0], zs=freq, color=color)


# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 空数组用于生成颜色条
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('Frequency (THz)')

# 设置图形格式
ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.legend(loc='upper right')
ax.set_box_aspect([1, 1, 1])
ax.set_title('polarization map')

plt.tight_layout()
plt.show()
