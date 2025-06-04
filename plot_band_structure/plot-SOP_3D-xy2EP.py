import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 数据
# df = pd.read_csv('data/xy2EP/xy2EP-BIC-test.csv', sep='\t')
df = pd.read_csv('data/xy2EP/xy2EP-QBIC-test.csv', sep='\t')

df = df[(df['品质因子 (1)'] > 20) & (df['频率 (THz)'] > 120)]

# 准备绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取频率并归一化
frequencies = [complex(row['特征频率 (THz)'].replace('i', 'j')).real for _, row in df.iterrows()]  # 提取频率 (仅实部用于颜色映射)
freq_min, freq_max = min(frequencies), max(frequencies)

# 创建颜色映射对象
norm = plt.Normalize(vmin=freq_min, vmax=freq_max)
cmap = plt.get_cmap('twilight')

color_by_rank = False

for _, row in df.iterrows():
    # m1, m2, freq, tanchi, phi, rank = d
    # m1, m2, freq, tanchi, phi, Q, S_air_prop, rank = d
    m1 = row['m1']
    m2 = row['m2']
    freq = row['特征频率 (THz)']
    tanchi = row['tanchi (1)']
    phi = row['phi (rad)']
    Q = row['品质因子 (1)']
    # S_air_prop = row['S_air_prop']
    # rank = row['rank']

    freq_re = complex(freq.replace('i', 'j')).real
    freq_im = complex(freq.replace('i', 'j')).imag

    # 计算椭圆的长轴和短轴
    major_axis = freq_im/1000+0.001
    minor_axis = major_axis * np.tan(tanchi)

    # 椭圆的角度
    theta = np.linspace(0, 2 * np.pi, 128)

    # 椭圆参数方程
    x = major_axis * np.cos(theta)
    y = minor_axis * np.sin(theta)

    # 旋转椭圆
    rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)],
                                [np.sin(phi), np.cos(phi)]])
    ellipse = np.dot(rotation_matrix, np.array([x, y]))

    color = cmap(norm(freq_re))
    alpha = 1
    # 绘制椭圆
    ax.plot(m1 + ellipse[0], m2 + ellipse[1], zs=freq_re, color=color, linewidth=1, alpha=alpha)


# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 空数组用于生成颜色条
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('Frequency (THz)')

# 设置图形格式
ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.legend(loc='upper right')
ax.set_box_aspect([1, 1, 2])
ax.set_title('polarization map')
ax.view_init(elev=30, azim=-60-90)

plt.tight_layout()
plt.savefig('./rsl/3D_SOP_fig.png', dpi=300, bbox_inches='tight', pad_inches=0.3, transparent=True)
plt.show()
