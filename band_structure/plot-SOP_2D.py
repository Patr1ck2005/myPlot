import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = pd.read_csv('expanded_VBG-final_design-0.12.csv', sep='\t').to_numpy()
# data = pd.read_csv('expanded_VBG-final_design.csv', sep='\t').to_numpy()
# data = pd.read_csv('sorted_VBG-final_design.csv', sep='\t').to_numpy()

# 选择rank为1的元素（或者您可以选择其他rank）
selected_rank = 3
rank_idx = 7
selected_data = [d for d in data if int(d[rank_idx]) == selected_rank]  # rank在第几列

selected_data = [d for d in data if 112 < complex(d[2].replace('i', 'j')).real < 116]

# 提取频率并归一化
frequencies = [complex(d[2].replace('i', 'j')).real for d in selected_data]  # 提取频率 (仅实部用于颜色映射)
freq_min, freq_max = min(frequencies), max(frequencies)

# 创建颜色映射对象
norm = plt.Normalize(vmin=freq_min, vmax=freq_max)
cmap = plt.get_cmap('twilight')

# 提取所有的m1, m2的范围
m1_values = [d[0] for d in selected_data]
m2_values = [d[1] for d in selected_data]

m1_min, m1_max = min(m1_values), max(m1_values)
m2_min, m2_max = min(m2_values), max(m2_values)

# 创建2D图
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))  # 通过两个子图绘制

# 频率热图矩阵初始化
colormap_size = 48
frequency_matrix = np.zeros((colormap_size, colormap_size))  # 假设的大小，根据您的数据可以调整

# 绘制每个点（使用频率作为颜色映射）
for d in selected_data:
    m1, m2, freq, tanchi, phi, Q, S_air_prop, rank = d
    if Q < 10:
        print('Q skip')
        continue
    elif S_air_prop > 10:
        print('S_air_prop skip')
    freq_re = complex(freq.replace('i', 'j')).real

    # 计算椭圆的长轴和短轴
    major_axis = complex(freq.replace('i', 'j')).imag / 2000 + 0.0010
    minor_axis = major_axis * np.tan(float(tanchi))

    # 绘制椭圆（2D）
    theta = np.linspace(0, 2 * np.pi, 400)
    x = major_axis * np.cos(theta)
    y = minor_axis * np.sin(theta)

    # 旋转椭圆
    rotation_matrix = np.array([[np.cos(float(phi)), -np.sin(float(phi))],
                                [np.sin(float(phi)), np.cos(float(phi))]])
    ellipse = np.dot(rotation_matrix, np.array([x, y]))

    # 频率颜色映射
    color = cmap(norm(freq_re))

    # 绘制椭圆
    ax1.plot(m1 + ellipse[0], m2 + ellipse[1], color=color, linewidth=2)

    # 填充热图矩阵（用频率值填充）
    # 将m1和m2映射到热图的索引
    x_idx = int((m1 - m1_min) / (m1_max - m1_min) * (colormap_size-1))
    y_idx = int((m2 - m2_min) / (m2_max - m2_min) * (colormap_size-1))
    frequency_matrix[x_idx, y_idx] = freq_re

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 空数组用于生成颜色条
cbar = plt.colorbar(sm, ax=ax1)
cbar.set_label('Frequency (THz)')
# 设置图形格式
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_title(f'Polarization Map for Rank {selected_rank}')

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
# 绘制频率热图
cax = ax2.imshow(frequency_matrix, cmap='twilight', origin='lower', aspect='equal', vmin=112)
fig2.colorbar(cax, ax=ax2, label='Frequency (THz)')

ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
ax2.set_title('Frequency Heatmap')

plt.tight_layout()
plt.show()

np.save('../data/SOP_2D-freq', frequency_matrix)