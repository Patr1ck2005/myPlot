import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.utils import clear_ax_ticks

# 数据
# data = pd.read_csv('expanded_VBG-final_design-old.csv', sep='\t').to_numpy()
# data = pd.read_csv('expanded_VBG-comparison_design-0.12.csv', sep='\t').to_numpy()
# data = pd.read_csv('./data/expanded_VBG-final_design.csv', sep='\t').to_numpy()
# data = pd.read_csv('sorted_VBG-band3D-final_design.csv', sep='\t').to_numpy()
# data = pd.read_csv('./data/expanded_merging_BICs.csv', sep='\t').to_numpy()
data = pd.read_csv('./data/expanded_merging_BICs.csv', sep='\t').to_numpy()
# data = pd.read_csv('./data/temp-expanded_SOP.csv', sep='\t').to_numpy()

# # 利用rank选择
# selected_rank = 3
# rank_idx = 7
# selected_data = [d for d in data if int(d[rank_idx]) == selected_rank]  # rank在第几列
# # 利用频率选择
# selected_data = [d for d in data if 115 < complex(d[2].replace('i', 'j')).real < 119]
selected_data = [d for d in data if 100 < complex(d[2].replace('i', 'j')).real < 116 and d[5] > 7]
# selected_data = [eigen_info for eigen_info in data if 121 < complex(eigen_info[2].replace('i', 'j')).real < 128]

# 按照每一个m1, m2下对频率进行排序
# 假设 m1, m2 分别是 d[0] 和 d[1]
grouped_data = {}
selected_TC = None
# selected_TC = -1
for eigen_info in selected_data:
    m1, m2 = eigen_info[0], eigen_info[1]
    if (m1, m2) not in grouped_data:
        grouped_data[(m1, m2)] = []

    frequency, phi = complex(eigen_info[2].replace('i', 'j')).real, eigen_info[4]
    phi = phi % (2*np.pi)
    theta = np.arctan2(m2, m1) % (2*np.pi)

    if selected_TC is None:
        grouped_data[(m1, m2)].append(frequency)
        continue
    elif selected_TC == 1:
        delta_angle = theta-phi
    elif selected_TC == -1:
        delta_angle = theta-phi-np.pi/2
    if m1 == 0.02 and m2 == 0:
        pass  # DEBUG
    if abs(delta_angle) < 0.4 or abs(delta_angle-np.pi) < 0.4 or abs(delta_angle+np.pi) < 0.4 or abs(delta_angle+2*np.pi) < 0.4:
        grouped_data[(m1, m2)].append(frequency)

# 对每一个 m1, m2 下的频率进行大小排序
for key in grouped_data:
    grouped_data[key] = sorted(grouped_data[key])

# 提取每一个m1, m2下 target_frequencies
target_frequencies = {}
for key in grouped_data:
    # target_frequencies[key] = max(sorted_data[key])
    print(len(grouped_data[key]))
    # select upper (max) or lower (min) band ###########################################################################
    target_frequencies[key] = min(grouped_data[key]) if len(grouped_data[key]) > 0 else 0

re_selected_data = [d for d in data if complex(d[2].replace('i', 'j')).real == target_frequencies.get((d[0], d[1]), (0, 0))]  # rank在第几列
# replace the data
selected_data = re_selected_data

# 提取频率并归一化
frequencies = [complex(d[2].replace('i', 'j')).real for d in data]  # 提取频率 (仅实部用于颜色映射)
cmap_freq_min, cmap_freq_max = 100, 118

# Qs = [d[5] for d in data]
log_Qs = [np.log(d[5]) for d in data]
log_Q_min, log_Q_max = min(log_Qs), max(log_Qs)

# 创建颜色映射对象
freq_norm = plt.Normalize(vmin=cmap_freq_min, vmax=cmap_freq_max)
log_Q_norm = plt.Normalize(vmin=log_Q_min+5, vmax=log_Q_max)
phi_norm = plt.Normalize(vmin=0, vmax=np.pi)
# cmap = plt.get_cmap('twilight')
freq_cmap = plt.get_cmap('inferno_r')
scatter_cmap = plt.get_cmap('hot')
phi_cmap = plt.get_cmap('hsv')

## 创建2D图 ########################################################
# 提取所有的m1, m2的范围
m1_values = [d[0] for d in selected_data]
m2_values = [d[1] for d in selected_data]

m1_min, m1_max = min(m1_values), max(m1_values)
m2_min, m2_max = min(m2_values), max(m2_values)

# 创建2D图
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))

# 频率热图矩阵初始化
heatmap_size = 60*2
# heatmap_size = 32*2
frequency_matrix = np.zeros((heatmap_size, heatmap_size))  # 假设的大小，根据您的数据可以调整
Q_matrix = np.zeros((heatmap_size, heatmap_size))  # 假设的大小，根据您的数据可以调整
phi_matrix = np.zeros((heatmap_size, heatmap_size))  # 假设的大小，根据您的数据可以调整

# 绘制每个点（使用频率作为颜色映射）
for d in selected_data:
    m1, m2, freq, tanchi, phi, Q, S_air_prop, rank = d
    phi %= np.pi
    if Q < 7:
        print('Q skip')
        continue
    # elif Q > 20:
    #     print('Q skip')
    #     continue
    elif S_air_prop > 10:
        print('S_air_prop skip')
    freq_re = complex(freq.replace('i', 'j')).real

    # 计算椭圆的长轴和短轴
    major_axis = 0.0002
    minor_axis = major_axis * np.tan(float(tanchi))

    # 绘制椭圆（2D）
    theta = np.linspace(0, 2 * np.pi, 400)
    x = major_axis * np.cos(theta)
    y = minor_axis * np.sin(theta)

    # 旋转椭圆
    rotation_matrix = np.array([[np.cos(float(phi)), -np.sin(float(phi))],
                                [np.sin(float(phi)), np.cos(float(phi))]])
    ellipse = np.dot(rotation_matrix, np.array([x, y]))

    # 颜色映射
    polar_color = freq_cmap(freq_norm(freq_re))
    phi_color = phi_cmap(phi_norm(float(phi)))
    scatter_color = scatter_cmap(log_Q_norm(np.log(Q)))
    # # 绘制中心点
    # ax1.scatter(m1, m2, color=scatter_color, s=5, alpha=1)
    # 绘制椭圆
    ax1.plot(m1 + ellipse[0], m2 + ellipse[1], color=phi_color, linewidth=3)

    # 填充热图矩阵（用频率值填充）
    # 将m1和m2映射到热图的索引
    x_idx = int((m1 - m1_min) / (m1_max - m1_min) * (heatmap_size - 1))
    y_idx = int((m2 - m2_min) / (m2_max - m2_min) * (heatmap_size - 1))
    frequency_matrix[x_idx, y_idx] = freq_re
    Q_matrix[x_idx, y_idx] = Q
    phi_matrix[x_idx, y_idx] = phi%(np.pi)

k_range = 0.06
# k_range = 2*0.01
plot_k_range = 0.01
ax1.set_xlim(-plot_k_range, plot_k_range)
ax1.set_ylim(-plot_k_range, plot_k_range)
clear_ax_ticks(ax1)
plt.tight_layout()
plt.savefig(f'./rsl/SOP_2D-polar-TC={selected_TC}.png', bbox_inches='tight', pad_inches=0.0, dpi=300, transparent=True)

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
# 绘制热图
ax2.imshow(frequency_matrix, cmap=freq_cmap, origin='lower', aspect='equal', vmin=cmap_freq_min, vmax=cmap_freq_max)
# cax = ax2.imshow(Q_matrix, cmap='hot', origin='lower', aspect='equal')
clear_ax_ticks(ax2)
plt.tight_layout()
plt.savefig(f'./rsl/SOP_2D-freq-TC={selected_TC}.png', bbox_inches='tight', pad_inches=0.0, dpi=300)

fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))
# 绘制热图
cax3 = ax3.imshow(
    np.log10(Q_matrix),
    cmap='hot',
    origin='lower',
    aspect='equal',
    extent=[-k_range, k_range, -k_range, k_range],
    vmin=2,
)
ax3.set_xlim(-k_range, k_range)
ax3.set_ylim(-k_range, k_range)
# clear_ax(ax3)
plt.colorbar(cax3)
plt.tight_layout()
plt.savefig(f'./rsl/SOP_2D-Q-TC={selected_TC}.png', bbox_inches='tight', pad_inches=0.0, dpi=300)

fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8))
# 绘制热图
ax4.imshow(phi_matrix, cmap='hsv', origin='lower', aspect='equal')
clear_ax_ticks(ax4)
plt.tight_layout()
plt.savefig(f'./rsl/SOP_2D-phi-TC={selected_TC}.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
plt.show()