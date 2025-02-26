import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 数据
# data = pd.read_csv('sorted_VBG-II-shrink_type-1stFP', sep='\t').to_numpy()
# data = pd.read_csv('sorted_VBG-II-shrink_type-1stFP-meshed', sep='\t').to_numpy()
# data = pd.read_csv('sorted_VBG_III_s1mple', sep='\t').to_numpy()
# data = pd.read_csv('sorted_VBG-II-shrink_type-1stFP-uncoupled-meshed', sep='\t').to_numpy()
# data = pd.read_csv('sorted_VBG-band3D-homo_layer.csv', sep='\t').to_numpy()
# data = pd.read_csv('data/sorted_VBG-band3D-final_design.csv', sep='\t').to_numpy()
# data = pd.read_csv('data/sorted_merging_BICs-band3D.csv', sep='\t').to_numpy()
data = pd.read_csv('data/expanded_merging_BICs.csv', sep='\t').to_numpy()

color_by_rank = False

#### 准备绘图 ################################################################################################
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 提取频率并归一化
frequencies = [complex(d[2].replace('i', 'j')).real for d in data]  # 提取频率 (仅实部用于颜色映射)
freq_min, freq_max = min(frequencies), max(frequencies)

# Qs = [d[5] for d in data]
log_Qs = [np.log(d[5]) for d in data]
log_Q_min, log_Q_max = min(log_Qs), max(log_Qs)

# 创建颜色映射对象
freq_norm = plt.Normalize(vmin=freq_min, vmax=freq_max)
log_Q_norm = plt.Normalize(vmin=log_Q_min+5, vmax=log_Q_max)
polar_cmap = plt.get_cmap('twilight')
# polar_cmap = plt.get_cmap('inferno_r')
scatter_cmap = plt.get_cmap('hot')

for d in data:
    # m1, m2, freq, tanchi, phi, rank = d
    m1, m2, freq, tanchi, phi, Q, S_air_prop, rank = d
    # m1, m2: momentum space coordinate
    # freq: complex frequency (THz)
    # tanchi, phi: polarization parameter (elliptical polarized)
    # rank: rank of mode frequency

    freq_re = complex(freq.replace('i', 'j')).real
    freq_im = complex(freq.replace('i', 'j')).imag

    if Q < 7:
        print('Q skip')
        continue
    elif freq_re > 115:
        print('f skip')
        continue
    # elif Q > 20:
    #     print('Q skip')
    #     continue
    elif S_air_prop > 10:
        print('S_air_prop skip')

    # 计算椭圆的长轴和短轴
    # major_axis = freq_im/1000+0.0001
    major_axis = 0.0005
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

    polar_color = polar_cmap(freq_norm(freq_re))
    scatter_color = scatter_cmap(log_Q_norm(np.log(Q)))
    # alpha = np.pow(np.clip(100 / Q, 0, 1), 1/2)  # 透明度由虚部控制
    alpha = 1  # 透明度由虚部控制
    # color = [c*np.clip(100 / Q, 0, 1) for c in color[:3]]

    if color_by_rank:
        if rank == 1:
            polar_color = 'gold'
            alpha = 0.1
        elif rank == 2:
            polar_color = 'green'
            alpha = 0.1
        elif rank == 3:  # imp
            polar_color = 'blue'
            alpha = 0.9
        elif rank == 4:  # imp
            polar_color = 'red'
            alpha = 0.9
        elif rank == 5:
            polar_color = 'purple'
            alpha = 0.1
        elif rank == 6:
            polar_color = 'orange'
            alpha = 0.1
    # 绘制中心点
    ax.scatter(m1, m2, freq_re, color=scatter_color, s=1, alpha=1)
    # 绘制椭圆
    # ax.plot(m1 + ellipse[0], m2 + ellipse[1], zs=freq_re, color=polar_color, linewidth=1, alpha=alpha)

# 添加颜色条
# sm = plt.cm.ScalarMappable(cmap=polar_cmap, norm=freq_norm)
sm = plt.cm.ScalarMappable(cmap=scatter_cmap, norm=log_Q_norm)
# sm.set_array([])  # 空数组用于生成颜色条
cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('Frequency (THz)')

# 设置图形格式
ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_box_aspect([1, 1, 1])  # x, y, z 轴的比例
# ax.legend(loc='upper right')

# elev = 30
# azim = -60
# ax.view_init(elev=elev, azim=azim)
# plt.tight_layout()
# plt.savefig(f'./rsl/bandSOP3D-{polar_cmap.name}-{"final_design"}-{elev}-{azim}.png', dpi=500, bbox_inches='tight', pad_inches=0.3, transparent=True)
#
# elev = 30
# azim = -60+45
# ax.view_init(elev=elev, azim=azim)
# plt.tight_layout()
# plt.savefig(f'./rsl/bandSOP3D-{polar_cmap.name}-{"final_design"}-{elev}-{azim}.png', dpi=500, bbox_inches='tight', pad_inches=0.3, transparent=True)
#
# elev = 45
# azim = -60
# ax.view_init(elev=elev, azim=azim)
# plt.tight_layout()
# plt.savefig(f'./rsl/bandSOP3D-{polar_cmap.name}-{"final_design"}-{elev}-{azim}.png', dpi=500, bbox_inches='tight', pad_inches=0.3, transparent=True)
#
# elev = 0
# azim = -60
# ax.view_init(elev=elev, azim=azim)
# plt.tight_layout()
# plt.savefig(f'./rsl/bandSOP3D-{polar_cmap.name}-{"final_design"}-{elev}-{azim}.png', dpi=500, bbox_inches='tight', pad_inches=0.3, transparent=True)


elev = -75
azim = -60
ax.view_init(elev=elev, azim=azim)
plt.tight_layout()
plt.savefig(f'./rsl/bandSOP3D-{polar_cmap.name}-{"final_design"}-{elev}-{azim}.png', dpi=500, bbox_inches='tight', pad_inches=0.3, transparent=True)
