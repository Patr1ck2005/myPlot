import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

plt.rcParams['font.size'] = 12
xlim = (1480, 1580)
xlabel = r'Wavelength (nm)'
ylabel = r'NA'
zlabel = r'Efficiency'

filename = 'optical_intensity_results-old-nonp2p.csv'
# 切片标签
df = pd.read_csv(f'./data/{filename}')
dataset = {}
slice_positions = [0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1]
for slice_value in slice_positions:
    # 对每个，提取 'wavelength_nm' 和 'average_intensity' 列，作为列表
    wavelength_array = df['wavelength_nm'].array
    intensity_array = df[f'avg_intensity_NA{slice_value}'].array
    # 将每个目录对应的波长和强度列表存入字典，键为 'wavelength' 和 'intensity'
    dataset[slice_value] = {
        'x': wavelength_array[wavelength_array < xlim[1]],
        'z': intensity_array[wavelength_array < xlim[1]],
        'max_point': (wavelength_array[intensity_array.argmax()], intensity_array.max())
    }
# slice_positions = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


# 遍历每个角度的光谱数据，绘制填充区域
verts = []
for i, slice in enumerate(slice_positions):
    x = dataset[slice]['x']  # x 数据
    y = dataset[slice]['z']  # 对应的 z 数据
    print(max(y))
    verts.append(polygon_under_graph(x, y))

# 创建3D图形
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 为每个多边形设置颜色
facecolors = plt.colormaps['inferno_r'](np.linspace(0, 1, len(verts)))[::-1]

# 绘制多边形集合
poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
ax.add_collection3d(poly, zs=slice_positions, zdir='y')

# 绘制切片位置的直线
for i, slice in enumerate(slice_positions):
    ax.plot([xlim[0], xlim[1]], [slice, slice], [0, 0], color=facecolors[i], linewidth=2)
    # ax.plot([dataset[slice]['max_point'][0]], [slice], [dataset[slice]['max_point'][1]], color='black', marker='o', markersize=5)
    ax.plot(
        [1480], [slice], [dataset[slice]['max_point'][1]],
        color=facecolors[i],
        marker='o',
        markersize=5,
        # markeredgecolor=0.8*facecolors[i],
        markeredgecolor='black',
        markeredgewidth=1,
        zorder=99,
    )
    # draw line connect max_point
    # ax.plot([dataset[slice]['max_point'][0], dataset[slice]['max_point'][0]], [slice, slice], [0, dataset[slice]['max_point'][1]], color='black', linewidth=1, linestyle='--')
    ax.plot([dataset[slice]['max_point'][0], 1480], [slice, slice], [dataset[slice]['max_point'][1], dataset[slice]['max_point'][1]], color=facecolors[i], linewidth=1, linestyle='--')

# 设置绘图样式
ax.grid(True)
ax.set_xlim(1480, 1580)
ax.set_ylim(0, 0.42)
ax.set_zlim(0, 1)
ax.set_xticks([1480, 1500, 1510, 1520, 1530, 1550, 1580])
ax.set_yticks(slice_positions[::2])
ax.set_zticks([0, 0.5, 0.75, 1])
# ax.set_xticklabels([-0.1, 0, 0.1])
ax.set_xticklabels([])
# ax.set_yticklabels(slice_positions[::2])
ax.set_yticklabels([])
ax.set_zticklabels([0, 0.5, 0.75, 1])
# 设置坐标轴标签和范围
# ax.set_xlabel(xlabel, labelpad=10)
# ax.set_ylabel(ylabel, labelpad=10)
# ax.set_zlabel(zlabel, labelpad=10)
# 调整刻度位置
ax.tick_params(axis='x', pad=1)
ax.tick_params(axis='y', pad=1)
ax.tick_params(axis='z', pad=1)
# ax.label_params(axis='x', pad=5)

ax.view_init(elev=30, azim=60)
ax.set_box_aspect([2, 2, 1])  # x, y, z 轴的比例

# 显示图形
plt.tight_layout()
# plt.savefig('../rsl/3D_slices_fig.png', dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True)
plt.savefig('./rsl/3D_slices_fig.png', dpi=300, bbox_inches='tight', pad_inches=0.3, transparent=True)
plt.show()

