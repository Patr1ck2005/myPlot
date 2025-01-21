import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

plt.rcParams['font.size'] = 24
xlim = (1480, 1600)
xlabel = r'Wavelength (nm)'
ylabel = r'NA'
zlabel = 'Efficiency'

# 切片标签
df = pd.read_csv('../data/optical_intensity_results.csv')
# 使用 groupby 按照 'dir' 分组
grouped = df.groupby('dir')
dataset = {}
for dir_name, group in grouped:
    # 对每个目录，提取 'wavelength_nm' 和 'average_intensity' 列，作为列表
    wavelength_array = group['wavelength_nm'].array
    intensity_array = group['average_intensity'].array
    # 将每个目录对应的波长和强度列表存入字典，键为 'wavelength' 和 'intensity'
    dataset[dir_name] = {
        'x': wavelength_array[wavelength_array < xlim[1]],
        'z': intensity_array[wavelength_array < xlim[1]]
    }
# slice_positions = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
slice_positions = np.array(list(dataset.keys()))

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
    y = dataset[slice]['z']  # 对应的 y 数据
    verts.append(polygon_under_graph(x, y))

# 创建3D图形
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# 为每个多边形设置颜色
facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))[::-1]

# 绘制多边形集合
poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
ax.add_collection3d(poly, zs=slice_positions, zdir='y')

ax.grid(False)
# 设置坐标轴标签和范围
ax.set_xlabel(xlabel, labelpad=20)
ax.set_ylabel(ylabel, labelpad=20)
ax.set_zlabel(zlabel)
# 调整刻度位置
ax.tick_params(axis='x', pad=5)
ax.tick_params(axis='y', pad=5)
ax.tick_params(axis='z', pad=5)
# ax.label_params(axis='x', pad=5)

# ax.set_ylim(0, 4)  # 角度从0到4度
ax.set_zlim(0.0, 1)
# ax.set_zticks([], [])

# 显示图形
plt.tight_layout()
plt.savefig('3D_slices_fig.png', dpi=300, bbox_inches='tight')
plt.show()

