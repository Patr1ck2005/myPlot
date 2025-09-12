import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm

# 假设数据（同原代码）
x_vals = np.linspace(0, 10, 100)
y_vals = np.exp(1j * x_vals)
real_vals = y_vals.real
imag_vals = y_vals.imag

# 归一化虚部值以控制填充宽度
scale = 0.5  # 控制填充宽度
widths = np.abs(imag_vals)
widths_norm = (widths - np.min(widths)) / (np.max(widths) - np.min(widths) + 1e-8)  # 归一化到 [0,1]
y_upper = real_vals + scale * widths_norm
y_lower = real_vals - scale * widths_norm

# 创建多边形集合用于动态颜色填充
verts = []
colors = []
for i in range(len(x_vals) - 1):
    # 每个多边形由相邻点的 y_upper 和 y_lower 构成
    verts.append([
        (x_vals[i], y_lower[i]),      # 左下
        (x_vals[i], y_upper[i]),      # 左上
        (x_vals[i+1], y_upper[i+1]),  # 右上
        (x_vals[i+1], y_lower[i+1])   # 右下
    ])
    # 颜色基于虚部值（取中间值以平滑过渡）
    colors.append(imag_vals[i])

# 创建 PolyCollection
norm = plt.Normalize(vmin=np.min(imag_vals), vmax=np.max(imag_vals))  # 虚部值归一化
cmap = cm.RdBu  # 双色调：负值红，正值蓝
poly = PolyCollection(verts, array=colors, cmap=cmap, alpha=0.3)  # 半透明填充

# 绘制图形
fig, ax = plt.subplots()
ax.add_collection(poly)  # 添加填充多边形
ax.plot(x_vals, real_vals, color='blue', linewidth=1, alpha=0.8, label='Real Part Line')  # 实部线

# 设置轴、标题等
ax.set_xlim(x_vals.min(), x_vals.max())
ax.set_ylim(real_vals.min() - scale, real_vals.max() + scale)
ax.set_xlabel('x')
ax.set_ylabel('Real Part')
ax.set_title('Real Part with Dynamic Color Fill by Imaginary Part')
ax.legend()
ax.grid(True)

# 添加颜色条
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array(imag_vals)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Imaginary Part (controls color)')

plt.show()
