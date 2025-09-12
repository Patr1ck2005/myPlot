import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

# 假设数据
x_vals = np.linspace(0, 10, 100)
y_vals = np.exp(1j * x_vals)
real_vals = y_vals.real
imag_vals = y_vals.imag

# 归一化虚部值以映射到颜色
norm = plt.Normalize(imag_vals.min(), imag_vals.max())
cmap = cm.viridis  # 使用 viridis 颜色映射

# 创建线段集合
points = np.array([x_vals, real_vals]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, array=imag_vals, cmap=cmap, norm=norm, linewidth=2)

# 绘制图形
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.set_xlim(x_vals.min(), x_vals.max())
ax.set_ylim(real_vals.min(), real_vals.max())
ax.set_xlabel('x')
ax.set_ylabel('Real Part')
ax.set_title('Real Part with Color by Imaginary Part')

# 添加颜色条
cbar = plt.colorbar(lc)
cbar.set_label('Imaginary Part')
plt.grid(True)
plt.show()
