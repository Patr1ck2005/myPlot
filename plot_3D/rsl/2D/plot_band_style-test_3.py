import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# 你的数据（同上）
x_vals = np.linspace(0, 10, 100)
y_lower = np.sin(x_vals) - 0.5
y_upper = np.sin(x_vals) + 0.5
alpha_fill = 0.3

fig, ax = plt.subplots()

# 创建填充路径（多边形：下边界 + 反转上边界 + 闭合）
verts = np.vstack([np.column_stack([x_vals, y_lower]),
                   np.column_stack([x_vals[::-1], y_upper[::-1]])])
path = Path(verts)
patch = PathPatch(path, facecolor='none')  # 无色patch，只用于剪裁
ax.add_patch(patch)

# 创建渐变图像：水平渐变（基于x）
# 生成一个2D数组，y范围覆盖整个axes，x用linspace渐变
xlim = ax.get_xlim()
ylim = ax.get_ylim()
gradient_data = np.linspace(0, 1, len(x_vals)).reshape(1, -1)  # 行1，列len(x)，值0到1
im = ax.imshow(
    gradient_data,
    cmap='Blues',  # 渐变色图
    aspect='auto',
    extent=[xlim[0], xlim[1], ylim[0], ylim[1]],  # 覆盖axes范围
    alpha=alpha_fill,
    clip_path=patch,  # 剪裁到路径内
    clip_on=True
)

# 绘制曲线
ax.plot(x_vals, (y_upper + y_lower)/2, color='black', label='Center Line')

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.show()
