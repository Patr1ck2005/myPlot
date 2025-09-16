import pickle

from matplotlib import pyplot as plt

from plot_3D.core.plot_3D_params_space_plt import *

data_path = r'D:\DELL\Documents\myPlots\plot_3D\projects\SE/rsl/delta_space\20250916_170739\plot_data__x-频率Hz_y-w_delta_factor.pkl'

# 2. 手动读取处理（展开代码）
with open(data_path, 'rb') as f:
    plot_data = pickle.load(f)
# ... 这里手动处理：打印、验证、修改 ...
x_vals = plot_data['x_vals']
y_vals = plot_data['y_vals'][::4]
subs = plot_data['subs']
# ... 自定义逻辑 ...

x_key = plot_data['metadata']['x_key'],
y_key = plot_data['metadata']['y_key'],
figsize = (4, 5)
save_dir = './rsl'
fixed_params = {}
show = True

Z1 = subs[0][:, ::4]
Z2 = subs[1][:, ::4]

"""
阶段3: 从已加载数据集成绘制图像。
"""

# Step 1: 调用现有绘图核心 (从历史plot_Z复制)
fig, ax = plt.subplots(figsize=figsize)

plot_params = {
    'add_colorbar': False, 'cmap': 'magma',
    'default_color': 'gray', 'alpha': 0.5,
    'title': False,
}
fig, ax = plot_2d_multiline(ax, x_vals, y_vals, Z2, plot_params)

plot_params = {
    'add_colorbar': False, 'cmap': 'magma',
    'title': False,
}
fig, ax = plot_2d_multiline(ax, x_vals, y_vals, Z1, plot_params)

# Step 2: 添加注解 (直接调用现有)
annotations = {
    'xlabel': r"f (c/P)", 'ylabel': "P", 'zlabel': "$\delta",
    'target_log_scale': False,
    'xlim': (0.430, 0.440), 'ylim': (0, 1.15e11),
}

fig, ax = add_annotations(ax, annotations)

plt.tight_layout()

# Step 3: 保存图像 (从历史复制)
full_params = {**plot_params}
image_path = generate_save_name(save_dir, full_params)
plt.savefig(image_path, dpi=300, bbox_inches="tight", transparent=True)
print(f"图像已保存为：{image_path} 🎨")

if show:
    plt.show()
