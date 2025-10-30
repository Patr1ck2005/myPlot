import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc  # 确保已安装：pip install colorcet

# 获取所有 Colorcet colormap 名称
cmaps = list(cc.cm.keys())
n_maps = len(cmaps)

# 按照名称排序
cmaps.sort()

# 构造一个 1×256 的渐变条用于示例
gradient = np.linspace(0, 1, 256).reshape(1, -1)

# 创建 n_maps 行 1 列的子图，每行展示一个 colormap
fig, axes = plt.subplots(n_maps, 1, figsize=(6, 0.25 * n_maps))
for ax, name in zip(axes, cmaps):
    ax.imshow(gradient, aspect='auto', cmap=cc.cm[name])
    ax.set_axis_off()
    ax.set_title(name, loc='left', fontsize=8)

plt.tight_layout(h_pad=0.0)
plt.savefig('colorcet_cmaps.png', dpi=300, bbox_inches='tight')
# plt.show()
