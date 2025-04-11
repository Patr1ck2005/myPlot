import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


def save_cmap_to_txt_and_plot(cmap_name, ncolors=32):
    # 获取颜色映射
    cmap = colormaps.get_cmap(cmap_name)
    filename = f"./{cmap_name}.txt"

    # 保存颜色数据到文件
    with open(filename, 'w') as f:
        f.write("% Continuous\n")
        f.write("% Category: Custom\n")
        f.write("% SortWeight: 10\n")
        for i in range(ncolors):
            r, g, b, _ = cmap(i / (ncolors - 1))
            f.write(f"{r:.8f} {g:.8f} {b:.8f}\n")

    # 生成颜色映射的示意图
    # 创建一个包含 ncolors 个不同颜色的条形图
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))  # 制作二维渐变图

    # 绘制颜色映射示意图
    plt.figure(figsize=(8, 2))
    plt.imshow(gradient, aspect='auto', cmap=cmap)
    plt.axis('off')  # 不显示坐标轴

    # 保存为 PNG 文件
    plot_filename = f"./{cmap_name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()


# 保存 coolwarm 和 RdBu 的颜色映射数据以及示意图
save_cmap_to_txt_and_plot("coolwarm")
save_cmap_to_txt_and_plot("RdBu")
