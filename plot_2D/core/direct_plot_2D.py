import json
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import compute_circular_average, load_img

def direct_plot_2D(
        twoD_data: np.ndarray,
        save_name: str,
        plot_paras: dict = None,
        show: bool = False,
        stretch: int = 1,
        **kwargs
):
    # 原始数组
    image_array = twoD_data
    # 获取数组的行数和列数（即高度和宽度，单位：像素）
    height, width = image_array.shape

    # 设定 DPI 值，并计算图像尺寸（英寸）
    dpi = 100
    figsize = (width / dpi, height / dpi)

    # 创建 figure，并设置尺寸和 DPI，使保存的图片与原始数组大小一致
    plt.figure(figsize=figsize, dpi=dpi * stretch)
    plt.imshow(image_array, cmap=plot_paras['colormap'], aspect='auto',
              origin='lower', interpolation='none', **kwargs)
    plt.axis('off')  # 关闭坐标轴显示
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 将参数字典转换为字符串，方便作为文件名的一部分
    dict_str = json.dumps(plot_paras, separators=(',', ':'))
    dict_str = dict_str.replace('{', '').replace('}', '').replace('"', '').replace(' ', '_').replace(':', '_')

    # 保存图像，确保输出分辨率与原始数组一致
    plt.savefig(f'./rsl/{save_name}+{dict_str}.png', bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()