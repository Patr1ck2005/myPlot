import numpy as np
import matplotlib.pyplot as plt

from utils.utils import clear_ax


def plot_2d_function(f, x_range, y_range, filename='function_plot.png', **kwargs):
    """
    绘制二维函数 f(x, y) 的颜色映射图并保存为图像文件。

    :param f: 二维函数，接受 x 和 y 值并返回对应的 z 值。
    :param x_range: x 轴的范围 (min, max)。
    :param y_range: y 轴的范围 (min, max)。
    :param filename: 保存的文件名 (默认 'function_plot.png')。
    """
    # 创建网格
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)

    # 计算函数值
    Z = f(X, Y)

    # 创建图像和轴对象
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制热图
    im = ax.imshow(Z, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower', **kwargs)
    # cbar = fig.colorbar(im, ax=ax, label='f(x, y)')  # 添加颜色条
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Function Heatmap')

    clear_ax(ax)

    # 保存图像
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    # 示例: 绘制函数 f(x, y) = sin(x) * cos(y)
    def sample_function(x, y):
        return np.sin(x) * np.cos(y)

    # 使用示例函数并保存图像
    plot_2d_function(sample_function, x_range=(-5, 5), y_range=(-5, 5), filename='./rsl/function_plot.png')
