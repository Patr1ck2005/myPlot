import numpy as np
import matplotlib.pyplot as plt

from utils.utils import clear_ax_ticks


def plot_2d_function(f, x_range, y_range, cmap_name, filename='function_plot.png', **kwargs):
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

    # 标准化 Z 至 [0, 1]，用于 alpha 通道
    norm = plt.Normalize(vmin=np.min(Z), vmax=1)
    norm = plt.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    Z_norm = norm(Z)

    # # 获取颜色映射
    # cmap = plt.get_cmap(cmap_name)

    vortex_pattern = (2*(np.atan(Y / X))+np.pi)/np.pi/2
    # 获取颜色映射
    cmap = plt.get_cmap('magma')
    # cmap = plt.get_cmap('twilight')
    # cmap = plt.get_cmap('hsv')

    # spherical_pattern = np.angle(np.exp(1j * 10*(X**2 + Y**2) + 1j*np.pi))
    # spherical_pattern = (spherical_pattern+np.pi)/np.pi/2

    # 应用颜色映射获得 RGBA 图像
    # rgba_img = cmap(vortex_pattern)
    rgba_img = cmap(Z_norm)

    # 将透明度通道设定为标准化后的函数值（也可以根据需要进行非线性变换）
    rgba_img[..., 3] = Z_norm

    # 叠加黑色背景（RGB = 0），公式为：
    # composite_RGB = alpha * foreground_RGB + (1-alpha) * background_RGB
    # 由于背景为黑色，(1-alpha)*background_RGB=0，故 composite_RGB = alpha * foreground_RGB
    composite_img = np.empty_like(rgba_img)
    composite_img[..., :3] = rgba_img[..., :3] * rgba_img[..., 3:4]  # 保证按像素计算 alpha
    composite_img[..., 3] = 1.0  # 最终图像完全不透明

    # 创建图像和轴对象
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制热图
    im = ax.imshow(rgba_img, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower', **kwargs)
    # cbar = fig.colorbar(im, ax=ax, label='f(x, y)')  # 添加颜色条
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Function Heatmap')

    clear_ax_ticks(ax)

    # 保存图像
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

if __name__ == '__main__':
    # 示例: 绘制函数 f(x, y) = sin(x) * cos(y)
    def sample_function(x, y):
        return np.sin(x) * np.cos(y)

    # 使用示例函数并保存图像
    plot_2d_function(sample_function, x_range=(-5, 5), y_range=(-5, 5), filename='./rsl/function_plot.png')
