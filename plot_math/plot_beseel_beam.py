import numpy as np
from scipy.special import j0  # 零阶贝塞尔函数
import matplotlib.pyplot as plt

# 参数定义
alpha = 5.0  # 横向波矢参数
grid_size = 2024  # 网格大小
x = np.linspace(-20, 20, grid_size)  # x坐标范围
y = np.linspace(-20, 20, grid_size)  # y坐标范围
X, Y = np.meshgrid(x, y)  # 创建2D网格
r = np.sqrt(X**2 + Y**2)  # 计算径向距离

# 计算实空间场强和强度
field_real = j0(alpha * r)  # 零阶贝塞尔光束场
intensity_real = np.abs(field_real)**2  # 强度 = |场|^2

# 可视化实空间强度分布（无坐标轴）
plt.figure(figsize=(8, 6))
plt.imshow(intensity_real, extent=[-10, 10, -10, 10], cmap='Reds', origin='lower')
plt.colorbar()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axis('off')  # 关闭所有轴和标注
plt.savefig('real_space.png', bbox_inches='tight', pad_inches=0, dpi=300)  # 保存PNG
plt.close()  # 关闭图像，避免显示

# 计算动量空间（通过2D FFT）
field_fft = np.fft.fft2(field_real)
field_fft_shifted = np.fft.fftshift(field_fft)  # 移到中心
intensity_momentum = np.abs(field_fft_shifted)**2  # 强度

# 计算k空间坐标
kx = np.fft.fftfreq(grid_size, d=(x[1] - x[0]))  # kx频率
ky = np.fft.fftfreq(grid_size, d=(y[1] - y[0]))  # ky频率
kx_shifted = np.fft.fftshift(kx)
ky_shifted = np.fft.fftshift(ky)

# 可视化动量空间强度分布（无坐标轴）
plt.figure(figsize=(8, 6))
plt.imshow(intensity_momentum, extent=[kx_shifted[0], kx_shifted[-1], ky_shifted[0], ky_shifted[-1]],
           cmap='Blues', origin='lower', vmax=5e8)
plt.colorbar()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axis('off')  # 关闭所有轴和标注
plt.savefig('momentum_space.png', bbox_inches='tight', pad_inches=0, dpi=300)  # 保存PNG
plt.close()  # 关闭图像，避免显示
