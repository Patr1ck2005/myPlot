import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初始化网格大小和时间步
Nx, Ny = 257, 257         # 网格大小
Nt = 30000                  # 时间步数
dx = 1.0                  # 空间步长
dy = 1.0

# 光速（自由空间），自由空间介电常数和磁导率
c = 3e8
epsilon_0 = 8.854e-12     # 自由空间介电常数
mu_0 = 4 * np.pi * 1e-7   # 自由空间磁导率

# 根据柯朗条件计算时间步长
dt = 0.5 * 1 / (c * np.sqrt(1 / dx**2 + 1 / dy**2))  # 稳定性因子

# 定义电场和磁场，以及介电常数（epsilon）、磁导率（mu）和导电率（sigma）
Ez = np.zeros((Nx, Ny))   # z方向电场
Hx = np.zeros((Nx, Ny))   # x方向磁场
Hy = np.zeros((Nx, Ny))   # y方向磁场

epsilon = np.ones((Nx, Ny)) * epsilon_0  # 默认自由空间值
mu = np.ones((Nx, Ny)) * mu_0            # 默认自由空间值
sigma = np.zeros((Nx, Ny))               # 默认自由空间无损耗

# 设置介质区域
sigma[120:150, 20:100] = 0e-3
epsilon[120:150, 20:100] = 4*epsilon_0
mu[120:150, 20:100] = 1*mu_0  # 磁导率

# 初始化比例系数
C1 = (1 - sigma * dt / (2 * epsilon)) / (1 + sigma * dt / (2 * epsilon))
C2 = dt / epsilon / (1 + sigma * dt / (2 * epsilon))
Db = dt / mu  # 磁场比例系数

# 高斯脉冲参数（激励源）
x0, y0 = Nx // 2, Ny // 2  # 在网格中心注入高斯脉冲
spread = 20                # 脉冲宽度

def update_fields(n):
    # 更新磁场
   Hx[:, :-1] = Hx[:, :-1] - Db[:, :-1] * (Ez[:, 1:] - Ez[:, :-1]) / dy
   Hy[:-1, :] = Hy[:-1, :] + Db[:-1, :] * (Ez[1:, :] - Ez[:-1, :]) / dx

   # 更新电场（使用金属修正系数）
   Ez[1:, 1:] = C1[1:, 1:] * Ez[1:, 1:] + \
                C2[1:, 1:] * (
                        (Hy[1:, 1:] - Hy[:-1, 1:]) / dx -
                        (Hx[1:, 1:] - Hx[1:, :-1]) / dy
                )

   # 注入源
   Ez[x0, y0] += 20*np.exp(- ((n - 30) / spread) ** 2)

# 动态可视化
fig, ax = plt.subplots()
img = ax.imshow(Ez, cmap='RdBu', interpolation='bilinear', extent=[0, Nx * dx, 0, Ny * dy], animated=True,
                vmin=-1, vmax=1)

def animate(n):
    update_fields(n)
    img.set_array(Ez)
    return img,

ani = FuncAnimation(fig, animate, frames=60, interval=20, blit=True)
plt.colorbar(img, ax=ax, label="H_z Field (A/m)")
plt.title("2D FDTD Simulation (H_z Component)")
plt.show()
