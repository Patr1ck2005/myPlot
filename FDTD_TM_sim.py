import numpy as np
import matplotlib.pyplot as plt

# ========== 物理常数 ==========
epsilon0 = 8.85e-12  # 真空介电常数
mu0 = np.pi * 4e-7  # 真空磁导率 (关键修复!)
c = 1 / np.sqrt(epsilon0 * mu0)  # 光速计算

# ========== 仿真参数 ==========
nx, ny = 200, 200  # 网格尺寸
dx = dy = 5e-9  # 空间步长 (5nm)
dt = dx / (c * np.sqrt(2))  # 严格CFL条件 (关键修复!)

# ========== 材料参数 ==========
metal_epsilon = -12.0  # 调整金属介电常数 (关键修复!)
air_epsilon = 3.0  # 介质材料更稳定

# ========== 场初始化 ==========
Ez = np.zeros((nx, ny))
Hx = np.zeros((nx, ny))
Hy = np.zeros((nx, ny))

# ========== 金属板设置 ==========
metal_thickness = 8
metal_start = ny // 2 - metal_thickness
metal_end = ny // 2 + metal_thickness
epsilon = air_epsilon * np.ones((nx, ny))
epsilon[:, metal_start:metal_end] = metal_epsilon

# ========== 更新系数 ==========
Cez = dt / (epsilon * epsilon0)  # 电场更新系数
Ch = dt / (mu0 * dx)  # 磁场更新系数

# ========== 激励源参数 ==========
source_pos = (nx // 2, ny // 2 + 20)  # 距金属表面20网格
t0, sigma = 40, 12  # 脉冲参数

# ========== 主循环 ==========
for t in range(6000):
    # === 磁场更新 (修正符号!) ===
    Hx[:, 1:-1] += Ch * (Ez[:, 2:] - Ez[:, 1:-1])  # 符号修复
    Hy[1:-1, :] -= Ch * (Ez[2:, :] - Ez[1:-1, :])  # 符号修复

    # === 电场更新 ===
    Ez[1:-1, 1:-1] += Cez[1:-1, 1:-1] * (
            (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) -
            (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]))

    # === 激励源注入 ===
    pulse = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    Ez[source_pos[0], source_pos[1]] += 2 * pulse

    # === 可视化 ===
    if t % 100 == 0:
        plt.imshow(Ez.T, cmap='seismic', vmin=-0.05, vmax=0.05)
        plt.title(f"SPP Propagation (t={t})")
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.001)
        plt.clf()
