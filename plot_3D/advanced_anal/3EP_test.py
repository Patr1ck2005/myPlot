import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# 1. 定[object Object]参数和函数
Ω0, Ω1, Ω2 = 0.0, 0.0, 0.0
γ_bg, γ1, γ3 = 0.0, 0.0, 1.0
v = 1.0


def Hfull(k, κ, γ2, Ω3):
    I = 1j
    H0 = (Ω0 - I * γ_bg) * np.eye(3)
    H_mat = np.array([[Ω3 / 2.5, v * k, 0],
                      [v * k, Ω2, κ],
                      [0, κ, Ω3]], dtype=complex)
    Γ = np.array([[γ1, np.sqrt(γ1 * γ2), np.sqrt(γ1 * γ3)],
                  [np.sqrt(γ1 * γ2), γ2, np.sqrt(γ2 * γ3)],
                  [np.sqrt(γ1 * γ3), np.sqrt(γ2 * γ3), γ3]], dtype=complex)
    return H0 + H_mat - 1j * Γ


def maxDiffSingle(k, κ, γ2, Ω3):
    evals = np.linalg.eigvals(Hfull(k, κ, γ2, Ω3))
    diffs = [abs(e1 - e2) for i, e1 in enumerate(evals)
             for e2 in evals[i + 1:]]
    return max(diffs)


def Ω3star(k, κ, γ2):
    grid = np.arange(-1, 0, 0.01)
    vals = [maxDiffSingle(k, κ, γ2, Ω3) for Ω3 in grid]
    return grid[int(np.argmin(vals))]


# 2. 准备 γ2 列表
gamma2List = [0, 0.05, 0.10, 0.15, 0.20]

# 3. 估计全局 Ω3* 范围
points = []
for γ2 in gamma2List:
    for k in np.linspace(-0.3, 0.3, 10):
        for κ in np.linspace(0.4, 0.6, 10):
            points.append(Ω3star(k, κ, γ2))
Ωmin, Ωmax = min(points), max(points)

# 4. 构造颜色映射
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=Ωmin, vmax=Ωmax)

# 5. 绘制 3D 图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

k_vals = np.linspace(-0.3, 0.3, 25)
κ_vals = np.linspace(0.4, 0.6, 25)
K, Kappa = np.meshgrid(k_vals, κ_vals)

for γ2 in gamma2List:
    # 计算 Ω3* 面
    Z = np.array([[Ω3star(k, κ, γ2) for k in k_vals] for κ in κ_vals])
    local_min, local_max = np.min(Z), np.max(Z)
    local_norm = plt.Normalize(vmin=local_min, vmax=local_max)
    # Surface 坐标
    X = K
    Y = Kappa
    Zsurf = np.full_like(X, γ2)

    # 计算面颜色
    facecolors = cmap(local_norm(Z))

    # 绘制半透明面
    ax.plot_surface(X, Y, Zsurf, rcount=25, ccount=25,
                    facecolors=facecolors, shade=False,
                    linewidth=0, alpha=0.8, antialiased=False)

# 6. 美化
ax.set_xlabel('k')
ax.set_ylabel('κ')
ax.set_zlabel('γ₂')
ax.set_title("Ω₃*(k, κ, γ₂) 在不同 γ₂ 切面")
ax.invert_zaxis()  # 对应 Mathematica 的 z 轴翻转
mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
ax.grid(False)
cbar = plt.colorbar(mappable, ax=ax, pad=0.1, label="Ω₃*")
# 设置 3D 坐标轴的比例
ax.set_box_aspect([1, 1, 3])  # 这里修改了比例
plt.tight_layout()
plt.show()
