import numpy as np
import pyvista as pv


def dipole_electric_field(X, Y, Z, p=np.array([0, 0, 0])):
    """
    计算电偶极子的电场在三维空间中每个点上的向量 (Ex, Ey, Ez)。

    参数:
    - X, Y, Z: 三维网格点坐标。
    - p: 偶极矩方向 (默认沿 Z 轴方向)。

    返回:
    - Ex, Ey, Ez: 在每个点的电场分量。
    """
    # 计算距离 r
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r[r == 0] = 1e-9  # 避免 0 除法

    # 电场公式
    rx, ry, rz = X / r, Y / r, Z / r
    r_dot_p = rx * p[0] + ry * p[1] + rz * p[2]

    Ex = (3 * r_dot_p * rx - p[0]) / r**3
    Ey = (3 * r_dot_p * ry - p[1]) / r**3
    Ez = (3 * r_dot_p * rz - p[2]) / r**3

    return Ex, Ey, Ez


def point_charge_field(X, Y, Z, q, position):
    """
    计算单个点电荷在空间中产生的电场。

    参数：
    - X, Y, Z: 三维网格点坐标。
    - q: 点电荷的电荷量（正负表示正电荷或负电荷）。
    - position: 点电荷的位置 (x, y, z)。

    返回：
    - Ex, Ey, Ez: 该点电荷在每个点的电场分量。
    """
    dx = X - position[0]
    dy = Y - position[1]
    dz = Z - position[2]

    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r[r == 0] = 1e-9  # 避免 0 除法

    E = q / r**2  # 库仑电场公式
    Ex = E * (dx / r)
    Ey = E * (dy / r)
    Ez = E * (dz / r)

    return Ex, Ey, Ez

N = 32
# 生成三维网格点
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
z = np.linspace(-1, 1, N)
X, Y, Z = np.meshgrid(x, y, z)
R2 = X**2+Y**2+Z**2

# 计算电偶极子的电场向量
Ex_d, Ey_d, Ez_d = dipole_electric_field(X, Y, Z)

# 添加独立点电荷贡献
# 定义多个点电荷（位置与电荷量）
point_charges = [
    {"q": +1.0, "position": [0.3, 0.3, 0.0]},   # 正电荷
    {"q": +1.0, "position": [0.1, 0.3, 0.3]},   # 正电荷
    {"q": -3.0, "position": [-0.1, -0.1, 0.0]}, # 负电荷
]

# 初始化点电荷电场为零
Ex_p, Ey_p, Ez_p = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)

for charge in point_charges:
    Ex_c, Ey_c, Ez_c = point_charge_field(X, Y, Z, charge["q"], charge["position"])
    Ex_p += Ex_c
    Ey_p += Ey_c
    Ez_p += Ez_c

# 将电偶极子和点电荷电场叠加
Ex = Ex_d + Ex_p
Ey = Ey_d + Ey_p
Ez = Ez_d + Ez_p

# 裁剪可视化区域
Ex[R2 > 1] = 0
Ey[R2 > 1] = 0
Ez[R2 > 1] = 0

# 计算电场强度（用于透明度渲染）
E_magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)
E_opacity = np.power(E_magnitude, 1/5)
E_opacity /= np.max(E_opacity)  # 归一化场强到范围 [0, 1]

# 构造 RGB 颜色映射（三个方向分量分别映射到 R, G, B）
# 逐点归一化电场曲线至 [-1, 1]
Ex_norm = Ex/E_magnitude
Ey_norm = Ey/E_magnitude
Ez_norm = Ez/E_magnitude
Ex_n = (Ex_norm+1)/2  # 归一化至 [0, 1]
Ey_n = (Ey_norm+1)/2
Ez_n = (Ez_norm+1)/2
colors = np.stack((Ex_n, Ey_n, Ez_n), axis=-1)  # 将 Ex, Ey, Ez 组合成 RGB

#######################################################################################################################
# 创建矢量箭头点云
vectors = np.c_[Ex_norm.ravel(), Ey_norm.ravel(), Ez_norm.ravel()]  # 每个点上的电场矢量
centers = np.c_[X.ravel(), Y.ravel(), Z.ravel()]    # 每个点的坐标位置

# 使用 PyVista 显示矢量箭头场
point_cloud = pv.PolyData(centers)
point_cloud["vectors"] = vectors

# 创建箭头
arrows = point_cloud.glyph(orient="vectors", scale=True, factor=0.03)  # factor 控制箭头长度缩放

# 创建 Plotter
plotter = pv.Plotter()
plotter.add_mesh(arrows, color=colors)  # 渲染箭头
plotter.add_axes(interactive=True)  # 显示交互式轴
plotter.show()
