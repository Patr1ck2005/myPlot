import numpy as np
import pyvista as pv


def dipole_electric_field(X, Y, Z, p=np.array([0, 0, 1])):
    """
    计算电偶极子的电场分布。

    参数:
    - X, Y, Z: 三维网格点坐标。
    - p: 偶极矩方向，默认是 Z 方向 (0, 0, 1)。

    返回:
    - Ex, Ey, Ez: 电场的三个分量。
    """
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    r[r == 0] = 1e-9  # 避免除以零

    # 单位方向向量 r̂
    rx, ry, rz = X / r, Y / r, Z / r
    r_dot_p = rx * p[0] + ry * p[1] + rz * p[2]

    # 偶极子电场公式
    Ex = (3 * r_dot_p * rx - p[0]) / r ** 3
    Ey = (3 * r_dot_p * ry - p[1]) / r ** 3
    Ez = (3 * r_dot_p * rz - p[2]) / r ** 3

    return Ex, Ey, Ez


# 1. 创建三维网格
x = np.linspace(-2, 2, 30)  # 网格分辨率
y = np.linspace(-2, 2, 30)
z = np.linspace(-2, 2, 30)
X, Y, Z = np.meshgrid(x, y, z)

# 2. 计算电场的三个分量和强度
Ex, Ey, Ez = dipole_electric_field(X, Y, Z)
E_magnitude = np.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)

# 3. 归一化三个分量和强度
Ex_n = (Ex - Ex.min()) / (Ex.max() - Ex.min())  # 归一化到 [0, 1]
Ey_n = (Ey - Ey.min()) / (Ey.max() - Ey.min())
Ez_n = (Ez - Ez.min()) / (Ez.max() - Ez.min())

# 将场的 RGB 分量作为颜色映射
colors = np.stack((Ex_n, Ey_n, Ez_n), axis=-1)  # RGB 颜色合成

# 4. 构造 PyVista 结构化网格
grid = pv.ImageData()
grid.dimensions = np.array(Ex.shape) + 1  # 设置网格尺寸
grid.origin = (x.min(), y.min(), z.min())  # 坐标网格原点
grid.spacing = (  # 网格步长
    (x.max() - x.min()) / Ex.shape[0],
    (y.max() - y.min()) / Ey.shape[1],
    (z.max() - z.min()) / Ez.shape[2],
)

# 将场强数据存储到网格中
grid["Electric Field"] = E_magnitude.ravel(order="F")

# 5. 使用场值（强度）控制透明度映射
opacity = [0.0, 0.3, 0.6, 0.8, 1.0]  # 根据强度值设置透明度分布

# 6. 渲染数据
plotter = pv.Plotter()
plotter.add_volume(
    grid,
    scalars="Electric Field",  # 使用电场强度作为标量
    cmap="coolwarm",  # 强度颜色映射
    opacity=opacity,  # 透明度控制
)
plotter.add_axes()  # 添加坐标轴
plotter.show()
