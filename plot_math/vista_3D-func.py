import numpy as np
import pyvista as pv


def create_paraboloid(kx_range, ky_range, resolution, omega_Gamma, a_1):
    """
    创建二次型抛物面
    """
    kx = np.linspace(kx_range[0], kx_range[1], resolution)
    ky = np.linspace(ky_range[0], ky_range[1], resolution)

    KX, KY = np.meshgrid(kx, ky)
    omega = omega_Gamma + a_1 * (KX ** 2 + KY ** 2)  # 二次型抛物面方程

    grid = pv.StructuredGrid(KX, KY, omega)
    grid.point_data["omega"] = omega.flatten()

    return grid


def create_arrow(origin, direction, scale, shaft_radius, tip_radius):
    """
    创建箭头
    """
    return pv.Arrow(start=origin, direction=direction, scale=scale, shaft_radius=shaft_radius, tip_radius=tip_radius)


def create_xy_plane(kx_range, ky_range, z_height, resolution):
    """
    创建一个xy平面并设置z值，并映射为omega的颜色
    """
    kx = np.linspace(kx_range[0], kx_range[1], resolution)
    ky = np.linspace(ky_range[0], ky_range[1], resolution)

    KX, KY = np.meshgrid(kx, ky)
    Z = np.full_like(KX, z_height)

    # 创建平面网格
    plane = pv.StructuredGrid(KX, KY, Z)

    # 映射 omega 数据，保持与抛物面一致
    plane.point_data["omega"] = z_height

    return plane

def create_xy_base_plane(kx_range, ky_range, resolution):
    """
    创建一个xy平面并设置z值，并映射为omega的颜色
    """
    kx = np.linspace(kx_range[0], kx_range[1], resolution)
    ky = np.linspace(ky_range[0], ky_range[1], resolution)

    KX, KY = np.meshgrid(kx, ky)
    Z = np.full_like(KX, 0)

    # 创建平面网格
    plane = pv.StructuredGrid(KX, KY, Z)

    return plane


def main():
    # 设置常数
    # omega_Gamma = 1.5
    # a_1 = -1.0

    omega_Gamma = 1.25
    a_1 = -0.75
    kx_range = (-1, 1)  # kx 的范围
    ky_range = (-1, 1)  # ky 的范围
    resolution = 100  # 分辨率，控制网格细腻程度

    # 创建抛物面
    grid = create_paraboloid(kx_range, ky_range, resolution, omega_Gamma, a_1)
    # grid2 = create_paraboloid(kx_range, ky_range, resolution, omega_Gamma, -a_1)

    # 创建箭头
    origin = np.array([0, 0, 0])  # 箭头的原点位置
    x_arrow = create_arrow(origin, (1, 0, 0), scale=2, shaft_radius=0.01, tip_radius=0.02)
    y_arrow = create_arrow(origin, (0, 1, 0), scale=2, shaft_radius=0.01, tip_radius=0.02)
    z_arrow = create_arrow(origin, (0, 0, 1), scale=5, shaft_radius=0.01, tip_radius=0.02)

    # 创建xy平面
    xy_base_plane = create_xy_base_plane(kx_range, ky_range, resolution=resolution)
    xy_plane = create_xy_plane(kx_range, ky_range, z_height=4, resolution=resolution)

    # 可视化并保存图像
    plotter = pv.Plotter(off_screen=True, window_size=[2048, 2048])

    # 添加xy平面
    # plotter.add_mesh(xy_base_plane, color="white", opacity=1)

    # 添加抛物面
    plotter.add_mesh(grid, cmap="inferno_r", scalars="omega", opacity=0.5, clim=[0.5, 2.25])  # 显式指定颜色映射
    # plotter.add_mesh(grid2, cmap="inferno_r", scalars="omega", opacity=0.0)  # 显式指定颜色映射

    # 添加xy平面，应用inferno颜色映射
    # plotter.add_mesh(xy_plane, cmap="inferno_r", scalars="omega", opacity=0.5)

    # # 绘制箭头
    # for arrow in arrows:
    #     plotter.add_mesh(arrow, color="black")

    # 移除颜色条
    plotter.remove_scalar_bar()

    # 设置视角
    plotter.camera.azimuth = 10  # 通过视角设置
    # 设置正交视图
    plotter.enable_parallel_projection()
    plotter.set_background("white")

    # 保存带透明度的图像
    plotter.screenshot('./rsl/3d_surface.png', transparent_background=True)

    # 显示图形
    plotter.show()


if __name__ == "__main__":
    main()
