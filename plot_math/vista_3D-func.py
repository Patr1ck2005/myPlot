import numpy as np
import pyvista as pv
from plot_math.tcmt import TCMTSolver  # 根据实际情况确认此模块是否需要


# 如果 utils 中的 clear_ax_ticks 用于 2D 绘图，可以根据需要决定是否保留引用
# from utils.utils import clear_ax_ticks


def create_paraboloid(kx_range, ky_range, resolution, omega_Gamma, a_1, a_2, scalar_field="omega"):
    """
    创建二次型抛物面，计算对应的函数值并将其存入 point_data 。

    :param kx_range: kx 的范围 (min, max)。
    :param ky_range: ky 的范围 (min, max)。
    :param resolution: 网格点数，决定曲面的细腻程度。
    :param omega_Gamma: 常量项。
    :param a_1: 一次非线性项系数。
    :param a_2: 二次非线性项系数。
    :param scalar_field: 指定存储函数值的字段名称，默认为 "omega"。
    :return: 包含计算数据的 StructuredGrid 对象。
    """
    kx = np.linspace(kx_range[0], kx_range[1], resolution)
    ky = np.linspace(ky_range[0], ky_range[1], resolution)
    KX, KY = np.meshgrid(kx, ky)

    # 计算抛物面函数值
    omega = omega_Gamma + a_1 * (KX ** 2 + KY ** 2) + a_2 * (KX ** 2 + KY ** 2) ** 2

    grid = pv.StructuredGrid(KX, KY, omega)
    grid.point_data[scalar_field] = omega.flatten()
    R2 = (KX ** 2 + KY ** 2)
    efficiency = 0 + np.exp(-R2/4)*np.clip(R2*10, 0, 1.0)
    grid.point_data['efficiency'] = efficiency.flatten()

    return grid


def create_arrow(origin, direction, scale, shaft_radius, tip_radius):
    """
    创建箭头，用于标示坐标轴。

    :param origin: 箭头起点。
    :param direction: 箭头方向。
    :param scale: 尺寸缩放。
    :param shaft_radius: 箭身半径。
    :param tip_radius: 箭头半径。
    :return: 一个 Arrow 对象。
    """
    return pv.Arrow(start=origin, direction=direction, scale=scale,
                    shaft_radius=shaft_radius, tip_radius=tip_radius)


def create_xy_plane(kx_range, ky_range, z_value, resolution, scalar_field="omega"):
    """
    创建一个 xy 平面，并在点数据中设置标量数据。

    :param kx_range: kx 轴范围。
    :param ky_range: ky 轴范围。
    :param z_value: 平面 z 坐标（高度）。
    :param resolution: 网格分辨率。
    :param scalar_field: 指定存储数据的字段名称。
    :return: 一个 xy 平面的 StructuredGrid 对象。
    """
    kx = np.linspace(kx_range[0], kx_range[1], resolution)
    ky = np.linspace(ky_range[0], ky_range[1], resolution)
    KX, KY = np.meshgrid(kx, ky)
    Z = np.full_like(KX, z_value)

    plane = pv.StructuredGrid(KX, KY, Z)
    plane.point_data[scalar_field] = np.full(KX.shape, z_value).flatten()

    return plane


def create_xy_base_plane(kx_range, ky_range, resolution):
    """
    创建一个基础的 xy 平面，z 坐标设为 0。

    :param kx_range: kx 轴范围。
    :param ky_range: ky 轴范围。
    :param resolution: 网格分辨率。
    :return: 一个 xy 平面的 StructuredGrid 对象。
    """
    return create_xy_plane(kx_range, ky_range, z_value=0, resolution=resolution, scalar_field="omega")


def add_surface(plotter, surface, scalar_field="omega", cmap="Blues_r", clim=None,
                opacity=1.0, lighting=False):
    """
    添加曲面到 Plotter 上，同时可以指定用于颜色映射的数据字段。

    :param plotter: pyvista.Plotter 实例。
    :param surface: 要添加的 surface 对象（例如 StructuredGrid）。
    :param scalar_field: 指定用于颜色映射的数据字段名称。
    :param cmap: 颜色映射名称。
    :param clim: 颜色范围 [min, max]。
    :param opacity: 不透明度。
    :param lighting: 是否启用光照计算。
    """
    plotter.add_mesh(
        surface,
        scalars=scalar_field,
        cmap=cmap,
        opacity=opacity,
        clim=clim,
        lighting=lighting,
        # interpolation="phong",  # 使用 Phong 着色
        ambient=0.2,  # 环境光系数（越大整体越亮）
        # diffuse=0.7,  # 漫反射系数（控制面光滑度）
        # specular=0.6,  # 镜面反射系数（高光强度）
        # specular_power=20  # 镜面高光粗糙度（越大高光越集中）
    )


def main():
    # 设置常数
    omega_Gamma = 1.4
    # a_1 = 0.8
    # a_2 = -1
    # a_1 = -1.5
    # a_2 = 0
    a_1 = 0.6
    a_2 = 0
    kx_range = (-1, 1)
    ky_range = (-1, 1)
    resolution = 100

    # 创建完整的抛物面
    grid_full = create_paraboloid(kx_range, ky_range, resolution, omega_Gamma, a_1, a_2, scalar_field="omega")

    # 利用圆柱体裁剪抛物面得到局部区域
    cylinder = pv.Cylinder(center=(0, 0, 0),
                           direction=(0, 0, 1),
                           radius=1.0,
                           height=100)
    grid_clipped = grid_full.clip_surface(cylinder)

    # 创建箭头用于标示坐标轴
    origin = np.array([0, 0, 0])
    x_arrow = create_arrow(origin, (1, 0, 0), scale=2, shaft_radius=0.01, tip_radius=0.02)
    y_arrow = create_arrow(origin, (0, 1, 0), scale=2, shaft_radius=0.01, tip_radius=0.02)
    z_arrow = create_arrow(origin, (0, 0, 1), scale=5, shaft_radius=0.01, tip_radius=0.02)

    # 创建基础 xy 平面（例如辅助显示）
    xy_base_plane = create_xy_base_plane(kx_range, ky_range, resolution=resolution)

    # 设置 Plotter，注意此处 off_screen=True 用于批量渲染或自动保存图像
    plotter = pv.Plotter(off_screen=True, window_size=[2048, 2048])

    # 添加剪裁后的抛物面，并允许通过 scalar_field 参数指定颜色映射数据
    add_surface(
        plotter,
        grid_clipped,
        scalar_field="efficiency",  # 可通过此参数指定映射数据
        cmap="magma",
        clim=[0, 1],
        opacity=1.0,
        lighting=False
    )

    # 根据实际需要，可以添加 xy 基础平面或箭头等其它几何体
    # plotter.add_mesh(xy_base_plane, color="white", opacity=1)
    # # 添加箭头
    # for arrow in (x_arrow, y_arrow, z_arrow):
    #     plotter.add_mesh(arrow, color="black")

    # 移除颜色条
    plotter.remove_scalar_bar()

    # 设置相机视角并启用正交投影
    plotter.camera.azimuth = 10
    plotter.enable_parallel_projection()
    plotter.set_background("white")

    # 保存图像，注意 transparent_background=True 可生成透明背景的 png
    plotter.screenshot('./rsl/3d_surface.png', transparent_background=True)

    # 显示图形（若需要交互显示）
    plotter.show()


if __name__ == "__main__":
    main()
