import numpy as np
import re
import os
import pyvista as pv


def plot_Z_diff_pyvista(
        coords,
        Z_raws,
        x_key,
        plot_params,
        y_key,
        fixed_params=None,
        show_live=False,
):
    """
    利用 PyVista 绘制 3D 曲面，并改善多曲面遮挡问题。

    参数：
        coords: dict，键为参数名，值为该参数的坐标数组（已排序）。
        Z_raws: [np.ndarray]，包含多个多维数组的一维列表，每个数组维度与 new_coords 顺序一致，存放绘图数据。
        x_key: str，要作为横坐标的参数名。
        y_key: str，要作为纵坐标的参数名（此处只处理 3D 曲面的情况）。
        plot_params: dict，绘图参数，例如：
            - 'cmap1': 实部使用的 colormap 名称（如 'viridis'）
            - 'cmap2': 虚部使用的 colormap 名称（如 'plasma'）
            - 'log_scale': 是否对数显示（True/False）
            - 'zlabel': z 轴标签文本
            - 'alpha': 曲面透明度（0~1之间）
        fixed_params: dict，其它参数的固定取值，例如 {"a": 0.0085}。
            固定参数的键应为 new_coords 中除 x_key, y_key 外的参数名。
        show_live: 是否开始交互式窗口实时显示绘图结果

    绘图文件将以 "plot_<参数信息>.png" 命名，并保存在当前工作目录。
    """

    # 1. 参数检查
    keys = list(coords.keys())
    if x_key not in keys:
        raise ValueError(f"{x_key} 不在 new_coords 中")
    if y_key not in keys or y_key == x_key:
        raise ValueError(f"{y_key} 不在 new_coords 中或与 {x_key} 重复")
    fixed_params = fixed_params or {}
    for k in fixed_params:
        if k not in keys or k in (x_key, y_key):
            raise ValueError(f"固定参数 {k} 无效")

    # 2. 构建切片索引
    slicer = []
    for k in keys:
        if k == x_key or k == y_key:
            slicer.append(slice(None))
        else:
            if k not in fixed_params:
                raise ValueError(f"参数 {k} 未在 fixed_params 中指定")
            val = fixed_params[k]
            idx = np.where(coords[k] == val)[0]
            if idx.size != 1:
                raise ValueError(f"{k} 中未找到唯一的值 {val}")
            slicer.append(idx[0])
    slicer = tuple(slicer)

    log_scale = plot_params.get('log_scale', False)
    zlabel = plot_params.get('zlabel', "Δ")
    cmap1_name = plot_params.get('cmap1', 'viridis')
    cmap2_name = plot_params.get('cmap2', 'plasma')
    alpha_val = plot_params.get('alpha', 1.0)
    data_scale = plot_params.get('data_scale', [1, 1, 1])
    render_real = plot_params.get('render_real', True)
    render_imag = plot_params.get('render_imag', True)
    apply_abs = plot_params.get('apply_abs', False)

    # 创建绘图器
    p = pv.Plotter(off_screen=not show_live, window_size=[1024, 1024])
    for i, Z_raw in enumerate(Z_raws):
        # 3. 取出子数组与坐标值，并进行缩放
        sub = Z_raw[slicer]
        x_vals = coords[x_key]
        y_vals = coords[y_key]
        scale_factor = 1 / 0.001
        Z = sub * scale_factor

        # 4. 构建网格，采用 indexing='ij' 构造二维网格
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        X, Z, Y = data_scale[0]*X, data_scale[1]*Z, data_scale[2]*Y

        if log_scale:
            Z_real_plot = np.log10(np.abs(Z.real))
            Z_imag_plot = np.log10(np.abs(Z.imag))
        elif apply_abs:
            Z_real_plot = abs(Z.real)
            Z_imag_plot = abs(Z.imag)

        vmin_real = plot_params.get('vmin_real', np.min(Z_real_plot))
        vmax_real = plot_params.get('vmax_real', np.max(Z_real_plot))
        if vmax_real < vmin_real:
            vmax_real = vmin_real+0.1
        vmin_imag = plot_params.get('vmin_imag', np.min(Z_imag_plot))
        vmax_imag = plot_params.get('vmax_imag', np.max(Z_imag_plot))
        if vmax_imag < vmin_imag:
            vmax_imag = vmin_imag+0.1

        # 构建结构化网格；注意 Z 数据在赋值前进行了转置以匹配 X, Y 维度
        grid_real = pv.StructuredGrid(X, Z_real_plot, Y)
        # 将 scalars 设置为 Fortran 顺序展平的数据
        grid_real.point_data["scalars"] = Z_real_plot.ravel(order="F")
        grid_real.set_active_scalars("scalars")

        grid_imag = None
        if np.iscomplexobj(Z):
            grid_imag = pv.StructuredGrid(X, Z_imag_plot, Y)
            grid_imag.point_data["scalars"] = Z_imag_plot.ravel(order="F")
            grid_imag.set_active_scalars("scalars")

        # 5. 向绘图器添加网格
        if grid_imag is not None and render_imag:  # render imag part
            actor = p.add_mesh(
                grid_imag, scalars="scalars", cmap=cmap2_name, opacity=alpha_val,
                show_scalar_bar=True, scalar_bar_args={'title': zlabel + ' (imag)'}, name=f'Imag Surface {i}',
                clim=[vmin_imag, vmax_imag]
            )
            # actor.SetScale(10000, 1, 1)
        if render_real:
            # render real part
            actor = p.add_mesh(
                grid_real, scalars="scalars", cmap=cmap1_name, opacity=alpha_val,
                show_scalar_bar=True, scalar_bar_args={'title': zlabel + ' (real)'}, name=f'Real Surface {i}',
                clim=[vmin_real, vmax_real]
            )
            # actor.SetScale(10000, 1, 1)
    p.show_grid()
    # p.show_bounds()
    title_str = f"{x_key} vs {y_key} @ { {k: v for k, v in fixed_params.items()} }"
    p.add_text(title_str, position="upper_left", font_size=12)
    p.camera_position = 'iso'

    # set perspective mode
    p.camera.parallel_projection = True
    p.camera.azimuth = 0
    p.camera.elevation = 0
    p.camera.roll = 0
    p.window_size = [1024, 1024]
    p.camera.Zoom(0.8)
    # 定义一个水平移动量
    ds = 3
    # 分别获取当前摄像机位置和焦点
    pos = list(p.camera.position)
    fp = list(p.camera.focal_point)
    # 对 position 和 focal_point 同时加上 dx（只修改 X 坐标）
    pos[1] -= ds
    fp[1] -= ds
    # 设置修改后的值
    p.camera.position = pos
    p.camera.focal_point = fp
    # p.camera.distance = 100
    # p.camera_position = [2000, 2000,  2000]
    # p.camera.focal_point = [90, 220,  3]
    # # 设置视角（视场角，单位为度；例如设置为30度）
    # p.camera.view_angle = 30

    if show_live:
        p.show()

    # 6. 保存截图
    def safe_str(val):
        return re.sub(r'[^\w.-]', '', str(val))

    full_params = {**fixed_params, **plot_params}
    param_items = [f"{k}-{safe_str(v)}" for k, v in sorted(full_params.items())]
    filename = "plot_" + "_".join(param_items) + ".png"
    if len(filename) > 200:
        filename = filename[:200] + ".png"
    save_path = os.path.join("./rsl/pv", filename)
    p.screenshot(save_path)
    print(f"图像已保存为：{save_path}")
    p.show()
