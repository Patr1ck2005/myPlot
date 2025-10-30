# plot_bands_advanced.py

from __future__ import annotations
import numpy as np
import scipy.ndimage
import skimage.measure  # 用于等值线提取
from typing import Callable, Dict, Tuple, List, Optional

from scipy.interpolate import griddata

# 导入高级绘图库
try:
    import plotly.graph_objects as go
    from plotly.io import write_image  # 需要 pip install plotly kaleido
except ImportError:
    go = None
    write_image = None
    print("[WARN] Plotly or Kaleido not installed. Plotly examples will be skipped.")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize  # 用于颜色映射
    from matplotlib import cm  # Colormaps
except ImportError:
    plt = None
    Axes3D = None
    Normalize = None
    cm = None
    print("[WARN] Matplotlib not installed. Matplotlib examples will be skipped.")

# =========================
# Physics parameters (从原脚本复制)
# =========================
GLOBAL_FACTOR = 0.5
E_ref = -0.6
A_iso = -1.5
A_4 = -1.0
inv_mass = 0.50
E_offset = -1
KMAX = 1.0
NGRID = 240


# =========================
# Colorizers (从原脚本复制)
# =========================
def colorize_height(kx, ky, z):
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if np.isclose(zmax, zmin): return np.zeros_like(z, dtype=float)
    return (z - zmin) / (zmax - zmin)


def colorize_radial(kx, ky, z):
    r = np.sqrt(kx ** 2 + ky ** 2)
    rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
    if np.isclose(rmax, rmin): return np.zeros_like(r, dtype=float)
    return (r - rmin) / (rmax - rmin)


COLORIZER_REGISTRY: Dict[str, Callable] = {
    "height": colorize_height,
    "radial": colorize_radial,
}


# =========================
# Surface definitions (从原脚本复制)
# =========================
def S1_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    return np.full_like(kx, E_ref, dtype=float)


def S2_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    r2 = kx ** 2 + ky ** 2
    th = np.arctan2(ky, kx)
    return r2 * (A_iso + A_4 * np.cos(4.0 * th))


def S3_func(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    r2 = kx ** 2 + ky ** 2
    return inv_mass * r2 + E_offset


SURFACE_FUNCS = {
    "S1": S1_func,
    "S2": S2_func,
    "S3": S3_func,
}

# =========================
# MASTER CONFIG (从原脚本简化并适应)
# =========================
CONFIG = {
    "scene": {
        "window_size": (1200, 900),  # Plotly/Matplotlib 可以用这个来设定输出分辨率
        "background_color": "white",  # 用于 Plotly/Matplotlib
        "camera_settings": {"elev": 25, "azim": 35, "roll": 0},  # Plotly/Matplotlib 视角
        "clean_view": True,  # 用于控制 Plotly/Matplotlib 是否显示默认轴背景等
    },

    "surfaces": {
        "S1": {
            "visible": True,
            "mode": "solid",
            "color": (0.55, 0.55, 0.55),  # Matplotlib/Plotly 支持 RGB tuple
            "opacity": 0.30,
            "cmap": "viridis",  # 用于 colormap 模式
            "colorizer": None,
            "scalar_bar": False,
            "line_width": 1.0,  # 用于 wireframe 模式或 Plotly 的 surface contour
            "polar_mode": False,  # 是否使用极坐标网格
        },
        "S2": {
            "visible": True,
            "mode": "colormap",
            "color": (0.2, 0.2, 0.2),  # solid 模式备用色
            "opacity": 0.95,
            "cmap": "twilight",
            "colorizer": "height",
            "scalar_bar": False,
            "line_width": 1.0,
            "polar_mode": False,
        },
        "S3": {
            "visible": True,
            "mode": "solid",
            "color": (0.3, 0.5, 0.85),
            "opacity": 0.40,
            "cmap": "viridis",
            "colorizer": None,
            "scalar_bar": False,
            "line_width": 1.0,
            "polar_mode": False,
        },
    },

    "intersections": {
        "draw": True,
        "pairs": [
            {"surface1": "S2", "surface2": "S1", "color": "white", "line_width": 3.0},
            {"surface1": "S2", "surface2": "S3", "color": "blue", "line_width": 4.0},
        ],
        "tol_abs": 1e-3,  # 等值线提取的绝对容差
    },

    "spheres": [
        {"center": (0.5, 0.5, None), "radius": 0.06, "color": (1.0, 0.4, 0.4), "opacity": 0.9},
        {"center": (1.0, 0.0, None), "radius": 0.06, "color": (0.4, 1.0, 0.4), "opacity": 0.9},
        {"center": (-0.8, 0.6, None), "radius": 0.06, "color": (0.4, 0.6, 1.0), "opacity": 0.9},
    ],

    "output": {
        "output_prefix": "./band_plot",  # 输出文件的前缀
        "matplotlib_axes_config": {  # 统一的轴配置，Plotly 和 Matplotlib 都可以用
            "labels_off": False,
            "label_font_size": 12,
            "title_font_size": 14,
            "font_family": "serif",
            "label_color": "black",
            "tick_color": "gray",
            "line_width": 1.5,
            "tick_width": 1.5,
            "xtitle": "k_x",
            "ytitle": "k_y",
            "ztitle": "E",
        },
    },
}


# =========================
# 数据处理和几何计算部分
# =========================

def generate_grid_and_z(surface_name: str, sconf: Dict, KMAX: float, NGRID: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """根据配置生成曲面的 KX, KY, Z 数据"""
    polar = sconf.get("polar_mode", False)
    if polar:
        r = np.linspace(0, KMAX, NGRID)
        # 增加 theta 采样点，确保圆形网格平滑
        theta = np.linspace(0, 2 * np.pi, int(NGRID * 1.5))
        R, THETA = np.meshgrid(r, theta, indexing="ij")
        KX = R * np.cos(THETA)
        KY = R * np.sin(THETA)
    else:
        k = np.linspace(-KMAX, KMAX, NGRID)
        KX, KY = np.meshgrid(k, k, indexing="xy")

    Z = SURFACE_FUNCS[surface_name](KX, KY)
    return KX, KY, Z


def find_contours_3d(KX: np.ndarray, KY: np.ndarray, Z_diff: np.ndarray, level: float, tol_abs: float) -> List[
    np.ndarray]:
    """
    在 3D 表面上寻找等值线。
    这里使用 skimage.measure.find_contours 寻找 2D 等值线，然后将其提升到 3D。
    """
    # 将 Z_diff 差值数组重采样到 uniform grid 以适应 find_contours
    # 或者直接在原始 grid 上查找

    # 查找 2D 等值线 (在 Z_diff 数组中找 level 处的等值线)
    # skimage.measure.find_contours 默认处理的是 (rows, cols) 索引
    # 我们需要将其映射回 (KX, KY) 坐标

    # 假设 KX, KY 都是 (M, N) 形状的二维数组
    # contours = skimage.measure.find_contours(Z_diff, level)
    # 对于不规则网格，find_contours 可能不够鲁棒，或者需要插值到一个规则网格。
    # 简单起见，我们直接在 Z_diff 数组上找，并假设 Z_diff 足够平滑。

    # 考虑到 PyVista 的 _zero_contour 是一种鲁棒方法，这里模拟其核心思想：
    # 1. 查找 Z_diff == level 的区域。
    # 2. 对这些区域进行细化，形成线条。

    # 一个更鲁棒的 2D 等值线提取方法，可以处理非均匀网格的：
    # 创建一个中间的 2D 规则网格，并插值 Z_diff
    k_x_flat = KX.flatten()
    k_y_flat = KY.flatten()
    z_diff_flat = Z_diff.flatten()

    # 定义一个统一的 2D 栅格，用于插值和等值线查找
    min_k = np.min(k_x_flat)
    max_k = np.max(k_x_flat)
    grid_res = NGRID * 2  # 更高的分辨率用于插值
    grid_x = np.linspace(min_k, max_k, grid_res)
    grid_y = np.linspace(min_k, max_k, grid_res)
    GRID_X, GRID_Y = np.meshgrid(grid_x, grid_y, indexing='xy')

    # 使用 Scipy 的 griddata 进行插值
    from scipy.interpolate import griddata
    Z_diff_interpolated = griddata((k_x_flat, k_y_flat), z_diff_flat, (GRID_X, GRID_Y), method='cubic')

    # 查找等值线
    raw_contours = skimage.measure.find_contours(Z_diff_interpolated, level, fully_connected='high')

    all_3d_points = []
    for contour in raw_contours:
        # contour 包含 (row, col) 索引，需要映射回 (k_x, k_y) 坐标
        k_x_coords = grid_x[contour[:, 1].astype(int)]
        k_y_coords = grid_y[contour[:, 0].astype(int)]

        # 重新计算这些 (k_x, k_y) 上的 Z 值（例如，使用曲面 S2 的函数）
        # 或者插值原始 Z 场
        # 这里为了简单，我们假设交线是在 Z_S2 上提取的
        # 应该使用原曲面函数来获取 Z，而不是 Z_diff
        Z_on_contour = SURFACE_FUNCS["S2"](k_x_coords, k_y_coords)  # 假设交线在 S2 上

        all_3d_points.append(np.vstack([k_x_coords, k_y_coords, Z_on_contour]).T)

    return all_3d_points


def prepare_plot_data(config: Dict, surface_funcs: Dict, KMAX: float, NGRID: int) -> Dict:
    """
    处理原始配置和物理函数，生成可在 Plotly/Matplotlib 中直接使用的绘图数据。
    """
    sc = config["scene"]
    surfaces_config = config["surfaces"]
    intersections_config = config["intersections"]

    # --- 1. 生成所有曲面的网格数据和 Z 值 ---
    plot_surfaces_data = {}  # 存储最终用于绘图的 KX, KY, Z
    raw_Z_fields = {}  # 存储原始 Z_fields (用于交线计算)

    # 先为所有曲面生成基础网格和Z_fields
    for name, sconf in surfaces_config.items():
        if not sconf.get("visible", True):
            continue
        KX, KY, Z = generate_grid_and_z(name, sconf, KMAX, NGRID)
        plot_surfaces_data[name] = {
            "KX": KX,
            "KY": KY,
            "Z": Z,
            "sconf": sconf,
        }
        raw_Z_fields[name] = Z  # 存储原始 Z_fields 用于差值计算

    # --- 2. 计算并提取交线数据 ---
    plot_intersections_data = []
    if intersections_config.get("draw", True):
        pairs = intersections_config.get("pairs", [])
        tol_abs = float(intersections_config.get("tol_abs", 1e-3))  # 使用绝对容差

        # 为了计算交线，我们需要一个统一的网格作为“基准”
        # 这里假设使用 S2 的网格作为基准，或者直接使用最密集的笛卡尔网格
        # 我们可以创建一个公共的笛卡尔网格作为参考
        base_k_lin = np.linspace(-KMAX, KMAX, NGRID)
        BASE_KX, BASE_KY = np.meshgrid(base_k_lin, base_k_lin, indexing="xy")

        # 插值所有 Z 字段到这个统一网格
        interpolated_Z_fields = {}
        for name, data in plot_surfaces_data.items():
            KX_flat = data["KX"].flatten()
            KY_flat = data["KY"].flatten()
            Z_flat = data["Z"].flatten()
            interpolated_Z_fields[name] = griddata((KX_flat, KY_flat), Z_flat, (BASE_KX, BASE_KY), method='cubic')
            # 确保插值结果的形状正确
            interpolated_Z_fields[name] = interpolated_Z_fields[name].reshape(NGRID, NGRID)

        for pconf in pairs:
            a, b = pconf["surface1"], pconf["surface2"]

            # 在统一的 BASE_KX, BASE_KY 网格上计算 Z_diff
            Z_diff = interpolated_Z_fields[a] - interpolated_Z_fields[b]

            # 使用 skimage.measure.find_contours 寻找等值线
            # find_contours 返回的是子像素精度，可以直接使用
            contours = skimage.measure.find_contours(Z_diff, level=0, fully_connected='high')

            for contour in contours:
                # contour 包含 (row, col) 索引，需要映射回 (k_x, k_y) 坐标
                # 假设 BASE_KX, BASE_KY 是均匀间隔的
                k_x_coords = BASE_KX[contour[:, 0].astype(int), contour[:, 1].astype(int)]
                k_y_coords = BASE_KY[contour[:, 0].astype(int), contour[:, 1].astype(int)]

                # Z 值应该来自交线所在的曲面之一，通常是 S2
                # 可以在这些 (k_x, k_y) 点上重新计算 S2 的 Z 值
                # 或者使用插值后的 Z_S2
                Z_on_contour = interpolated_Z_fields["S2"][contour[:, 0].astype(int), contour[:, 1].astype(int)]

                if len(k_x_coords) > 1:  # 确保至少有两点形成线段
                    plot_intersections_data.append({
                        "label": f"{a}∩{b}",
                        "points": np.vstack([k_x_coords, k_y_coords, Z_on_contour]).T,
                        "pconf": pconf,
                    })
                else:
                    print(f"[WARN] Intersection {a}∩{b} has too few points. Skipped.")

    # --- 3. 提取相机和轴标题配置 ---
    camera_config = config["scene"]["camera_settings"]  # 直接使用新的 camera_settings

    return {
        "surfaces_data": plot_surfaces_data,
        "intersections_data": plot_intersections_data,
        "spheres_data": config["spheres"],  # 直接传递原始球体配置
        "camera_config": camera_config,
        "axes_config": config["output"]["matplotlib_axes_config"],  # 统一的轴配置
        "scene_config": config["scene"],  # 传递场景配置 (例如 clean_view)
        "KMAX": KMAX,  # 传递 KMAX 用于球体位置检查
    }


# =========================
# Plotly 绘图函数 (测试样例)
# =========================
def plot_with_plotly(plot_data: Dict, output_path_prefix: str = './band_plot'):
    if go is None or write_image is None:
        return

    print("[INFO] Plotting with Plotly...")
    fig = go.Figure()

    axes_config = plot_data["axes_config"]
    scene_config = plot_data["scene_config"]

    # --- 1. 绘制曲面 ---
    for name, data in plot_data["surfaces_data"].items():
        sconf = data["sconf"]
        if not sconf.get("visible", True):
            continue

        color_scalars = None
        if sconf.get("mode") == "colormap":
            color_scalars = COLORIZER_REGISTRY[sconf["colorizer"]](data["KX"], data["KY"], data["Z"])
            surface_trace = go.Surface(
                x=data["KX"],
                y=data["KY"],
                z=data["Z"],
                surfacecolor=color_scalars,
                colorscale=sconf.get("cmap", "Viridis"),
                opacity=sconf.get("opacity", 1.0),
                name=name,
                showscale=sconf.get("scalar_bar", False),
                colorbar=dict(title=sconf.get("colorizer", "")),  # 色条标题
                cmin=np.nanmin(color_scalars) if color_scalars is not None else None,
                cmax=np.nanmax(color_scalars) if color_scalars is not None else None,
            )
        elif sconf.get("mode") == "solid":
            surface_trace = go.Surface(
                x=data["KX"],
                y=data["KY"],
                z=data["Z"],
                colorscale=[[0, sconf.get("color")], [1, sconf.get("color")]],  # 模拟纯色
                showscale=False,
                opacity=sconf.get("opacity", 1.0),
                name=name,
            )
        elif sconf.get("mode") == "wireframe":
            surface_trace = go.Surface(
                x=data["KX"],
                y=data["KY"],
                z=data["Z"],
                opacity=sconf.get("opacity", 0.1),  # 低透明度模拟线框
                name=name,
                # Plotly 的 wireframe 效果需要开启 contours，但不能直接控制线宽
                contours_x_show=True, contours_x_start=np.min(data["KX"]), contours_x_end=np.max(data["KX"]),
                contours_x_size=(np.max(data["KX"]) - np.min(data["KX"])) / 10,
                contours_y_show=True, contours_y_start=np.min(data["KY"]), contours_y_end=np.max(data["KY"]),
                contours_y_size=(np.max(data["KY"]) - np.min(data["KY"])) / 10,
                line_width=sconf.get("line_width", 1.0),  # Plotly 1.15+ 支持 surface.line_width
            )
        else:
            print(f"[WARN] Unknown mode {sconf.get('mode')} for {name}. Skipping.")
            continue
        fig.add_trace(surface_trace)

    # --- 2. 绘制交线 ---
    for inter_data in plot_data["intersections_data"]:
        points = inter_data["points"]
        pconf = inter_data["pconf"]
        if points.shape[0] > 0:
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='lines',
                line=dict(color=pconf.get("color", "black"), width=pconf.get("line_width", 3.0)),
                name=inter_data["label"]
            ))

    # --- 3. 绘制球体 ---
    KMAX_val = plot_data["KMAX"]
    for i, sph in enumerate(plot_data["spheres_data"], start=1):
        cx, cy, cz = sph.get("center", (0.0, 0.0, None))
        r_sph = np.sqrt(cx ** 2 + cy ** 2)
        if r_sph > KMAX_val:
            print(f"[WARN] Sphere {i} at r={r_sph:.2f} > KMAX={KMAX_val}; skipped for Plotly.")
            continue
        if cz is None:  # 如果 Z 未指定，尝试放置在 S2 曲面上
            # 需要插值 S2 的 Z 值
            if "S2" in plot_data["surfaces_data"]:
                kx_s2_flat = plot_data["surfaces_data"]["S2"]["KX"].flatten()
                ky_s2_flat = plot_data["surfaces_data"]["S2"]["KY"].flatten()
                z_s2_flat = plot_data["surfaces_data"]["S2"]["Z"].flatten()
                cz = griddata((kx_s2_flat, ky_s2_flat), z_s2_flat, (cx, cy), method='linear')
                if np.isnan(cz):
                    cz = np.min(z_s2_flat)  # 回退到最小值
            else:
                cz = 0  # 没有S2曲面，默认Z=0

        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode='markers',
            marker=dict(
                size=sph.get("radius", 0.06) * 100,  # 调整尺寸以匹配 Plotly 视觉
                color=sph.get("color", (1.0, 0.4, 0.4)),
                opacity=sph.get("opacity", 0.9),
                symbol='circle'
            ),
            name=f"Sphere {i}"
        ))

    # --- 4. 配置布局和轴 ---
    camera_conf = plot_data["camera_config"]
    # Plotly 相机位置：eye 是相机位置相对于 (0,0,0) 的相对坐标
    # 转换为 Plotly 的 eye.x/y/z 形式
    eye_x = np.cos(np.deg2rad(camera_conf.get("azim", 35))) * np.cos(np.deg2rad(camera_conf.get("elev", 25)))
    eye_y = np.sin(np.deg2rad(camera_conf.get("azim", 35))) * np.cos(np.deg2rad(camera_conf.get("elev", 25)))
    eye_z = np.sin(np.deg2rad(camera_conf.get("elev", 25)))

    fig.update_layout(
        scene=dict(
            xaxis_title=axes_config.get("xtitle", "k_x"),
            yaxis_title=axes_config.get("ytitle", "k_y"),
            zaxis_title=axes_config.get("ztitle", "E"),
            xaxis=dict(
                backgroundcolor=scene_config.get("background_color", "white"),
                gridcolor=axes_config.get("tick_color", "lightgray"),
                showbackground=True,
                zerolinecolor=axes_config.get("tick_color", "gray"),
                tickfont=dict(size=axes_config.get("label_font_size", 12),
                              color=axes_config.get("label_color", "black"),
                              family=axes_config.get("font_family", "serif")),
                title_font=dict(size=axes_config.get("title_font_size", 14),
                                color=axes_config.get("label_color", "black"),
                                family=axes_config.get("font_family", "serif")),
                linecolor=axes_config.get("label_color", "black"),  # 轴线颜色
                linewidth=axes_config.get("line_width", 1.5),  # 轴线宽度
            ),
            yaxis=dict(
                backgroundcolor=scene_config.get("background_color", "white"),
                gridcolor=axes_config.get("tick_color", "lightgray"),
                showbackground=True,
                zerolinecolor=axes_config.get("tick_color", "gray"),
                tickfont=dict(size=axes_config.get("label_font_size", 12),
                              color=axes_config.get("label_color", "black"),
                              family=axes_config.get("font_family", "serif")),
                title_font=dict(size=axes_config.get("title_font_size", 14),
                                color=axes_config.get("label_color", "black"),
                                family=axes_config.get("font_family", "serif")),
                linecolor=axes_config.get("label_color", "black"),
                linewidth=axes_config.get("line_width", 1.5),
            ),
            zaxis=dict(
                backgroundcolor=scene_config.get("background_color", "white"),
                gridcolor=axes_config.get("tick_color", "lightgray"),
                showbackground=True,
                zerolinecolor=axes_config.get("tick_color", "gray"),
                tickfont=dict(size=axes_config.get("label_font_size", 12),
                              color=axes_config.get("label_color", "black"),
                              family=axes_config.get("font_family", "serif")),
                title_font=dict(size=axes_config.get("title_font_size", 14),
                                color=axes_config.get("label_color", "black"),
                                family=axes_config.get("font_family", "serif")),
                linecolor=axes_config.get("label_color", "black"),
                linewidth=axes_config.get("line_width", 1.5),
            ),
            aspectmode='data',  # 保持x,y,z轴的比例
            camera=dict(
                eye=dict(x=eye_x, y=eye_y, z=eye_z),
                up=dict(x=0, y=0, z=1)  # Z轴向上
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        paper_bgcolor=scene_config.get("background_color", "white"),
        plot_bgcolor=scene_config.get("background_color", "white"),
        font=dict(family=axes_config.get("font_family", "serif"),
                  color=axes_config.get("label_color", "black"))
    )

    # 清洁视图设置
    if scene_config.get("clean_view", False):
        fig.update_layout(scene=dict(
            xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False,  # 隐藏网格线
            xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,  # 隐藏背景面板
        ))
    if axes_config.get("labels_off", False) or scene_config.get("clean_view", False):
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False,
                          scene_xaxis_title='', scene_yaxis_title='', scene_zaxis_title='')  # 隐藏标题和刻度标签

    # 保存为静态图片
    output_png_path = output_path_prefix + '_plotly.png'
    try:
        write_image(fig, output_png_path, width=scene_config["window_size"][0], height=scene_config["window_size"][1])
        print(f"[INFO] Plotly plot saved to {output_png_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save Plotly image: {e}. Ensure 'kaleido' is installed (pip install kaleido).")

    # fig.show() # 可取消注释在浏览器中交互式查看


# =========================
# Matplotlib mplot3d 绘图函数 (测试样例)
# =========================
def plot_with_matplotlib_mplot3d(plot_data: Dict, output_path_prefix: str = './band_plot'):
    if plt is None or Axes3D is None:
        return

    print("[INFO] Plotting with Matplotlib mplot3d...")

    plt_axes_config = plot_data["axes_config"]
    scene_config = plot_data["scene_config"]
    camera_conf = plot_data["camera_config"]

    # # 设置全局字体和大小，这样所有 Matplotlib 元素都会遵循
    # plt.rcParams.update({
    #     'font.family': plt_axes_config.get("font_family", "serif"),
    #     'font.size': plt_axes_config.get("label_font_size", 12),
    #     'axes.labelsize': plt_axes_config.get("title_font_size", 14),
    #     'xtick.labelsize': plt_axes_config.get("label_font_size", 12),
    #     'ytick.labelsize': plt_axes_config.get("label_font_size", 12),
    #     'ztick.labelsize': plt_axes_config.get("label_font_size", 12),
    #     'axes.labelcolor': plt_axes_config.get("label_color", "black"),
    #     'xtick.color': plt_axes_config.get("tick_color", "gray"),
    #     'ytick.color': plt_axes_config.get("tick_color", "gray"),
    #     'ztick.color': plt_axes_config.get("tick_color", "gray"),
    # })

    fig = plt.figure(figsize=(scene_config["window_size"][0] / 100, scene_config["window_size"][1] / 100),
                     dpi=100)  # figsize基于像素换算，dpi固定
    ax = fig.add_subplot(111, projection='3d', facecolor=scene_config.get("background_color", "white"))

    # 匹配相机视角
    ax.view_init(elev=camera_conf.get("elev", 25), azim=camera_conf.get("azim", 35), roll=camera_conf.get("roll", 0))

    # --- 1. 绘制曲面 ---
    for name, data in plot_data["surfaces_data"].items():
        sconf = data["sconf"]
        if not sconf.get("visible", True):
            continue
        KX, KY, Z = data["KX"], data["KY"], data["Z"]

        if Z.shape != KX.shape:  # 确保Z的形状与KX, KY匹配
            Z_reshaped = Z.reshape(KX.shape)
        else:
            Z_reshaped = Z

        if sconf.get("mode") == "colormap":
            color_scalars = COLORIZER_REGISTRY[sconf["colorizer"]](KX, KY, Z_reshaped)
            norm = Normalize(vmin=np.nanmin(color_scalars), vmax=np.nanmax(color_scalars))
            cmap = cm.get_cmap(sconf.get("cmap", "viridis"))

            ax.plot_surface(
                KX, KY, Z_reshaped,
                facecolors=cmap(norm(color_scalars)),
                alpha=sconf.get("opacity", 1.0),
                antialiased=True,
                linewidth=0,  # 移除曲面网格线
                shade=True  # 启用着色
            )
            if sconf.get("scalar_bar", False):
                m = cm.ScalarMappable(cmap=cmap, norm=norm)
                m.set_array(color_scalars)
                cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=10)
                cbar.set_label(sconf.get("colorizer", ""), fontsize=plt_axes_config.get("label_font_size", 12))

        elif sconf.get("mode") == "solid":
            ax.plot_surface(
                KX, KY, Z_reshaped,
                color=sconf.get("color", (0.6, 0.6, 0.6)),  # Matplotlib直接支持 RGB tuple
                alpha=sconf.get("opacity", 1.0),
                antialiased=True,
                linewidth=0,
                shade=True
            )
        elif sconf.get("mode") == "wireframe":
            ax.plot_wireframe(
                KX, KY, Z_reshaped,
                color=sconf.get("color", (0.0, 0.0, 0.0)),  # Wireframe模式使用sconf.color作为线颜色
                linewidth=sconf.get("line_width", 1.0),
                alpha=sconf.get("opacity", 1.0)
            )

    # --- 2. 绘制交线 ---
    for inter_data in plot_data["intersections_data"]:
        points = inter_data["points"]
        pconf = inter_data["pconf"]
        if points.shape[0] > 0:
            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=pconf.get("color", "black"),
                linewidth=pconf.get("line_width", 3.0),
                label=inter_data["label"]  # 为交线添加标签
            )

    # --- 3. 绘制球体 ---
    KMAX_val = plot_data["KMAX"]
    for i, sph in enumerate(plot_data["spheres_data"], start=1):
        cx, cy, cz = sph.get("center", (0.0, 0.0, None))
        r_sph = np.sqrt(cx ** 2 + cy ** 2)
        if r_sph > KMAX_val:
            print(f"[WARN] Sphere {i} at r={r_sph:.2f} > KMAX={KMAX_val}; skipped for Matplotlib.")
            continue
        if cz is None:  # 如果 Z 未指定，尝试放置在 S2 曲面上
            if "S2" in plot_data["surfaces_data"]:
                kx_s2_flat = plot_data["surfaces_data"]["S2"]["KX"].flatten()
                ky_s2_flat = plot_data["surfaces_data"]["S2"]["KY"].flatten()
                z_s2_flat = plot_data["surfaces_data"]["S2"]["Z"].flatten()
                cz = griddata((kx_s2_flat, ky_s2_flat), z_s2_flat, (cx, cy), method='linear')
                if np.isnan(cz):
                    cz = np.min(z_s2_flat)
            else:
                cz = 0

        ax.scatter(
            [cx], [cy], [cz],
            s=sph.get("radius", 0.06) * 2000,  # 调整尺寸以匹配 Matplotlib 视觉
            color=sph.get("color", (1.0, 0.4, 0.4)),
            alpha=sph.get("opacity", 0.9),
            marker='o',
            label=f"Sphere {i}"
        )

    # --- 4. 配置轴和背景 ---
    ax.set_xlabel(plt_axes_config.get("xtitle", "k_x"), labelpad=10)
    ax.set_ylabel(plt_axes_config.get("ytitle", "k_y"), labelpad=10)
    ax.set_zlabel(plt_axes_config.get("ztitle", "E"), labelpad=10)

    # 刻度线样式
    ax.tick_params(axis='x', which='major', pad=5, length=plt_axes_config.get("tick_width", 1.5) * 3,
                   width=plt_axes_config.get("tick_width", 1.5), direction='out')
    ax.tick_params(axis='y', which='major', pad=5, length=plt_axes_config.get("tick_width", 1.5) * 3,
                   width=plt_axes_config.get("tick_width", 1.5), direction='out')
    ax.tick_params(axis='z', which='major', pad=5, length=plt_axes_config.get("tick_width", 1.5) * 3,
                   width=plt_axes_config.get("tick_width", 1.5), direction='out')

    # 隐藏刻度标签（如果labels_off为True）
    if plt_axes_config.get("labels_off", False) or scene_config.get("clean_view", False):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # 移除背景网格线和面板 (Matplotlib mplot3d 的 clean_view)
    if scene_config.get("clean_view", False):
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(scene_config.get("background_color", "white"))
        ax.yaxis.pane.set_edgecolor(scene_config.get("background_color", "white"))
        ax.zaxis.pane.set_edgecolor(scene_config.get("background_color", "white"))
    else:
        ax.xaxis.pane.set_edgecolor(plt_axes_config.get("label_color", "black"))  # 显示边框
        ax.yaxis.pane.set_edgecolor(plt_axes_config.get("label_color", "black"))
        ax.zaxis.pane.set_edgecolor(plt_axes_config.get("label_color", "black"))
        ax.xaxis.pane.set_linewidth(plt_axes_config.get("line_width", 1.5))  # 设置边框线宽
        ax.yaxis.pane.set_linewidth(plt_axes_config.get("line_width", 1.5))
        ax.zaxis.pane.set_linewidth(plt_axes_config.get("line_width", 1.5))

    fig.tight_layout()
    output_mpl3d_path = output_path_prefix + '_mpl3d.png'
    plt.savefig(output_mpl3d_path, transparent=True, bbox_inches='tight', dpi=fig.dpi)
    plt.close()
    print(f"[INFO] Matplotlib mplot3d plot saved to {output_mpl3d_path}")


# =========================
# 主执行逻辑
# =========================
def main():
    # 1. 数据准备 (使用 PyVista 的计算逻辑)
    plot_data = prepare_plot_data(CONFIG, SURFACE_FUNCS, KMAX, NGRID)

    # 2. 绘图测试样例
    output_prefix = CONFIG["output"]["output_prefix"]

    # # --- Plotly 绘图 ---
    # if go is not None and write_image is not None:
    #     plot_with_plotly(plot_data, output_path_prefix=output_prefix)

    # --- Matplotlib mplot3d 绘图 ---
    if plt is not None:
        plot_with_matplotlib_mplot3d(plot_data, output_path_prefix=output_prefix)

    print("\n[INFO] Advanced plotting examples finished. Check the output files.")


if __name__ == "__main__":
    main()
