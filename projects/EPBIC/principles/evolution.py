import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import scipy.ndimage as ndimage
from scipy.interpolate import make_interp_spline
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. 计算内核 & 2. 谷值提取 (保持不变)
# -----------------------------
# (此处省略 compute_maps_core 和 find_valleys 函数，直接复用你之前的代码)
# 请确保 compute_maps_core 和 find_valleys 已经定义

def compute_maps_core(S, omega1, omega2, gamma_a, gamma_r, K_grid, W3_grid, dtype=np.complex64):
    # ... (与你上一段代码完全一致) ...
    # 为了完整性简单重复关键部分
    s_val = np.float32(np.sin(2.0 * np.pi * S))
    c_val = np.float32(np.cos(2.0 * np.pi * S))
    g1 = K_grid * s_val;
    g2 = K_grid * c_val
    d1 = np.complex64(omega1) - 1j * np.complex64(gamma_a + gamma_r)
    d2 = np.complex64(omega2) - 1j * np.complex64(gamma_a)
    d3 = W3_grid.astype(np.float32).astype(np.complex64)
    Ny, Nx = K_grid.shape
    H = np.zeros((Ny, Nx, 3, 3), dtype=dtype)
    H[..., 0, 0] = d1;
    H[..., 1, 1] = d2;
    H[..., 2, 2] = d3
    H[..., 0, 2] = g1;
    H[..., 2, 0] = g1
    H[..., 1, 2] = g2;
    H[..., 2, 1] = g2
    lam = np.linalg.eigvals(H)
    d12 = np.abs(lam[..., 0] - lam[..., 1])
    d13 = np.abs(lam[..., 0] - lam[..., 2])
    d23 = np.abs(lam[..., 1] - lam[..., 2])
    dmax = np.maximum(np.maximum(d12, d13), d23).astype(np.float32)
    dmin = np.minimum(np.minimum(d12, d13), d23).astype(np.float32)
    return dmax, dmin


def find_valleys(data, threshold_abs=None, neighborhood=3):
    local_min_filter = ndimage.minimum_filter(data, size=neighborhood)
    is_local_min = (data == local_min_filter)
    if threshold_abs is not None:
        is_local_min = is_local_min & (data < threshold_abs)
    y_idx, x_idx = np.where(is_local_min)
    return y_idx, x_idx, data[y_idx, x_idx]


# -----------------------------
# 3. 轨迹处理核心 (新增功能)
# -----------------------------
def process_trajectories(points_data):
    """
    输入: points_data shape (N, 4) -> [k, w3, s, val]
    输出: 一个列表，列表包含若干个字典，每个字典代表一条平滑曲线 {'k', 'w3', 's', 'val'}
    """
    if len(points_data) == 0:
        return []

    # 1. 提取用于聚类的空间坐标 (k, w3, s)
    # 注意: S 的范围通常比 k, w3 小，为了让 DBSCAN 认为 S 方向是连续的，
    # 我们通常不需要特别归一化，或者根据实际跨度调整权重。
    X = points_data[:, [0, 1, 2]]

    # 2. 使用 DBSCAN 聚类
    # eps: 邻域半径。需要根据你的 S_steps 和网格密度调整。
    #      如果 S 步长是 0.5/80 = 0.006，eps 设为 0.04 左右可以连接相邻切片的点。
    # min_samples: 构成核心点的最少样本数，设小一点以允许细线。
    clustering = DBSCAN(eps=0.06, min_samples=3).fit(X)
    labels = clustering.labels_

    unique_labels = set(labels)
    curves = []

    print(f"Cluster found: {len(unique_labels) - (1 if -1 in unique_labels else 0)} distinct curves.")

    for k_label in unique_labels:
        if k_label == -1: continue  # -1 是噪声点

        # 提取属于该类的点
        mask = (labels == k_label)
        cluster_points = points_data[mask]

        # 3. 排序: 沿着 S (第3列) 排序
        # 这一步至关重要，否则线是乱连的
        sort_idx = np.argsort(cluster_points[:, 2])
        cluster_points = cluster_points[sort_idx]

        # 4. 去重 (可选): 如果同一 S 切片有多个点（可能是网格导致的双像素），取平均
        # 这里用简易方法：直接用原始点，依靠 Spline 平滑

        # 5. B-Spline 平滑插值
        # 只有点数足够多才插值
        if len(cluster_points) > 5:
            s_raw = cluster_points[:, 2]
            # 为了平滑，生成更密的 S 轴
            s_smooth = np.linspace(s_raw.min(), s_raw.max(), 300)

            # 对 k, w3, val 分别进行样条插值
            # k=3 (三次样条), bc_type=natural 防止端点飞出
            try:
                spl_k = make_interp_spline(s_raw, cluster_points[:, 0], k=3)
                spl_w3 = make_interp_spline(s_raw, cluster_points[:, 1], k=3)
                spl_val = make_interp_spline(s_raw, cluster_points[:, 3], k=3)

                k_smooth = spl_k(s_smooth)
                w3_smooth = spl_w3(s_smooth)
                val_smooth = spl_val(s_smooth)

                curves.append({
                    'k': k_smooth,
                    'w3': w3_smooth,
                    's': s_smooth,
                    'val': val_smooth
                })
            except Exception as e:
                # 如果插值失败（比如点太少或非单调），回退到原始数据
                curves.append({
                    'k': cluster_points[:, 0],
                    'w3': cluster_points[:, 1],
                    's': cluster_points[:, 2],
                    'val': cluster_points[:, 3]
                })
        else:
            curves.append({
                'k': cluster_points[:, 0],
                'w3': cluster_points[:, 1],
                's': cluster_points[:, 2],
                'val': cluster_points[:, 3]
            })

    return curves


def plot_gradient_line(ax, k, s, w3, val, cmap='jet', lw=2, zorder=1):
    """
    在 3D 轴上绘制随 val 变色的渐变线
    """
    # 构造点集 (N, 3)
    points = np.array([k, s, w3]).T.reshape(-1, 1, 3)
    # 构造线段 (N-1, 2, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 创建集合
    lc = Line3DCollection(segments, cmap=cmap, norm=plt.Normalize(0, 0.5), zorder=zorder)
    lc.set_array(val)
    lc.set_linewidth(lw)

    ax.add_collection(lc)
    return lc


# -----------------------------
# 4. 主流程
# -----------------------------
def analyze_and_trace():
    Nx, Ny = 512, 512  # 网格分辨率
    S_steps = 100*2  # 增加 S 切片数以获得更好的连续性

    # 物理参数
    omega1, omega2 = -0.2, 0.2
    gamma_a, gamma_r = 0.5, 0.0

    k_vec = np.linspace(0, 1, Nx, dtype=np.float32)
    w3_vec = np.linspace(-.5, .5, Ny, dtype=np.float32)
    K, W3 = np.meshgrid(k_vec, w3_vec)
    S_vec = np.linspace(0, 0.5, S_steps)

    # --- 数据采集 ---
    dmin_points = []

    print("Scanning and extracting points...")
    for s_val in S_vec:
        _, dmin = compute_maps_core(s_val, omega1, omega2, gamma_a, gamma_r, K, W3)

        # 阈值设得稍微宽松一点，确保连续性，DBSCAN 会过滤掉离群点
        y, x, v = find_valleys(dmin, threshold_abs=0.08, neighborhood=3)
        for _y, _x, _v in zip(y, x, v):
            dmin_points.append([k_vec[_x], w3_vec[_y], s_val, _v])

    dmin_data = np.array(dmin_points)

    if len(dmin_data) == 0:
        print("No points found!")
        return

    # --- 聚类与平滑 ---
    print("Clustering and smoothing curves...")
    curves = process_trajectories(dmin_data)

    # --- 绘图 ---
    plt.rcParams['font.size'] = 9
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Arial'

    fig = plt.figure(figsize=(2*3, 2))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制每一条识别出的曲线
    line_collection = None
    for curve in curves:
        # 注意：matplotlib 3D plot 参数顺序通常是 (x, y, z)
        # 这里我们将 S 放在 y 轴位置，方便观察演化: x=kappa, y=S, z=omega3
        # lc = plot_gradient_line(ax, curve['k'], curve['s'], curve['w3'], curve['val'], cmap='jet_r', lw=3)
        lc = plot_gradient_line(ax, curve['k'], curve['s'], curve['w3'], curve['s'], cmap='rainbow', lw=3)
        line_collection = lc

    # 装饰
    ax.set_xlabel(r'$\kappa$')
    ax.set_ylabel(r'$S$')
    ax.set_zlabel(r'$\omega_3$')
    # 去掉背景网格线和背景色
    ax.grid(False)
    ax.set_facecolor('white')
    # 设置tick参数
    ax.tick_params(axis='x', which='both', direction='in', pad=0.1)
    ax.tick_params(axis='y', which='both', direction='in', pad=0.1)
    ax.tick_params(axis='z', which='both', direction='in', pad=0.1)
    # ax.set_title("Continuous Evolution of Exceptional Points (Smoothed)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_zlim(-0.5, 0.5)

    # set box aspect ratio
    ax.set_box_aspect([1, 5, 1])  # x:y:z ratio

    # # 添加 Colorbar
    # if line_collection:
    #     cbar = plt.colorbar(line_collection, ax=ax, shrink=0.6, pad=0.1)
    #     cbar.set_label(r'Gap Size ($d_{min}$)')

    # 调整视角
    ax.view_init(elev=30, azim=-10)

    plt.savefig("ep_trajectories_3d.svg", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == "__main__":
    analyze_and_trace()
