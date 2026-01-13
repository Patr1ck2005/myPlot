import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from skimage import measure


# -----------------------------
# 1. 2D 切片计算内核
# -----------------------------
def compute_dmin_slice(S, Nx=200, Ny=200):
    """
    计算特定 S 下的 dmin(k, w3) 2D 网格
    """
    # 物理参数
    omega1, omega2 = -0.2, 0.2
    gamma_a, gamma_r = 0.5, 0.0
    k_min, k_max = 0.0, 1.0
    w3_min, w3_max = -0.5, 0.5

    # 生成网格
    k_vec = np.linspace(k_min, k_max, Nx).astype(np.float32)
    w3_vec = np.linspace(w3_min, w3_max, Ny).astype(np.float32)
    K, W3 = np.meshgrid(k_vec, w3_vec)

    # 耦合项
    g1 = K * np.sin(2 * np.pi * S)
    g2 = K * np.cos(2 * np.pi * S)

    # 矩阵构造
    d1 = complex(omega1, -(gamma_a + gamma_r))
    d2 = complex(omega2, -gamma_a)

    # 批量计算特征值
    H = np.zeros((Ny, Nx, 3, 3), dtype=np.complex64)
    H[..., 0, 0] = d1
    H[..., 1, 1] = d2
    H[..., 2, 2] = W3
    H[..., 0, 2] = g1;
    H[..., 2, 0] = g1
    H[..., 1, 2] = g2;
    H[..., 2, 1] = g2

    evals = np.linalg.eigvals(H)

    d12 = np.abs(evals[..., 0] - evals[..., 1])
    d13 = np.abs(evals[..., 0] - evals[..., 2])
    d23 = np.abs(evals[..., 1] - evals[..., 2])

    dmin = np.minimum(np.minimum(d12, d13), d23)

    return dmin, (k_vec, w3_vec)


# -----------------------------
# 2. 扫描 S 并提取轮廓
# -----------------------------
def main():
    # 参数设置
    S_steps = 60  # 切片数量 (越多越密集，看起来越像管子)
    Nx, Ny = 200, 200  # 2D 分辨率
    level = 0.1  # 等高线阈值 (管子半径)

    s_values = np.linspace(0, 0.5, S_steps)

    # 存储所有的线段用于绘图
    # 格式: List of (N, 3) arrays
    segments = []
    s_color_vals = []  # 用于颜色映射

    print(f"Start scanning {S_steps} slices...")

    for s_val in s_values:
        # 1. 计算当前切片
        dmin_grid, (k_ax, w3_ax) = compute_dmin_slice(s_val, Nx, Ny)

        # 2. 提取等高线 (skimage.measure.find_contours)
        # 返回的是 list of (row, col) 坐标
        contours = measure.find_contours(dmin_grid, level)

        # 3. 坐标变换并收集
        for contour in contours:
            # contour[:, 0] 是 row (y轴索引 -> w3)
            # contour[:, 1] 是 col (x轴索引 -> k)

            # 线性插值映射回物理坐标
            # row_idx -> w3
            r_idx = contour[:, 0]
            c_idx = contour[:, 1]

            w3_coords = w3_ax[0] + (r_idx / (Ny - 1)) * (w3_ax[-1] - w3_ax[0])
            k_coords = k_ax[0] + (c_idx / (Nx - 1)) * (k_ax[-1] - k_ax[0])

            # 构造 3D 线条: (x=k, y=S, z=w3)
            # 每一条线上的点的 S 都是常数 s_val
            s_coords = np.full_like(k_coords, s_val)

            # 组合成 (N, 3) 数组
            line_3d = np.column_stack([k_coords, s_coords, w3_coords])

            segments.append(line_3d)
            s_color_vals.append(s_val)  # 这条线的颜色由 S 决定

    print(f"Extraction done. Total contour segments: {len(segments)}")

    # -----------------------------
    # 3. 3D 绘图 (Line3DCollection)
    # -----------------------------
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 创建线集合
    lc = Line3DCollection(segments, cmap='jet', linewidths=1.5, alpha=0.8)

    # 设置颜色映射依据 (这里根据 S 值变色)
    lc.set_array(np.array(s_color_vals))
    lc.set_clim(0, 0.5)

    ax.add_collection(lc)

    # 设置坐标轴范围 (必须手动设置，因为 add_collection 不会自动更新 limits)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_zlim(-0.5, 0.5)

    ax.set_xlabel(r'$\kappa$')
    ax.set_ylabel(r'$S$ (Slice Axis)')
    ax.set_zlabel(r'$\omega_3$')

    ax.set_title(f'Stacked Contours of EPs ($d_{{min}}={level}$)')

    # 视角调整
    ax.view_init(elev=20, azim=-60)

    # Colorbar
    cbar = plt.colorbar(lc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(r'$S$ Value')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
