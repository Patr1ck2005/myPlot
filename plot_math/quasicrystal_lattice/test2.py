import numpy as np
import matplotlib.pyplot as plt
import itertools


def generate_12_fold_quasicrystal(L_superspace=3, acceptance_window_width=1.0):
    """
    使用切片与投影法生成十二重对称的准晶晶格点。

    参数:
    L_superspace (int): 在6维超空间中整数坐标的搜索范围。例如，L=3表示每个坐标从-3到3。
                        L越大，生成的点越多，覆盖范围越大，计算时间越长。
    acceptance_window_width (float): 在4维垂直空间中的接受窗口的宽度。
                                     W越大，生成的点越密集。

    返回:
    numpy.ndarray: 一个(N, 2)的数组，包含准晶格点的(x, y)坐标。
    """
    N_FOLD = 12
    SUP_DIM = 6  # 超空间维度，对于12重对称，通常使用6维

    # 1. 定义物理空间和垂直空间的投影基向量
    # 对于12重对称，使用6个基向量，角度间隔 pi/6 (30度)
    thetas = np.array([k * np.pi / 6 for k in range(SUP_DIM)])

    # P_parallel (2x6 矩阵): 投影到2维物理空间
    # 其列向量是 (cos(theta_k), sin(theta_k))
    P_parallel = np.array([
        np.cos(thetas),
        np.sin(thetas)
    ])

    # P_perp (4x6 矩阵): 投影到4维垂直空间
    # 这里的构造方式是确保其与P_parallel正交，并能生成准晶结构
    # 常见的构造方法是使用不同的相位偏移
    P_perp = np.zeros((SUP_DIM - 2, SUP_DIM))  # 4x6 矩阵
    # 垂直空间基向量1：相位偏移 pi/3
    P_perp[0, :] = np.cos(thetas + np.pi / 3)
    P_perp[1, :] = np.sin(thetas + np.pi / 3)
    # 垂直空间基向量2：相位偏移 2*pi/3 (或 pi/6 等其他值，取决于具体数学模型)
    P_perp[2, :] = np.cos(thetas + 2 * np.pi / 3)
    P_perp[3, :] = np.sin(thetas + 2 * np.pi / 3)

    print(f"正在生成 {SUP_DIM} 维晶格点 (每个维度范围: [{-L_superspace}, {L_superspace}])...")
    # 2. 生成高维晶格点 (使用 itertools.product 效率更高)
    # 坐标范围 [-L, L]
    coords_range = range(-L_superspace, L_superspace + 1)
    # 生成所有6维整数坐标的组合 (这是计算量最大的部分)
    all_6d_coords = np.array(list(itertools.product(coords_range, repeat=SUP_DIM)))
    print(f"共生成 {len(all_6d_coords)} 个 {SUP_DIM} 维候选点。")

    # 3. 投影到物理空间和垂直空间
    # 矩阵乘法：(N_points, SUP_DIM) @ (SUP_DIM, 2) -> (N_points, 2)
    points_parallel = all_6d_coords @ P_parallel.T
    # 矩阵乘法：(N_points, SUP_DIM) @ (SUP_DIM, 4) -> (N_points, 4)
    points_perp = all_6d_coords @ P_perp.T

    # 4. 应用接受窗口
    # 检查垂直空间中的投影点是否落在 [-W/2, W/2]^4 的超立方体内
    # np.all(..., axis=1) 确保所有4个垂直维度的坐标都在窗口内
    mask = np.all(np.abs(points_perp) <= acceptance_window_width / 2, axis=1)

    quasicrystal_points = points_parallel[mask]
    print(f"经过筛选，共获得 {len(quasicrystal_points)} 个准晶格点。")

    return quasicrystal_points


def visualize_quasicrystal(points, title="12-Fold Quasicrystal Lattice (Cn,n=12)"):
    """
    可视化生成的准晶格点。

    参数:
    points (numpy.ndarray): 包含准晶格点(x, y)坐标的数组。
    title (str): 图表的标题。
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.8, c='blue', marker='.')
    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.gca().set_aspect('equal', adjustable='box')  # 保持X和Y轴比例一致，避免变形
    plt.grid(False)  # 通常准晶不显示网格
    plt.axis('off')  # 隐藏坐标轴，更美观
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    # 参数设置
    L_superspace = 3  # 6维超空间整数坐标的搜索范围 [-L, L]
    # L=2 -> 5^6 = 15625 候选点
    # L=3 -> 7^6 = 117649 候选点
    # L=4 -> 9^6 = 531441 候选点 (计算会慢一些)
    acceptance_window_width = 1.0  # 垂直空间接受窗口宽度

    print("开始生成 Cn,n=12 准晶晶格...")
    quasicrystal_points = generate_12_fold_quasicrystal(L_superspace, acceptance_window_width)

    print("\n开始可视化准晶晶格...")
    if len(quasicrystal_points) > 0:
        visualize_quasicrystal(quasicrystal_points,
                               title=f"12-Fold Quasicrystal (Cut-and-Project Method)\n"
                                     f"L={L_superspace}, Window Width={acceptance_window_width}")
    else:
        print("未生成任何准晶格点，请尝试调整参数 L_superspace 或 acceptance_window_width。")

