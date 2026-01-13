import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


# -----------------------------
# 1. 极速计算内核 (Vectorized)
# -----------------------------
def compute_volume_fast(Nx=100, Ny=100, Nz=100):
    """
    生成 3D 体数据 dmin(k, w3, s)
    分辨率: Nx(k), Ny(w3), Nz(s)
    """
    # 物理参数
    omega1, omega2 = -0.2, 0.2
    # omega1, omega2 = -0.1, 0.1
    gamma_a, gamma_r = 0.5, 0.0

    # 坐标范围
    # 注意：为了符合 numpy 的 (z, y, x) 存储习惯，我们将 S 放在最外层循环(或者第一个维度)
    # 这里定义 Grid 维度顺序为: (S, w3, k)
    s_vals = np.linspace(0.0, 0.5, Nz).astype(np.float32)
    w3_vals = np.linspace(-0.5, 0.5, Ny).astype(np.float32)
    k_vals = np.linspace(0.0, 1.0, Nx).astype(np.float32)

    # 预分配体积数据 (S, w3, k)
    volume = np.zeros((Nz, Ny, Nx), dtype=np.float32)

    print(f"Computing 3D volume ({Nz}x{Ny}x{Nx})...")

    # 循环 S 切片 (比完全 3D 广播更省内存，且速度极快)
    for i, s in enumerate(s_vals):
        # 构建 2D 网格
        K, W3 = np.meshgrid(k_vals, w3_vals)  # Shape: (Ny, Nx)

        # 耦合项
        g1 = K * np.sin(2 * np.pi * s)
        g2 = K * np.cos(2 * np.pi * s)

        # 矩阵对角元
        # d1 = -0.2 - 0.5j
        # d2 =  0.2 - 0.5j
        d1 = complex(omega1, -(gamma_a + gamma_r))
        d2 = complex(omega2, -gamma_a)

        # 构造批量矩阵 H (Ny, Nx, 3, 3)
        # 这是一个小技巧：手动算特征值比调 eigvals 更快吗？
        # 对于 numpy，直接调 eigvals 已经经过高度优化。

        H = np.zeros((Ny, Nx, 3, 3), dtype=np.complex64)
        H[..., 0, 0] = d1
        H[..., 1, 1] = d2
        H[..., 2, 2] = W3
        H[..., 0, 2] = g1;
        H[..., 2, 0] = g1
        H[..., 1, 2] = g2;
        H[..., 2, 1] = g2

        # 计算特征值 (Ny, Nx, 3)
        evals = np.linalg.eigvals(H)

        # 计算差值
        d12 = np.abs(evals[..., 0] - evals[..., 1])
        d13 = np.abs(evals[..., 0] - evals[..., 2])
        d23 = np.abs(evals[..., 1] - evals[..., 2])

        # 取最小能隙并存入体数据
        volume[i, :, :] = np.minimum(np.minimum(d12, d13), d23)

    return volume, (k_vals, w3_vals, s_vals)


# -----------------------------
# 2. 等值面提取与绘制
# -----------------------------
def plot_isosurface_tubes():
    # 1. 计算数据
    # 分辨率设为 100x100x100 既快又平滑
    vol, (k_ax, w3_ax, s_ax) = compute_volume_fast(Nx=200, Ny=200, Nz=200)

    # 2. Marching Cubes 提取等值面
    level = 0.1
    print(f"Extracting isosurface at level={level}...")

    # verts 返回的是索引坐标 (s_idx, w3_idx, k_idx)
    try:
        verts, faces, normals, values = measure.marching_cubes(vol, level)
    except ValueError:
        print("未找到等值面！请检查阈值 level 是否过小。")
        return

    # 3. 将索引坐标转换为物理坐标
    # verts[:, 0]对应 S轴(Nz), verts[:, 1]对应 w3轴(Ny), verts[:, 2]对应 k轴(Nx)

    # 缩放因子
    scale_s = (s_ax[-1] - s_ax[0]) / (len(s_ax) - 1)
    scale_w3 = (w3_ax[-1] - w3_ax[0]) / (len(w3_ax) - 1)
    scale_k = (k_ax[-1] - k_ax[0]) / (len(k_ax) - 1)

    # 物理坐标
    real_s = verts[:, 0] * scale_s + s_ax[0]
    real_w3 = verts[:, 1] * scale_w3 + w3_ax[0]
    real_k = verts[:, 2] * scale_k + k_ax[0]

    # 4. 绘图
    plt.rcParams['font.size'] = 9
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Arial'

    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_subplot(111, projection='3d')

    # 创建 3D网格对象
    # 注意：matplotlib 的 plot_trisurf 也可以，但 Poly3DCollection 控制力更强
    # 我们需要构建 vertex 列表：[ [x1,y1,z1], [x2,y2,z2], ... ]

    # 映射到绘图轴：X=k, Y=S, Z=w3
    # vertices shape: (N_verts, 3) -> (k, S, w3)
    mesh_verts = np.column_stack([real_k, real_s, real_w3])

    # 生成面列表
    poly3d = [[mesh_verts[vert_id] for vert_id in face] for face in faces]

    # --- 渐变着色核心 ---
    # 我们计算每个三角形中心的 S 坐标 (Y轴)，根据它来决定颜色
    # face_centers_s = np.mean([v[1] for v in poly3d], axis=1) # 太慢
    # 快速方法：取三个顶点的 S 平均值
    face_s = np.mean(real_s[faces], axis=1)

    # 归一化颜色
    from matplotlib import cm
    norm = plt.Normalize(vmin=0, vmax=0.5)
    colors = cm.hsv(norm(face_s))

    # 创建集合对象
    mesh = Poly3DCollection(poly3d, alpha=0.8)
    mesh.set_facecolor(colors)
    mesh.set_edgecolor('none')  # 去掉网格线，看起来更像光滑管子

    ax.add_collection3d(mesh)

    # # 设置坐标轴范围
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_zlim(-0.5, 0.5)

    # ax.set_xlabel(r'$\kappa$')
    # ax.set_ylabel(r'$S$')
    # ax.set_zlabel(r'$\omega_3$')
    # ax.set_title(f'EP Tubes ($d_{{min}}={level}$)')

    plt.savefig("ep_tubes_isosurface_marked.png", dpi=500, bbox_inches='tight', transparent=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # set box aspect ratio
    ax.set_box_aspect([1, 1.5, 1])  # x:y:z ratio
    # 去掉背景网格线和背景色
    ax.grid(False)
    ax.set_facecolor('white')

    # 调整视角
    ax.view_init(elev=20, azim=-30)

    # # 添加 Colorbar
    # mappable = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    # mappable.set_array([])
    # plt.colorbar(mappable, ax=ax, shrink=0.6, label='Parameter S (Evolution)')

    plt.savefig("ep_tubes_isosurface.png", dpi=500, bbox_inches='tight', transparent=True)
    plt.savefig("ep_tubes_isosurface.svg", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == "__main__":
    plot_isosurface_tubes()
