import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

def estimate_derivative_one_sided(Zg, assigned, idx, s, axis, delta):
    """
    在多维数组 Zg 上，沿 axis 方向在点 idx 处
    用单边差分估计第 s 条曲面的偏导数。
    优先使用已赋值的后向差分，不可用时使用已赋值的前向差分；
    若都不可用，返回 0.
    """
    idx_list = list(idx)

    # 后向差分
    if idx[axis] - 1 >= 0:
        idx_list_b = idx_list.copy()
        idx_list_b[axis] -= 1
        nb = tuple(idx_list_b)
        if assigned[nb]:
            return (Zg[idx][s] - Zg[nb][s]) / delta

    # 前向差分
    if idx[axis] + 1 < Zg.shape[axis]:
        idx_list_f = idx_list.copy()
        idx_list_f[axis] += 1
        nf = tuple(idx_list_f)
        if assigned[nf]:
            return (Zg[nf][s] - Zg[idx][s]) / delta

    # 都不可用
    return 0.0

def group_surfaces_one_sided_hungarian(
    Z,
    deltas,
    value_weights,
    deriv_weights,
    initial_derivatives=None
):
    """
    任意 n 维参数空间的分组算法，可用矩阵指定不同生长方向上的
    值连续性权重（value_weights）与导数连续性权重（deriv_weights）。

    参数
    ----
    Z : np.ndarray, dtype=object, shape=dims
        每个格点存放长度为 m 的 complex 列表（乱序）。
    deltas : sequence of float, length = n_dims
        各维度的网格步长 Δ_d。
    value_weights : np.ndarray, shape=(n_dims, n_dims)
        生长方向 d（行）对应的“值差”权重向量，列索引 j。
    deriv_weights : np.ndarray, shape=(n_dims, n_dims)
        生长方向 d（行）对应的“导数不连续”权重向量，列索引 j。
    initial_derivatives : optional np.ndarray, shape=(m, n_dims)
        原点处 m 条曲面的初始偏导数猜测值。

    返回
    ----
    Zg : np.ndarray, same shape and dtype as Z
        每个点存放已排序（按曲面索引）长度 m 的 complex np.array。
    """
    dims    = Z.shape
    n_dims  = len(dims)
    origin  = tuple([0] * n_dims)
    # m       = len(Z[origin])
    m       = min(len(Z[idx]) for idx in np.ndindex(*dims))  # 对于长度不齐的列表数据, 选取最保守的处理数量

    # 输出与标记数组
    Zg       = np.empty(dims, dtype=object)
    assigned = np.zeros(dims, dtype=bool)

    # 原点按实部排序
    Zg[origin]   = np.array(sorted(Z[origin], key=lambda c: c.real), dtype=complex)
    assigned[origin] = True

    # 按字典序遍历
    for idx in np.ndindex(*dims):
        if assigned[idx]:
            continue

        # 收集所有已赋值的后向邻点
        neighbors = []
        for d in range(n_dims):
            if idx[d] > 0:
                nb = list(idx); nb[d] -= 1; nb = tuple(nb)
                if assigned[nb]:
                    neighbors.append((nb, d))
        if not neighbors:
            neighbors = [(origin, 0)]

        # 候选集合
        candidates = Z[idx]
        C_total    = np.zeros((m, m), dtype=float)

        # 针对每个邻点累加代价
        for neighbor, grow_dir in neighbors:
            v_prev = Zg[neighbor]

            # 若是原点且提供了初始导数，则用之
            if neighbor == origin and initial_derivatives is not None:
                d_prev = initial_derivatives.copy()
            else:
                d_prev = np.zeros((m, n_dims), dtype=float)
                for s in range(m):
                    for d in range(n_dims):
                        d_prev[s, d] = estimate_derivative_one_sided(
                            Zg, assigned, neighbor, s, d, deltas[d]
                        )

            # 构造该邻点的局部 C 矩阵
            C_nb = np.zeros((m, m), dtype=float)
            for s in range(m):
                for c_idx, c in enumerate(candidates):
                    if c_idx >= m:
                        break
                    # 值连续性（按 value_weights[grow_dir, :] 加权）
                    cost_val = sum(
                        value_weights[grow_dir, j] * abs(c - v_prev[s])
                        for j in range(n_dims)
                    )
                    # 导数连续性（按 deriv_weights[grow_dir, :] 加权）
                    cost_der = sum(
                        deriv_weights[grow_dir, j]
                        * abs((c - v_prev[s]) / deltas[j] - d_prev[s, j])
                        for j in range(n_dims)
                    )
                    C_nb[s, c_idx] = cost_val + cost_der

            C_total += C_nb

        # 全局最优指派
        row_ind, col_ind = linear_sum_assignment(C_total)

        # 填入 Zg
        ordered = [None] * m
        for s, c_idx in zip(row_ind, col_ind):
            # ordered[s] = candidates[min(c_idx, len(candidates)-1)]
            ordered[s] = candidates[c_idx]
        Zg[idx] = np.array(ordered, dtype=complex)
        assigned[idx] = True

    return Zg

if __name__ == '__main__':
    # === 3D 示例 ===
    nx, ny, nz, m = 30, 20, 15, 3
    dims3 = (nx, ny, nz)
    deltas3 = (1.0, 1.0, 1.0)

    # 当沿维度 d 生长时，值差权重矩阵（n×n）
    # 例如：value_weights[d, j] = 在 grow_dir=d 时，对维度 j 的值差权重
    value_weights = np.array([
        [1.0, 0.5, 0.2],   # 沿维度0生长时，对 0,1,2 维度的值差权重
        [0.5, 1.0, 0.3],   # 沿维度1生长时
        [0.2, 0.3, 1.0],   # 沿维度2生长时
    ])

    # 当沿维度 d 生长时，导数不连续权重矩阵（n×n）
    deriv_weights = np.array([
        [10.0, 2.0, 1.0],
        [2.0, 5.0, 1.5],
        [1.0, 1.5, 2.0],
    ])

    # 可选：原点初始导数（m×n）
    initial_derivs = np.zeros((m, len(deltas3)))

    # 构建格点与真曲面
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    z = np.linspace(0, 2*np.pi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    true3 = np.stack([
        np.sin(X + s) * np.cos(Y + s) * np.sin(Z + s)
        for s in range(m)
    ], axis=-1)

    # 乱序数据初始化
    Z3 = np.empty(dims3, dtype=object)
    for idx in np.ndindex(*dims3):
        vals = list(true3[idx])
        random.shuffle(vals)
        Z3[idx] = vals

    # 调用分组，传入两个权重矩阵
    Zg3 = group_surfaces_one_sided_hungarian(
        Z3, deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        initial_derivatives=initial_derivs
    )

    # 可视化：固定 y,z
    j0, k0 = ny//2, nz//2
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    for i in range(nx):
        for v in Z3[i, j0, k0]:
            plt.scatter(x[i], v.real, s=5, alpha=0.6)
    plt.title(f'分组前 (y={y[j0]:.2f}, z={z[k0]:.2f})')
    plt.xlabel('x'); plt.ylabel('实部')

    plt.subplot(1,2,2)
    for s in range(m):
        ys = [Zg3[i, j0, k0][s].real for i in range(nx)]
        plt.plot(x, ys, label=f'曲面{s}')
    plt.title(f'分组后 (y={y[j0]:.2f}, z={z[k0]:.2f})')
    plt.xlabel('x'); plt.legend()

    plt.tight_layout()
    plt.show()
