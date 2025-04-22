import numpy as np
import random

from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment


def estimate_derivative_one_sided(Zg, assigned, idx, s, axis, delta):
    """
    在多维数组 Zg 上，沿 axis 方向（整数维度索引）在点 idx 处
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


def group_surfaces_one_sided_hungarian(Z, deltas, lams):
    """
    任意 n 维参数空间的分组算法。
    Z: np.ndarray, dtype=object, shape = dims (tuple of length n),
       每个元素是长度为 m 的 complex 列表。
    deltas: list 或 tuple, 长度 n，每个维度的单元格大小 Δ。
    lams:   list 或 tuple, 长度 n，每个维度的导数连续性权重 λ。
    返回 Zg: 与 Z 相同 shape 和 dtype，每个元素是已排序的长度 m 的 complex np.array。
    """
    dims = Z.shape
    n_dims = len(dims)
    origin = tuple([0] * n_dims)
    m = len(Z[origin])

    # 初始化 Zg 和 assigned 标记
    Zg = np.empty(dims, dtype=object)
    assigned = np.zeros(dims, dtype=bool)

    # 原点按实部排序赋值
    Zg[origin] = np.array(sorted(Z[origin], key=lambda c: c.real), dtype=complex)
    assigned[origin] = True

    # 按 lexicographic 顺序遍历所有格点
    for idx in np.ndindex(*dims):
        if assigned[idx]:
            continue

        # 找一个已赋值的邻点：优先沿各维度的后向
        for axis in range(n_dims):
            if idx[axis] > 0:
                neighbor = list(idx)
                neighbor[axis] -= 1
                neighbor = tuple(neighbor)
                if assigned[neighbor]:
                    break
        else:
            # 没找到已赋值的后向邻点，回退到原点
            neighbor = origin
            axis = 0

        # 上一格的值 (复杂数数组) 和各维度导数 (m×n_dims 矩阵)
        v_prev = Zg[neighbor]
        d_prev = np.zeros((m, n_dims), dtype=float)
        for s in range(m):
            for d in range(n_dims):
                d_prev[s, d] = estimate_derivative_one_sided(
                    Zg, assigned, neighbor, s, d, deltas[d]
                )

        # 构造 m×m 代价矩阵
        candidates = Z[idx]
        C = np.zeros((m, m))
        for s in range(m):
            for c_idx, c in enumerate(candidates):
                cost_val = abs(c - v_prev[s])
                cost_der = sum(
                    lams[d] * abs((c - v_prev[s]) / deltas[d] - d_prev[s, d])
                    for d in range(n_dims)
                )
                C[s, c_idx] = cost_val + cost_der

        # 匈牙利算法匹配
        row_ind, col_ind = linear_sum_assignment(C)

        # 根据匹配结果填充 Zg[idx]
        ordered = [None] * m
        for s, c_idx in zip(row_ind, col_ind):
            ordered[s] = candidates[c_idx]
        Zg[idx] = np.array(ordered, dtype=complex)
        assigned[idx] = True

    return Zg


if __name__ == '__main__':
    # === 3D 测试参数 ===
    nx, ny, nz, m = 30, 20, 15, 3
    dims3 = (nx, ny, nz)
    deltas3 = (1.0, 1.0, 1.0)  # 三个维度的网格间距
    lams3 = (10.0, 5.0, 2.0)  # 三个维度的导数连续性权重

    # 构建三维网格
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    z = np.linspace(0, 2 * np.pi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 生成 m 条“真曲面”：这里取 sin*cos*sin 叠加相位
    # true3.shape = (nx, ny, nz, m)
    true3 = np.stack([
        np.sin(X + s) * np.cos(Y + s) * np.sin(Z + s)
        for s in range(m)
    ], axis=-1)

    # 构造 Z3：object 数组，每个格点存一个乱序的 m 元素列表
    Z3 = np.empty(dims3, dtype=object)
    for idx in np.ndindex(*dims3):
        vals = list(true3[idx])  # 取出长度 m 的 array
        random.shuffle(vals)
        Z3[idx] = vals

    # 调用分组算法
    Zg3 = group_surfaces_one_sided_hungarian(Z3, deltas3, lams3)

    # === 可视化：固定 y、z，沿 x 方向切片 ===
    j0 = ny // 2
    k0 = nz // 2

    plt.figure(figsize=(10, 4))

    # 分组前 (散点)
    plt.subplot(1, 2, 1)
    for i in range(nx):
        for val in Z3[i, j0, k0]:
            plt.scatter(x[i], val.real, s=5, alpha=0.6)
    plt.title(f'分组前 (y={y[j0]:.2f}, z={z[k0]:.2f})')
    plt.xlabel('x');
    plt.ylabel('实部')

    # 分组后 (曲线)
    plt.subplot(1, 2, 2)
    for s in range(m):
        y_s = [Zg3[i, j0, k0][s].real for i in range(nx)]
        plt.plot(x, y_s, label=f'曲面{s}')
    plt.title(f'分组后 (y={y[j0]:.2f}, z={z[k0]:.2f})')
    plt.xlabel('x');
    plt.legend()

    plt.tight_layout()
    plt.show()
