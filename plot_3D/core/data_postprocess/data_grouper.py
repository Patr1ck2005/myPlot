import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment


def estimate_derivative_one_sided(Zg, assigned, idx, s, axis, delta):
    """
    在多维数组 Zg 上，沿 axis 方向在点 idx 处
    用单边差分估计第 s 条曲面的偏导数。
    优先使用已赋值的后向差分，不可用时使用已赋值的前向差分；
    若都不可用，返回 0.0。
    如果 Zg[idx][s] 或邻居点的值为 None/NaN，则无法计算导数，返回 0.0。
    """
    idx_list = list(idx)

    # 检查当前点的值是否存在
    if Zg[idx] is None or Zg[idx][s] is None or np.isnan(Zg[idx][s]):
        return 0.0

    # 后向差分
    if idx[axis] - 1 >= 0:
        idx_list_b = idx_list.copy()
        idx_list_b[axis] -= 1
        nb = tuple(idx_list_b)
        # 检查邻点是否已赋值且值存在
        if assigned[nb] and Zg[nb] is not None and Zg[nb][s] is not None and not np.isnan(Zg[nb][s]):
            return (Zg[idx][s] - Zg[nb][s]) / delta

    # 前向差分
    if idx[axis] + 1 < Zg.shape[axis]:
        idx_list_f = idx_list.copy()
        idx_list_f[axis] += 1
        nf = tuple(idx_list_f)
        # 检查邻点是否已赋值且值存在
        if assigned[nf] and Zg[nf] is not None and Zg[nf][s] is not None and not np.isnan(Zg[nf][s]):
            return (Zg[nf][s] - Zg[idx][s]) / delta

    # 都不可用，或者值缺失
    return 0.0


def group_surfaces_one_sided_hungarian(
        Z,
        deltas,
        value_weights,
        deriv_weights,
        max_m=None,  # 新增参数：显式指定要追踪的最大曲面数量
        initial_derivatives=None,
        nan_cost_penalty=1e9  # 新增参数：NaN曲面与真实值匹配的惩罚
):
    """
    任意 n 维参数空间的分组算法，可用矩阵指定不同生长方向上的
    值连续性权重（value_weights）与导数连续性权重（deriv_weights）。
    此版本可处理每个格点处 Z 列表中元素数量不一致的情况。

    参数
    ----
    Z : np.ndarray, dtype=object, shape=dims
        每个格点存放长度可变的 complex 列表（乱序）。
    deltas : sequence of float, length = n_dims
        各维度的网格步长 Δ_d。
    value_weights : np.ndarray, shape=(n_dims, n_dims)
        生长方向 d（行）对应的“值差”权重向量，列索引 j。
    deriv_weights : np.ndarray, shape=(n_dims, n_dims)
        生长方向 d（行）对应的“导数不连续”权重向量，列索引 j。
    max_m : int, optional
        要追踪的最大曲面数量。如果未指定，则取所有格点中列表长度的最大值。
    initial_derivatives : optional np.ndarray, shape=(max_m, n_dims)
        原点处 max_m 条曲面的初始偏导数猜测值。
    nan_cost_penalty : float, optional
        当一个曲面在前一个点是 np.nan 时，它与当前点真实候选值匹配的惩罚。
        此值应足够大，以避免随意匹配，但又不能是 np.inf。

    返回
    ----
    Zg : np.ndarray, same shape and dtype as Z
        每个点存放已排序（按曲面索引）长度 max_m 的 complex np.array。
        如果某个曲面在某个点没有找到匹配，对应位置将为 np.nan。
    """
    dims = Z.shape
    n_dims = len(dims)
    origin = tuple([0] * n_dims)

    # 确定要追踪的曲面数量 m
    if max_m is None:
        m = 0
        has_data = False
        for idx in np.ndindex(*dims):
            if Z[idx] is not None:
                m = max(m, len(Z[idx]))
                if len(Z[idx]) > 0:
                    has_data = True
        if not has_data:  # 如果Z中所有列表都为空或None，则m为0
            raise ValueError("Z appears to contain no data for any point.")
        if m == 0:  # 如果有数据但是所有列表都是空的，也可能到这里
            raise ValueError("Z contains data points, but all lists are empty.")
    else:
        m = max_m

    # 输出与标记数组
    # Zg 存储 np.array，其中包含 m 个 complex 值或 np.nan
    Zg = np.empty(dims, dtype=object)
    assigned = np.zeros(dims, dtype=bool)

    # 处理原点
    origin_vals = Z[origin] if (Z[origin] is not None and len(Z[origin]) > 0) else []

    if not origin_vals:
        # 如果原点没有值，所有曲面初始化为 NaN
        Zg[origin] = np.full(m, np.nan, dtype=complex)
    else:
        # 对原点的值进行排序，然后填充到 Zg[origin]
        sorted_origin_vals = np.array(sorted(origin_vals, key=lambda c: c.real), dtype=complex)
        temp_arr = np.full(m, np.nan, dtype=complex)
        # 确保不会越界
        num_to_fill = min(m, len(sorted_origin_vals))
        temp_arr[:num_to_fill] = sorted_origin_vals[:num_to_fill]
        Zg[origin] = temp_arr
    assigned[origin] = True

    # 按字典序遍历
    for idx in np.ndindex(*dims):
        if assigned[idx]:
            continue

        # 收集所有已赋值的后向邻点
        neighbors = []
        for d in range(n_dims):
            if idx[d] > 0:
                nb = list(idx);
                nb[d] -= 1;
                nb = tuple(nb)
                if assigned[nb]:
                    neighbors.append((nb, d))

        # 如果没有已赋值的后向邻点（除了原点，原点已处理），则无法计算连续性。
        # 这里，我们选择将所有曲面标记为NaN，并继续。
        # 或者可以选择回溯到原点作为唯一的邻点（如果非原点的话）。
        # 当前实现是：如果没有有效邻居，就标记为 NaN
        if not neighbors:
            Zg[idx] = np.full(m, np.nan, dtype=complex)
            assigned[idx] = True
            continue  # 跳过代价计算

        candidates = Z[idx] if (Z[idx] is not None and len(Z[idx]) > 0) else []
        num_candidates = len(candidates)

        if num_candidates == 0:
            # 如果当前点没有候选值，所有曲面都设为 NaN
            Zg[idx] = np.full(m, np.nan, dtype=complex)
            assigned[idx] = True
            continue

        # 构建代价矩阵，维度为 (m, num_candidates)
        # m 是要追踪的曲面数量，num_candidates 是当前点的实际观测值数量
        C_total = np.zeros((m, num_candidates), dtype=float)

        # 针对每个邻点累加代价
        for neighbor, grow_dir in neighbors:
            v_prev_arr = Zg[neighbor]  # 这是 m 长度的 np.array，可能含 NaN

            # 只有当 v_prev_arr 确实是一个数组时才处理
            if v_prev_arr is None:
                # 如果邻点数据本身就是None，那么我们无法从它继承信息
                # 可以选择跳过这个邻点，或者给它一个默认的惩罚
                continue

            d_prev = np.zeros((m, n_dims), dtype=float)
            for s in range(m):
                # 只有当 v_prev[s] 不是 NaN 时才计算导数
                if not np.isnan(v_prev_arr[s]):
                    for d in range(n_dims):
                        d_prev[s, d] = estimate_derivative_one_sided(
                            Zg, assigned, neighbor, s, d, deltas[d]
                        )

            # 构造该邻点的局部 C 矩阵
            # 初始化为一个足够大的值，但不至于 infeasible
            C_nb = np.full((m, num_candidates), fill_value=nan_cost_penalty * 2, dtype=float)
            # 使用 nan_cost_penalty * 2 是为了确保如果一个点是NaN，
            # 那么它匹配到真实值的代价高于从非NaN到真实值的“正常”高代价。
            # 这里的目的是让nan的曲面更倾向于不匹配真实值，
            # 但如果所有曲面都曾是NaN，且只有少量真实值，它仍然可以选择匹配。

            for s in range(m):
                if np.isnan(v_prev_arr[s]):
                    # 如果前一个点此曲面是 NaN，与任何真实候选值匹配的代价
                    # 应该是一个较大的惩罚，但不能是 np.inf
                    # 这样它会优先保持 NaN，除非没有其他选择
                    for c_idx in range(num_candidates):
                        C_nb[s, c_idx] = nan_cost_penalty
                else:
                    v_prev_s = v_prev_arr[s]
                    for c_idx, c in enumerate(candidates):
                        # 值连续性（按 value_weights[grow_dir, :] 加权）
                        cost_val = sum(
                            value_weights[grow_dir, j] * abs(c - v_prev_s)
                            for j in range(n_dims)
                        )
                        # 导数连续性（按 deriv_weights[grow_dir, :] 加权）
                        cost_der = sum(
                            deriv_weights[grow_dir, j]
                            * abs((c - v_prev_s) / deltas[j] - d_prev[s, j])
                            for j in range(n_dims)
                        )
                        C_nb[s, c_idx] = cost_val + cost_der

            C_total += C_nb

        # 全局最优指派
        # linear_sum_assignment 可以在非方阵上工作，它会尝试找到最小的匹配。
        row_ind, col_ind = linear_sum_assignment(C_total)

        # 填入 Zg
        ordered = np.full(m, np.nan, dtype=complex)  # 默认所有曲面为 NaN
        for s_idx, c_idx in zip(row_ind, col_ind):
            # 确保 s_idx 在 m 范围内
            if s_idx < m:  # 这通常应该为真，除非 row_ind 包含了超出 m 的索引（不应该发生）
                ordered[s_idx] = candidates[c_idx]

        Zg[idx] = ordered
        assigned[idx] = True

    return Zg


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # === 1D 示例：零散曲线分类 ===
    nx = 100
    dims1 = (nx,)
    deltas1 = (1.0,)  # 1D 只有一个维度，步长为1.0

    # 权重矩阵降为 1x1
    value_weights = np.array([[1.0]])
    deriv_weights = np.array([[10.0]])

    # 要追踪的曲面数量，这里假设最多有 3 条曲线（即使并非所有点都有 3 条）
    m_1d_curves = 3
    initial_derivs_1d = np.zeros((m_1d_curves, len(deltas1)))

    # 构建格点与真曲线：模拟零散曲线段
    x = np.linspace(0, 4 * np.pi, nx)  # 扩大范围，让曲线形态更明显
    Z1_scattered = np.empty(dims1, dtype=object)
    for i in range(nx):
        vals = []
        # Curve 0: 贯穿始终
        vals.append(np.sin(x[i]))

        # Curve 1: 在中间部分出现
        if nx // 4 <= i < 3 * nx // 4:
            vals.append(np.cos(x[i] * 0.8 + 1))  # 不同频率的cos曲线

        # Curve 2: 在另一部分出现
        if nx // 2 <= i < nx:
            vals.append(np.sin(x[i] * 1.2 - 0.5) + 0.5)  # 另一个不同的sin曲线

        random.shuffle(vals)  # 模拟乱序
        Z1_scattered[i] = vals

    print(
        f"1D Scattered Data: Max m detected in Z1_scattered is {max(len(Z1_scattered[i]) for i in np.ndindex(*dims1))}")

    # 调用分组，传入 max_m 参数
    Zg1_scattered = group_surfaces_one_sided_hungarian(
        Z1_scattered, deltas1,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=m_1d_curves,  # 显式指定要追踪 3 条曲线
        initial_derivatives=initial_derivs_1d,
        nan_cost_penalty=1e5  # 适当的惩罚值
    )

    # 可视化 1D 零散曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(nx):
        if Z1_scattered[i] is not None:
            for v in Z1_scattered[i]:
                plt.scatter(x[i], v.real, s=5, alpha=0.6, color='gray')
    plt.title('分组前 (1D 零散曲线)')
    plt.xlabel('x');
    plt.ylabel('实部')

    plt.subplot(1, 2, 2)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 更多颜色以防万一
    for s in range(m_1d_curves):
        # 提取曲面 s 的实部值，忽略 NaN
        ys = [Zg1_scattered[i][s].real if not np.isnan(Zg1_scattered[i][s]) else np.nan for i in range(nx)]
        # 使用 np.ma.masked_invalid 可以自动处理NaN
        plt.plot(x, np.ma.masked_invalid(ys), label=f'曲面{s}', color=colors[s % len(colors)])
    plt.title('分组后 (1D 零散曲线)')
    plt.xlabel('x');
    plt.ylabel('实部');
    plt.legend()

    plt.tight_layout()
    plt.show()

    # === 3D 示例 (保持不变，以验证修改未破坏原有功能) ===
    nx, ny, nz, m_3d_surfaces = 30, 20, 15, 3
    dims3 = (nx, ny, nz)
    deltas3 = (1.0, 1.0, 1.0)

    value_weights = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ])

    deriv_weights = np.array([
        [10.0, 2.0, 1.0],
        [2.0, 5.0, 1.5],
        [1.0, 1.5, 2.0],
    ])

    initial_derivs = np.zeros((m_3d_surfaces, len(deltas3)))

    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    z = np.linspace(0, 2 * np.pi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    true3 = np.stack([
        np.sin(X + s) * np.cos(Y + s) * np.sin(Z + s)
        for s in range(m_3d_surfaces)
    ], axis=-1)

    Z3 = np.empty(dims3, dtype=object)
    for idx in np.ndindex(*dims3):
        vals = list(true3[idx])
        random.shuffle(vals)
        Z3[idx] = vals

    print(f"\n3D Full Data: Max m detected in Z3 is {max(len(Z3[idx]) for idx in np.ndindex(*dims3))}")

    Zg3 = group_surfaces_one_sided_hungarian(
        Z3, deltas3,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=m_3d_surfaces,  # 显式指定 m
        initial_derivatives=initial_derivs,
        nan_cost_penalty=1e5
    )

    j0, k0 = ny // 2, nz // 2
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for i in range(nx):
        if Z3[i, j0, k0] is not None:
            for v in Z3[i, j0, k0]:
                plt.scatter(x[i], v.real, s=5, alpha=0.6, color='gray')
    plt.title(f'分组前 (y={y[j0]:.2f}, z={z[k0]:.2f})')
    plt.xlabel('x');
    plt.ylabel('实部')

    plt.subplot(1, 2, 2)
    for s in range(m_3d_surfaces):
        ys = [Zg3[i, j0, k0][s].real if not np.isnan(Zg3[i, j0, k0][s]) else np.nan for i in range(nx)]
        plt.plot(x, np.ma.masked_invalid(ys), label=f'曲面{s}', color=colors[s % len(colors)])
    plt.title(f'分组后 (y={y[j0]:.2f}, z={z[k0]:.2f})')
    plt.xlabel('x');
    plt.ylabel('实部');
    plt.legend()

    plt.tight_layout()
    plt.show()

