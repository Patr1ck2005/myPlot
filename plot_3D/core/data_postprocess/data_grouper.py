import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment


# --- 辅助函数：估计向量分量的偏导数 ---
def estimate_derivative_one_sided_vector_component(Zg_out, assigned, idx, s, axis, delta, vector_component_idx,
                                                   output_dtype):
    """
    在多维数组 Zg_out (存储形状为 (m, num_vector_dims) 的 np.array) 上，
    沿 axis 方向在点 idx 处，估计第 s 条曲面在第 vector_component_idx 个向量分量上的偏导数。
    优先使用已赋值的后向差分，不可用时使用已赋值的前向差分；
    若都不可用，返回 0.0 (或 0.0+0.0j)。
    如果 Zg_out[idx][s, vector_component_idx] 或邻居点的值为 NaN，则无法计算导数，返回 0.0 (或 0.0+0.0j)。
    """
    idx_list = list(idx)

    current_val_s_comp = Zg_out[idx][s, vector_component_idx]
    if np.isnan(current_val_s_comp):
        return output_dtype(0.0)

    # 后向差分
    if idx[axis] - 1 >= 0:
        idx_list_b = idx_list.copy()
        idx_list_b[axis] -= 1
        nb = tuple(idx_list_b)
        if assigned[nb]:
            prev_val_s_comp = Zg_out[nb][s, vector_component_idx]
            if not np.isnan(prev_val_s_comp):
                return (current_val_s_comp - prev_val_s_comp) / delta

    # 前向差分
    if idx[axis] + 1 < Zg_out.shape[axis]:
        idx_list_f = idx_list.copy()
        idx_list_f[axis] += 1
        nf = tuple(idx_list_f)
        if assigned[nf]:
            next_val_s_comp = Zg_out[nf][s, vector_component_idx]
            if not np.isnan(next_val_s_comp):
                return (next_val_s_comp - current_val_s_comp) / delta

    return output_dtype(0.0)


def group_vectors_one_sided_hungarian(
        Z_vector_components,  # list of np.ndarray, each containing lists of numbers (float or complex)
        deltas,
        value_weights,  # (n_dims, n_dims)
        deriv_weights,  # (n_dims, n_dims)
        max_m=None,
        initial_derivatives=None,  # (max_m, n_dims, num_vector_dims)
        nan_cost_penalty=1e9
):
    """
    任意 n 维参数空间中，对多维向量场进行分组。
    每个格点上的数据是 K 维向量的列表，向量分量可以是 float 或 complex。

    参数
    ----
    Z_vector_components : list of np.ndarray
        长度为 num_vector_dims 的列表。每个元素 Z_k 是一个 np.ndarray，
        其 dtype=object, shape=dims。Z_k[idx] 存放长度可变的 numbers 列表，
        代表在点 idx 处所有观测向量的第 k 个分量。
        如果需要处理复数作为一个整体，则 list 长度为 1，
        Z_vector_components[0][idx] = [complex1, complex2, ...]
    deltas : sequence of float, length = n_dims
        各维度的网格步长 Δ_d。
    value_weights : np.ndarray, shape=(n_dims, n_dims)
        生长方向 d（行）对应的“值差”权重向量，列索引 j。
    deriv_weights : np.ndarray, shape=(n_dims, n_dims)
        生长方向 d（行）对应的“导数不连续”权重向量，列索引 j。
    max_m : int, optional
        要追踪的最大曲面/向量流数量。如果未指定，则取所有格点中
        Z_vector_components[0] 列表中长度的最大值。
    initial_derivatives : optional np.ndarray, shape=(max_m, n_dims, num_vector_dims)
        原点处 max_m 条曲面/向量流的初始偏导数猜测值。
        d_prev[s, grow_dir, vector_comp]
    nan_cost_penalty : float, optional
        当一个向量流在前一个点是 np.nan 时，它与当前点真实观测值匹配的惩罚。

    返回
    ----
    Zg_out : np.ndarray, dtype=object, shape=dims
        每个点存放已排序（按向量流索引）的 np.ndarray，形状为 (max_m, num_vector_dims)。
        如果某个向量流在某个点没有找到匹配，对应位置的所有分量将为 np.nan。
    """
    if not Z_vector_components:
        raise ValueError("Z_vector_components cannot be empty.")

    dims = Z_vector_components[0].shape
    n_dims = len(dims)
    num_vector_dims = len(Z_vector_components)
    origin = tuple([0] * n_dims)

    # 确定输出数组的dtype，基于输入数据的类型
    # 检查第一个非空列表的第一个元素的类型
    output_dtype = float
    for idx in np.ndindex(*dims):
        if Z_vector_components[0][idx] is not None and len(Z_vector_components[0][idx]) > 0:
            if isinstance(Z_vector_components[0][idx][0], complex):
                output_dtype = complex
            break

    # 确定要追踪的向量流数量 m
    if max_m is None:
        m = 0
        has_data = False
        for idx in np.ndindex(*dims):
            if Z_vector_components[0][idx] is not None:
                m = max(m, len(Z_vector_components[0][idx]))
                if len(Z_vector_components[0][idx]) > 0:
                    has_data = True
        if not has_data:
            raise ValueError("Z_vector_components appears to contain no data for any point.")
        if m == 0:
            raise ValueError("Z_vector_components contains data points, but all lists are empty.")
    else:
        m = max_m

    # 输出与标记数组
    # Zg_out[idx] 存储一个形状为 (m, num_vector_dims) 的 np.array
    Zg_out = np.empty(dims, dtype=object)
    assigned = np.zeros(dims, dtype=bool)

    # --- 处理原点 ---
    origin_num_candidates = 0
    if Z_vector_components[0][origin] is not None:
        origin_num_candidates = len(Z_vector_components[0][origin])

    if origin_num_candidates == 0:
        Zg_out[origin] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
    else:
        # 收集原点处的观测向量
        origin_vectors = []
        for i in range(origin_num_candidates):
            vec = [Z_k[origin][i] for Z_k in Z_vector_components]
            origin_vectors.append(np.array(vec, dtype=output_dtype))

        # 对向量进行排序（例如，按第一个分量的实部/值）
        # 对于复数，默认会按模和角度排序，或按实部，再按虚部
        # 这里为了稳定，我们强制按第一个分量的实部（如果存在）或直接值排序
        origin_vectors.sort(key=lambda v: (v[0].real if isinstance(v[0], complex) else v[0]))

        temp_arr = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        num_to_fill = min(m, len(origin_vectors))
        temp_arr[:num_to_fill, :] = np.array(origin_vectors[:num_to_fill])
        Zg_out[origin] = temp_arr
    assigned[origin] = True

    # --- 按字典序遍历 ---
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

        if not neighbors:
            Zg_out[idx] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
            assigned[idx] = True
            continue

        # 收集当前格点处的候选观测向量
        num_candidates = 0
        if Z_vector_components[0][idx] is not None:
            num_candidates = len(Z_vector_components[0][idx])

        if num_candidates == 0:
            Zg_out[idx] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
            assigned[idx] = True
            continue

        candidates = []  # list of (num_vector_dims,) np.array
        for i in range(num_candidates):
            vec = [Z_k[idx][i] for Z_k in Z_vector_components]
            candidates.append(np.array(vec, dtype=output_dtype))

        # 构建代价矩阵，维度为 (m, num_candidates)
        C_total = np.zeros((m, num_candidates), dtype=float)

        for neighbor, grow_dir in neighbors:
            v_prev_arr = Zg_out[neighbor]  # (m, num_vector_dims) np.array, possibly with NaNs

            if v_prev_arr is None:
                continue

            # d_prev: (m, n_dims, num_vector_dims)
            d_prev = np.zeros((m, n_dims, num_vector_dims), dtype=output_dtype)
            for s in range(m):
                # 只有当此向量流在邻点至少有一个非NaN分量时才尝试计算导数
                if not np.all(np.isnan(v_prev_arr[s])):
                    for d in range(n_dims):
                        for k in range(num_vector_dims):
                            d_prev[s, d, k] = estimate_derivative_one_sided_vector_component(
                                Zg_out, assigned, neighbor, s, d, deltas[d], k, output_dtype
                            )

            C_nb = np.full((m, num_candidates), fill_value=nan_cost_penalty * 2, dtype=float)

            for s in range(m):
                if np.all(np.isnan(v_prev_arr[s])):
                    # 如果前一个点此向量流所有分量都是 NaN
                    for c_idx in range(num_candidates):
                        C_nb[s, c_idx] = nan_cost_penalty
                else:
                    v_prev_s_vector = v_prev_arr[s]  # (num_vector_dims,) np.array
                    for c_idx, c_vector in enumerate(candidates):  # c_vector is (num_vector_dims,) np.array
                        # 值连续性
                        cost_val = 0.0
                        for j in range(n_dims):
                            cost_val_j = value_weights[grow_dir, j] * np.linalg.norm(c_vector - v_prev_s_vector)
                            cost_val += cost_val_j

                            # 导数连续性
                        cost_der = 0.0
                        for j in range(n_dims):
                            deriv_diff_vector = (c_vector - v_prev_s_vector) / deltas[j] - d_prev[s, j,
                                                                                           :]  # (num_vector_dims,) vector
                            cost_der_j = deriv_weights[grow_dir, j] * np.linalg.norm(deriv_diff_vector)
                            cost_der += cost_der_j

                        C_nb[s, c_idx] = cost_val + cost_der

            C_total += C_nb

        row_ind, col_ind = linear_sum_assignment(C_total)

        ordered = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        for s_idx, c_idx in zip(row_ind, col_ind):
            if s_idx < m:
                ordered[s_idx, :] = candidates[c_idx]

        Zg_out[idx] = ordered
        assigned[idx] = True

    return Zg_out


if __name__ == '__main__':
    # --- 1D 示例：普通复数曲线分类 (复数作为整体处理) ---
    nx_comp = 100
    dims_comp = (nx_comp,)
    deltas_comp = (1.0,)

    value_weights_comp = np.array([[1.0]])
    deriv_weights_comp = np.array([[10.0]])

    m_comp_curves = 3
    num_vector_dims_complex_as_whole = 1  # 复数作为单个“向量分量”

    initial_derivs_comp_as_whole = np.zeros((m_comp_curves, len(deltas_comp), num_vector_dims_complex_as_whole),
                                            dtype=complex)

    x_comp = np.linspace(0, 4 * np.pi, nx_comp)

    Z_complex = np.empty(dims_comp, dtype=object)

    for i in range(nx_comp):
        vals_complex_total = []
        # Curve 0: z = sin(x) + i * cos(x)
        vals_complex_total.append(np.sin(x_comp[i]) + 1j * np.cos(x_comp[i]))

        # Curve 1: z = cos(0.8x+1) + i * sin(0.8x+1)
        if nx_comp // 4 <= i < 3 * nx_comp // 4:
            vals_complex_total.append(0.5*np.cos(x_comp[i] * 0.8 + 1) + 1j * np.sin(x_comp[i] * 0.8 + 1))

        # Curve 2: z = (sin(1.2x-0.5)+0.5) + i * x/10
        if nx_comp // 2 <= i < nx_comp:
            vals_complex_total.append((np.sin(x_comp[i] * 1.2 - 0.5) + 0.5) + 1j * (x_comp[i] / 10))

        random.shuffle(vals_complex_total)

        # 作为一个整体，只有一个分量列表
        Z_complex[i] = vals_complex_total

    # 传递给函数时，Z_vector_components 列表只包含一个元素
    Z_vector_components_comp_as_whole = [Z_complex]

    print(
        f"\n1D Scattered Complex Data (As Whole): Max m detected is {max(len(Z_vector_components_comp_as_whole[0][i]) for i in np.ndindex(*dims_comp))}")

    Zg_comp_as_whole = group_vectors_one_sided_hungarian(
        Z_vector_components_comp_as_whole, deltas_comp,
        value_weights=value_weights_comp,
        deriv_weights=deriv_weights_comp,
        max_m=m_comp_curves,
        initial_derivatives=initial_derivs_comp_as_whole,
        nan_cost_penalty=1e5
    )

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for i in range(nx_comp):
        complex_vals = Z_vector_components_comp_as_whole[0][i]
        if complex_vals is not None:
            for c_val in complex_vals:
                plt.scatter(x_comp[i], c_val.real, s=10, alpha=0.6, color='gray')
    plt.title('分组前 (复平面上的散点 - 复数整体)')
    plt.xlabel('实部');
    plt.ylabel('虚部')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for s in range(m_comp_curves):
        # Zg_comp_as_whole[i] 是一个 (m_comp_curves, 1) 数组，其中唯一分量是复数
        re_coords = [Zg_comp_as_whole[i][s, 0].real if not np.isnan(Zg_comp_as_whole[i][s, 0]) else np.nan for i in
                     range(nx_comp)]
        im_coords = [Zg_comp_as_whole[i][s, 0].imag if not np.isnan(Zg_comp_as_whole[i][s, 0]) else np.nan for i in
                     range(nx_comp)]

        plt.plot(x_comp, np.ma.masked_invalid(re_coords),
                 label=f'复数流{s}', color=colors[s % len(colors)], marker='.', markersize=4)
    plt.title('分组后 (复平面上的复数流 - 复数整体)')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- 1D 示例：零散复数曲线分类 (现在作为2D向量处理) ---
    # 保持不变，以验证双模式兼容性
    nx = 100
    dims1 = (nx,)
    deltas1 = (1.0,)

    value_weights_1d = np.array([[1.0]])
    deriv_weights_1d = np.array([[10.0]])

    m_curves = 3
    num_vector_dims_complex = 2  # Real and Imaginary parts
    initial_derivs_1d_vec = np.zeros((m_curves, len(deltas1), num_vector_dims_complex),
                                     dtype=float)  # Note dtype=float here

    Z_real = np.empty(dims1, dtype=object)
    Z_imag = np.empty(dims1, dtype=object)

    for i in range(nx):
        vals_complex = Z_complex[i]
        Z_real[i] = [c.real for c in vals_complex]
        Z_imag[i] = [c.imag for c in vals_complex]

    Z_vector_components_1d = [Z_real, Z_imag]

    print(
        f"\n1D Scattered Complex Data (Separated): Max m detected is {max(len(Z_vector_components_1d[0][i]) for i in np.ndindex(*dims1))}")

    Zg1_vectorized = group_vectors_one_sided_hungarian(
        Z_vector_components_1d, deltas1,
        value_weights=value_weights_1d,
        deriv_weights=deriv_weights_1d,
        max_m=m_curves,
        initial_derivatives=initial_derivs_1d_vec,
        nan_cost_penalty=1e5
    )

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for i in range(nx):
        real_vals = Z_vector_components_1d[0][i]
        imag_vals = Z_vector_components_1d[1][i]
        if real_vals is not None:
            for r, im in zip(real_vals, imag_vals):
                plt.scatter(r, im, s=10, alpha=0.6, color='gray')
    plt.title('分组前 (复平面上的散点 - 实虚分离)')
    plt.xlabel('实部');
    plt.ylabel('虚部')
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for s in range(m_curves):
        re_coords = [Zg1_vectorized[i][s, 0] if not np.isnan(Zg1_vectorized[i][s, 0]) else np.nan for i in range(nx)]
        im_coords = [Zg1_vectorized[i][s, 1] if not np.isnan(Zg1_vectorized[i][s, 1]) else np.nan for i in range(nx)]

        plt.plot(np.ma.masked_invalid(re_coords), np.ma.masked_invalid(im_coords),
                 label=f'向量流{s}', color=colors[s % len(colors)], marker='.', markersize=4)
    plt.title('分组后 (复平面上的向量流 - 实虚分离)')
    plt.xlabel('实部');
    plt.ylabel('虚部');
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    # --- 3D 示例 (多维向量，例如 3D 向量场) ---
    # 保持不变
    nx, ny, nz = 10, 10, 10
    dims3 = (nx, ny, nz)
    deltas3 = (1.0, 1.0, 1.0)
    m_3d_vectors = 2
    num_vector_dims_3d = 3

    value_weights_3d = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ])

    deriv_weights_3d = np.array([
        [10.0, 2.0, 1.0],
        [2.0, 5.0, 1.5],
        [1.0, 1.5, 2.0],
    ])

    initial_derivs_3d_vec = np.zeros((m_3d_vectors, len(deltas3), num_vector_dims_3d), dtype=float)

    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    z = np.linspace(0, 2 * np.pi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    Z_X_comp = np.empty(dims3, dtype=object)
    Z_Y_comp = np.empty(dims3, dtype=object)
    Z_Z_comp = np.empty(dims3, dtype=object)

    for idx in np.ndindex(*dims3):
        vectors = []
        v0_x = -np.sin(Y[idx]) * np.cos(Z[idx])
        v0_y = np.cos(X[idx]) * np.sin(Z[idx])
        v0_z = np.sin(X[idx]) * np.cos(Y[idx])
        vectors.append(np.array([v0_x, v0_y, v0_z]))

        if (X[idx] > np.pi / 2 and X[idx] < 3 * np.pi / 2) and \
                (Y[idx] > np.pi / 2 and Y[idx] < 3 * np.pi / 2):
            v1_x = np.cos(Y[idx]) + 0.5
            v1_y = -np.sin(X[idx]) + 0.3
            v1_z = np.cos(Z[idx]) * 0.8
            vectors.append(np.array([v1_x, v1_y, v1_z]))

        random.shuffle(vectors)

        Z_X_comp[idx] = [v[0] for v in vectors]
        Z_Y_comp[idx] = [v[1] for v in vectors]
        Z_Z_comp[idx] = [v[2] for v in vectors]

    Z_vector_components_3d = [Z_X_comp, Z_Y_comp, Z_Z_comp]

    print(
        f"\n3D Scattered Vector Data: Max m detected is {max(len(Z_vector_components_3d[0][idx]) for idx in np.ndindex(*dims3))}")

    Zg3_vectorized = group_vectors_one_sided_hungarian(
        Z_vector_components_3d, deltas3,
        value_weights=value_weights_3d,
        deriv_weights=deriv_weights_3d,
        max_m=m_3d_vectors,
        initial_derivatives=initial_derivs_3d_vec,
        nan_cost_penalty=1e5
    )

    y0_slice = ny // 2
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 2, 1)
    for i in range(nx):
        for k in range(nz):
            vecs_at_point = []
            if Z_vector_components_3d[0][i, y0_slice, k] is not None:
                for c_idx in range(len(Z_vector_components_3d[0][i, y0_slice, k])):
                    vecs_at_point.append(np.array([Z_X_comp[i, y0_slice, k][c_idx], Z_Y_comp[i, y0_slice, k][c_idx],
                                                   Z_Z_comp[i, y0_slice, k][c_idx]]))

            for vec in vecs_at_point:
                plt.arrow(x[i], z[k], vec[0] * 0.1, vec[1] * 0.1, color='gray', alpha=0.5, head_width=0.05,
                          head_length=0.1)
    plt.title(f'分组前 3D 向量场 (y={y[y0_slice]:.2f} 切片)')
    plt.xlabel('X');
    plt.ylabel('Z')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    for s in range(m_3d_vectors):
        x_coords = [x[i] for i in range(nx) for k in range(nz) if not np.isnan(Zg3_vectorized[i, y0_slice, k][s, 0])]
        z_coords = [z[k] for i in range(nx) for k in range(nz) if not np.isnan(Zg3_vectorized[i, y0_slice, k][s, 0])]
        u_coords = [Zg3_vectorized[i, y0_slice, k][s, 0] for i in range(nx) for k in range(nz) if
                    not np.isnan(Zg3_vectorized[i, y0_slice, k][s, 0])]
        v_coords = [Zg3_vectorized[i, y0_slice, k][s, 1] for i in range(nx) for k in range(nz) if
                    not np.isnan(Zg3_vectorized[i, y0_slice, k][s, 0])]

        for i_idx, _x in enumerate(x_coords):
            plt.arrow(_x, z_coords[i_idx], u_coords[i_idx] * 0.1, v_coords[i_idx] * 0.1,
                      color=colors[s % len(colors)], alpha=0.7, head_width=0.05, head_length=0.1)
    plt.title(f'分组后 3D 向量场 (y={y[y0_slice]:.2f} 切片)')
    plt.xlabel('X');
    plt.ylabel('Z')
    plt.axis('equal')
    plt.legend()

    plt.tight_layout()
    plt.show()
