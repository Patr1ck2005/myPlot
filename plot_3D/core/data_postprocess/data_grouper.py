import numpy as np
from scipy.optimize import linear_sum_assignment


# --- 辅助函数：估计向量分量的偏导数（与你原实现一致） ---
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
    # 注意：np.isnan 对复数不支持，先判断是否为浮点/复数再分别处理
    try:
        if np.isnan(current_val_s_comp):
            return output_dtype(0.0)
    except TypeError:
        # 复数不支持 isnan，改用实部/虚部判断
        if isinstance(current_val_s_comp, complex) and (np.isnan(current_val_s_comp.real) or np.isnan(current_val_s_comp.imag)):
            return output_dtype(0.0)

    # 后向差分
    if idx[axis] - 1 >= 0:
        idx_list_b = idx_list.copy()
        idx_list_b[axis] -= 1
        nb = tuple(idx_list_b)
        if assigned[nb]:
            prev_val_s_comp = Zg_out[nb][s, vector_component_idx]
            try:
                if not np.isnan(prev_val_s_comp):
                    return (current_val_s_comp - prev_val_s_comp) / delta
            except TypeError:
                if isinstance(prev_val_s_comp, complex) and not (np.isnan(prev_val_s_comp.real) or np.isnan(prev_val_s_comp.imag)):
                    return (current_val_s_comp - prev_val_s_comp) / delta

    # 前向差分
    if idx[axis] + 1 < Zg_out.shape[axis]:
        idx_list_f = idx_list.copy()
        idx_list_f[axis] += 1
        nf = tuple(idx_list_f)
        if assigned[nf]:
            next_val_s_comp = Zg_out[nf][s, vector_component_idx]
            try:
                if not np.isnan(next_val_s_comp):
                    return (next_val_s_comp - current_val_s_comp) / delta
            except TypeError:
                if isinstance(next_val_s_comp, complex) and not (np.isnan(next_val_s_comp.real) or np.isnan(next_val_s_comp.imag)):
                    return (next_val_s_comp - current_val_s_comp) / delta

    return output_dtype(0.0)


# --- 新增：从 additional_data 的一个格点单元中，取“第 cand_idx 个候选” ---
def _pick_additional_candidate(ad_cell, cand_idx, num_candidates):
    """
    从 additional_data 的一个格点单元 ad_cell 中，选取“第 cand_idx 个候选”对应的数据，
    并尽可能保留其它额外维度的结构（例如 ad_cell 形如 [b][候选c][...]，会对每个 b 取第 cand_idx 个）。

    规则（自适应多层嵌套）：
      - 如果当前层长度 == 候选数：直接返回该层的 cand_idx 元素。
      - 否则如果下一层（第 0 个元素）也是序列，且长度 == 候选数：
            对当前层的每个元素都取其 cand_idx，返回同结构的列表。
      - 否则递归向下，直到找到“长度 == 候选数”的那一层；找不到则返回 None。

    ad_cell: 任意结构（list/tuple/np.ndarray/嵌套），或 None
    cand_idx: int，候选索引
    num_candidates: int，候选总数（应与当前格点 Z 的候选数一致）
    """
    if ad_cell is None:
        return None

    # 能否取长度
    try:
        L = len(ad_cell)
    except Exception:
        return None

    # 情形1：当前层就是候选维
    if L == num_candidates:
        return ad_cell[cand_idx] if (0 <= cand_idx < L) else None

    # 尝试看下一层
    try:
        first = ad_cell[0]
        L1 = len(first)
    except Exception:
        # 这一层不是候选维也不是可下探的序列
        return None

    # 情形2：候选维在下一层
    if L1 == num_candidates:
        out = []
        for sub in ad_cell:
            if sub is None:
                out.append(None)
            else:
                out.append(sub[cand_idx] if (0 <= cand_idx < len(sub)) else None)
        return out

    # 情形3：更深层，再递归
    out = []
    any_found = False
    for sub in ad_cell:
        picked = _pick_additional_candidate(sub, cand_idx, num_candidates)
        if picked is not None:
            any_found = True
        out.append(picked)
    return out if any_found else None


def group_vectors_one_sided_hungarian(
        Z_vector_components,  # list of np.ndarray, each containing lists of numbers (float or complex)
        deltas,
        value_weights,  # (n_dims, n_dims)
        deriv_weights,  # (n_dims, n_dims)
        max_m=None,
        initial_derivatives=None,  # (max_m, n_dims, num_vector_dims)
        nan_cost_penalty=1e9,
        additional_data=None   # <--- 新增：与 Z 同格点（前 n 维）的“附加数据”，随排序输出
):
    """
    任意 n 维参数空间中，对多维向量场进行分组（匈牙利算法 + 单边导数连续性）。
    现在支持把一个与 Z 同步的 additional_data 一起按匹配顺序输出。

    输入
    ----
    Z_vector_components : list[np.ndarray]
        长度 = num_vector_dims。每个元素 Z_k 的 dtype=object、shape=dims（dims 为前 n 维网格），
        Z_k[idx] 是“候选列表”（长度可变），其中每个候选是该分量的数值（float/complex）。
        若复数作为“整体”，则只传一个分量（num_vector_dims=1，Z_k[idx][c] 为 complex）。

    deltas : tuple[float], len = n_dims
        各维度的网格步长。

    value_weights, deriv_weights : np.ndarray, shape=(n_dims, n_dims)
        代价函数中的权重矩阵（行是生长方向，列是维度 j）。

    max_m : int or None
        追踪的最大“向量流/曲面条数”。None 时自动取全局最大候选数。

    initial_derivatives : np.ndarray or None, shape = (max_m, n_dims, num_vector_dims)
        原点处的导数初值猜测（未使用时可为 None）。

    nan_cost_penalty : float
        当上一点缺失（NaN）时的惩罚。

    additional_data : np.ndarray or None
        dtype=object。要求 additional_data.shape[:n_dims] == dims（即前 n 维与 Z 对齐），
        后续维度任意。additional_data[idx] 内部可以有任意嵌套结构；
        其中某一层（或多层的下一层）应当是“候选维”（长度 == 当前格点的候选数）。
        函数会自动在该结构中取出被分配的“第 cand_idx 个候选”，并尽量保留其它轴。

    返回
    ----
    Zg_out : np.ndarray, dtype=object, shape=dims
        每个格点存放形状 (m, num_vector_dims) 的 np.ndarray，按向量流 s 排好序；缺失为 NaN。
    additional_grouped : np.ndarray or None
        若提供 additional_data，则返回 shape=dims 的 object 数组。
        每个单元是长度为 m 的 list（或 object 数组），第 s 项就是与 Zg_out[s] 对应的“附加数据条目”（保留额外维度结构）。
        若对应流缺失或找不到候选维，则为 None。
        若 additional_data 为 None，返回 None。
    """
    if not Z_vector_components:
        raise ValueError("Z_vector_components cannot be empty.")

    dims = Z_vector_components[0].shape
    n_dims = len(dims)
    num_vector_dims = len(Z_vector_components)
    origin = tuple([0] * n_dims)

    # 校验 additional_data 的前 n_dims 维
    if additional_data is not None:
        if additional_data.shape[:n_dims] != dims:
            raise ValueError(
                f"additional_data 前 {n_dims} 维应与 Z 的 dims 一致：{dims}，但得到 {additional_data.shape[:n_dims]}"
            )

    # --- 确定输出 dtype（float/complex） ---
    output_dtype = float
    for idx in np.ndindex(*dims):
        if Z_vector_components[0][idx] is not None and len(Z_vector_components[0][idx]) > 0:
            if isinstance(Z_vector_components[0][idx][0], complex):
                output_dtype = complex
            break

    # --- 确定要追踪的 m ---
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

    # --- 输出容器 ---
    Zg_out = np.empty(dims, dtype=object)
    assigned = np.zeros(dims, dtype=bool)
    additional_grouped = np.empty(dims, dtype=object) if additional_data is not None else None

    # --- 处理原点（按值排序，作为起点） ---
    origin_num_candidates = 0
    if Z_vector_components[0][origin] is not None:
        origin_num_candidates = len(Z_vector_components[0][origin])

    if origin_num_candidates == 0:
        Zg_out[origin] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        if additional_grouped is not None:
            additional_grouped[origin] = [None] * m
    else:
        # 候选索引按第一个分量的值（或实部）排序
        idxs = list(range(origin_num_candidates))
        idxs.sort(key=lambda ii: (
            (Z_vector_components[0][origin][ii].real
             if isinstance(Z_vector_components[0][origin][ii], complex)
             else Z_vector_components[0][origin][ii])
        ))

        # 组装排序后的候选向量
        sorted_origin_vectors = []
        for ii in idxs:
            vec = [Z_k[origin][ii] for Z_k in Z_vector_components]
            sorted_origin_vectors.append(np.array(vec, dtype=output_dtype))

        temp_arr = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        num_to_fill = min(m, len(sorted_origin_vectors))
        if num_to_fill > 0:
            temp_arr[:num_to_fill, :] = np.array(sorted_origin_vectors[:num_to_fill])
        Zg_out[origin] = temp_arr

        if additional_grouped is not None:
            ad = additional_data[origin]
            ordered_ad = [None] * m
            for s in range(num_to_fill):
                cand_idx = idxs[s]  # 与 Z 的排序一致
                picked = _pick_additional_candidate(ad, cand_idx, origin_num_candidates)
                ordered_ad[s] = picked
            additional_grouped[origin] = ordered_ad

    assigned[origin] = True

    # --- 按字典序遍历网格 ---
    for idx in np.ndindex(*dims):
        if assigned[idx]:
            continue

        # 已赋值的后向邻点
        neighbors = []
        for d in range(n_dims):
            if idx[d] > 0:
                nb = list(idx); nb[d] -= 1; nb = tuple(nb)
                if assigned[nb]:
                    neighbors.append((nb, d))

        if not neighbors:
            Zg_out[idx] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
            if additional_grouped is not None:
                additional_grouped[idx] = [None] * m
            assigned[idx] = True
            continue

        # 当前格点候选
        num_candidates = 0
        if Z_vector_components[0][idx] is not None:
            num_candidates = len(Z_vector_components[0][idx])

        if num_candidates == 0:
            Zg_out[idx] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
            if additional_grouped is not None:
                additional_grouped[idx] = [None] * m
            assigned[idx] = True
            continue

        candidates = []  # list of (num_vector_dims,) np.array
        for i in range(num_candidates):
            vec = [Z_k[idx][i] for Z_k in Z_vector_components]
            candidates.append(np.array(vec, dtype=output_dtype))

        ad_candidates = additional_data[idx] if additional_data is not None else None

        # 代价矩阵 (m, num_candidates)
        C_total = np.zeros((m, num_candidates), dtype=float)

        for neighbor, grow_dir in neighbors:
            v_prev_arr = Zg_out[neighbor]  # (m, num_vector_dims)

            # d_prev: (m, n_dims, num_vector_dims)
            d_prev = np.zeros((m, n_dims, num_vector_dims), dtype=output_dtype)
            for s in range(m):
                # 只有当此向量流在邻点至少有一个非NaN分量时才尝试计算导数
                try:
                    all_nan = np.all(np.isnan(v_prev_arr[s]))
                except TypeError:
                    # 复数情形下 np.isnan 不适用；退化为检查实部/虚部
                    row = v_prev_arr[s]
                    all_nan = True
                    for val in row:
                        if isinstance(val, complex):
                            if not (np.isnan(val.real) or np.isnan(val.imag)):
                                all_nan = False
                                break
                        else:
                            if not np.isnan(val):
                                all_nan = False
                                break
                if not all_nan:
                    for d in range(n_dims):
                        for k in range(num_vector_dims):
                            d_prev[s, d, k] = estimate_derivative_one_sided_vector_component(
                                Zg_out, assigned, neighbor, s, d, deltas[d], k, output_dtype
                            )

            C_nb = np.full((m, num_candidates), fill_value=nan_cost_penalty * 2, dtype=float)

            for s in range(m):
                # 判断该流在前一格是否“全 NaN”
                try:
                    prev_all_nan = np.all(np.isnan(v_prev_arr[s]))
                except TypeError:
                    row = v_prev_arr[s]
                    prev_all_nan = True
                    for val in row:
                        if isinstance(val, complex):
                            if not (np.isnan(val.real) or np.isnan(val.imag)):
                                prev_all_nan = False
                                break
                        else:
                            if not np.isnan(val):
                                prev_all_nan = False
                                break

                if prev_all_nan:
                    for c_idx in range(num_candidates):
                        C_nb[s, c_idx] = nan_cost_penalty
                else:
                    v_prev_s_vector = v_prev_arr[s]  # (num_vector_dims,)
                    for c_idx, c_vector in enumerate(candidates):
                        # 值连续性
                        cost_val = 0.0
                        for j in range(n_dims):
                            cost_val_j = value_weights[grow_dir, j] * np.linalg.norm(c_vector - v_prev_s_vector)
                            cost_val += cost_val_j

                        # 导数连续性
                        cost_der = 0.0
                        for j in range(n_dims):
                            deriv_diff_vector = (c_vector - v_prev_s_vector) / deltas[j] - d_prev[s, j, :]
                            cost_der_j = deriv_weights[grow_dir, j] * np.linalg.norm(deriv_diff_vector)
                            cost_der += cost_der_j

                        C_nb[s, c_idx] = cost_val + cost_der

            C_total += C_nb

        row_ind, col_ind = linear_sum_assignment(C_total)

        ordered = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        if additional_grouped is not None:
            ordered_ad = [None] * m

        for s_idx, c_idx in zip(row_ind, col_ind):
            if s_idx < m:
                ordered[s_idx, :] = candidates[c_idx]
                if additional_grouped is not None:
                    picked = _pick_additional_candidate(ad_candidates, c_idx, num_candidates)
                    ordered_ad[s_idx] = picked

        Zg_out[idx] = ordered
        if additional_grouped is not None:
            additional_grouped[idx] = ordered_ad
        assigned[idx] = True
    if additional_grouped is None:
        return Zg_out
    return Zg_out, additional_grouped
