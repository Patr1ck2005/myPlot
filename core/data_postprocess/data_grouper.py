import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import median_abs_deviation


# ---------- 工具函数：NaN 判定（兼容复数） ----------
def _is_nan_scalar(x):
    """兼容 float/complex 的 NaN 判定；非数值返回 False。"""
    try:
        return np.isnan(x)
    except TypeError:
        if isinstance(x, complex):
            return np.isnan(x.real) or np.isnan(x.imag)
    return False


# ---------- 一侧差分法估计导数 ----------
def estimate_derivative_one_sided_vector_component(
    Zg_out, assigned, idx, s, axis, delta, vector_component_idx, output_dtype
):
    """
    在多维数组 Zg_out (存储形状为 (m, num_vector_dims) 的 np.array) 上，
    沿 axis 方向在点 idx 处，估计第 s 条曲面在第 vector_component_idx 个向量分量上的偏导数。
    优先使用已赋值的后向差分，不可用时使用已赋值的前向差分；
    若都不可用或涉及 NaN，返回 0.0 (或 0.0+0.0j)。
    """
    idx_list = list(idx)

    current_val_s_comp = Zg_out[idx][s, vector_component_idx]
    if _is_nan_scalar(current_val_s_comp):
        return output_dtype(0.0)

    # 后向差分
    if idx[axis] - 1 >= 0:
        idx_list_b = idx_list.copy()
        idx_list_b[axis] -= 1
        nb = tuple(idx_list_b)
        if assigned[nb]:
            prev_val_s_comp = Zg_out[nb][s, vector_component_idx]
            if not _is_nan_scalar(prev_val_s_comp):
                return (current_val_s_comp - prev_val_s_comp) / delta

    # 前向差分
    if idx[axis] + 1 < Zg_out.shape[axis]:
        idx_list_f = idx_list.copy()
        idx_list_f[axis] += 1
        nf = tuple(idx_list_f)
        if assigned[nf]:
            next_val_s_comp = Zg_out[nf][s, vector_component_idx]
            if not _is_nan_scalar(next_val_s_comp):
                return (next_val_s_comp - current_val_s_comp) / delta

    return output_dtype(0.0)


# ---------- 伴随数据：选择器（selector）机制 ----------
def _infer_additional_selector(ad_cell, num_candidates, resolve="deepest"):
    """
    在一个参考单元 ad_cell 上“定位”候选维的层级/轴，并返回 selector（若干个 '*' 组成的 tuple）：
      ()        -> 当前层就是候选层（len == num_candidates）
      ('*',)    -> 下一层是候选层（对当前层每个元素取 [cand_idx]）
      ('*','*') -> 再下一层是候选层，依此类推。

    当存在多个匹配层级时依据 resolve 规则：
      - 'deepest'    : 选更深的路径（推荐，用于 [b][cand][...]）
      - 'shallowest' : 选更浅的路径
      - 'error'      : 抛出错误（严格模式）
    """
    def _len_safe(x):
        try:
            return len(x)
        except Exception:
            return None

    def _is_seq(x):
        return isinstance(x, (list, tuple, np.ndarray))

    paths = []

    def dfs(node, path):
        L = _len_safe(node)
        if L is None:
            return
        if L == num_candidates:
            paths.append(tuple(path))
        if _is_seq(node) and L > 0:
            try:
                first = node[0]
            except Exception:
                first = None
            dfs(first, path + ('*',))

    dfs(ad_cell, ())

    if not paths:
        return None

    uniq_paths = list(sorted(set(paths), key=lambda p: len(p)))
    if len(uniq_paths) == 1:
        return uniq_paths[0]

    # 多条路径 -> 依据 resolve 处理
    if resolve == "error":
        raise ValueError(
            f"additional_data 结构在参照点存在歧义：检测到多个层级长度都等于候选数，路径={uniq_paths}。"
            f"请使候选维位于固定且唯一的层级，或在调用时显式指定 additional_selector。"
        )
    elif resolve == "deepest":
        return uniq_paths[-1]   # 选更深的（例如优先 ('*',) 而不是 ()）
    elif resolve == "shallowest":
        return uniq_paths[0]    # 选更浅的
    else:
        # 未知策略，回退为最深
        return uniq_paths[-1]


def _pick_additional_by_selector(ad_cell, cand_idx, selector):
    """
    使用 selector 稳定地抽取第 cand_idx 个候选的伴随数据。
    selector: 由若干个 '*' 组成的 tuple，表示“向下映射的层数”。
    返回：保持其它轴/维的结构；若遇到 None/越界，返回 None。
    """
    if ad_cell is None or selector is None:
        return None

    def map_down(node, depth=0):
        if depth == len(selector):
            # 命中候选层：在此层按 cand_idx 取元素
            try:
                L = len(node)
                if not (0 <= cand_idx < L):
                    return None
                return node[cand_idx]
            except Exception:
                return None

        token = selector[depth]
        if token != '*':
            return None

        # 这一层需要对每个元素 map
        try:
            L = len(node)
        except Exception:
            return None

        out = []
        for i in range(L):
            try:
                sub = node[i]
            except Exception:
                sub = None
            out.append(map_down(sub, depth + 1))
        return out

    return map_down(ad_cell)


def _pick_additional_candidate_stable(ad_cell, cand_idx, selector):
    """封装一层：selector 缺失则直接返回 None（宁缺毋滥，避免误取）。"""
    return _pick_additional_by_selector(ad_cell, cand_idx, selector) if selector is not None else None


# ---------- 主函数：匈牙利分组 + 单边导数连续性 ----------
def group_vectors_one_sided_hungarian(
    Z_vector_components,   # list[np.ndarray], 每个元素 dtype=object, shape=dims；单元里是“候选列表”
    deltas,                # tuple[float], len=n_dims
    value_weights,         # np.ndarray, shape=(n_dims, n_dims)
    deriv_weights,         # np.ndarray, shape=(n_dims, n_dims)
    max_m=None,
    initial_derivatives=None,  # (max_m, n_dims, num_vector_dims) 未使用可为 None
    nan_cost_penalty=1e9,
    additional_data=None,      # dtype=object，与 Z 前 n 维对齐；后续维度任意（如 [b][cand][...]）
    additional_selector=None,  # tuple，如 (), ('*',), ('*','*')；显式指定候选层（推荐）
    selector_resolve="deepest",  # 'deepest'|'shallowest'|'error'：自动推断多解时的处理策略
    auto_split_streams: bool = True,  # 新增开关：是否启用自动拆分
    min_segment_points: int = 3,  # 可选：子段最小点数（< 此值剔除）
    mad_multiplier: float = 5.0  # 异常检测倍数
):
    """
    在任意 n 维网格上，对“候选列表”进行分组（匈牙利算法 + 单边导数连续性）。
    - 主数据：由 Z_vector_components 决定；每个格点给出若干候选（list），每个候选是 num_vector_dims 维向量。
    - 伴随数据：additional_data 与网格对齐；其单元内包含多轴/多层数据，其中某一层是“候选维”，
                 由 additional_selector 指定；若未指定则自动推断（多解时依据 selector_resolve）。
    返回：
      - 若 additional_data 为 None -> 仅返回 Zg_out
      - 否则 -> 返回 (Zg_out, additional_grouped)
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

    # 确定输出 dtype（float/complex）
    output_dtype = float
    for idx in np.ndindex(*dims):
        if Z_vector_components[0][idx] is not None and len(Z_vector_components[0][idx]) > 0:
            if isinstance(Z_vector_components[0][idx][0], complex):
                output_dtype = complex
            break

    # 确定追踪的 m
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

    # 输出容器
    Zg_out = np.empty(dims, dtype=object)
    assigned = np.zeros(dims, dtype=bool)
    additional_grouped = np.empty(dims, dtype=object) if additional_data is not None else None

    # --- 确定 additional_data 的 selector（一次性，保证全局一致） ---
    ad_selector = None
    if additional_data is not None:
        if additional_selector is not None:
            # 用户显式指定的 selector：最稳定、推荐
            ad_selector = tuple(additional_selector)
        else:
            # 自动推断（遇多解按策略处理；默认 'deepest'，可避免 [(), ('*',)] 的常见歧义）
            # 参照格点：优先 origin，否则找第一个有候选的格点
            ref_idx = origin
            ref_num_candidates = 0
            if Z_vector_components[0][origin] is not None:
                ref_num_candidates = len(Z_vector_components[0][origin])
            if ref_num_candidates == 0:
                for _idx in np.ndindex(*dims):
                    if Z_vector_components[0][_idx] is not None and len(Z_vector_components[0][_idx]) > 0:
                        ref_idx = _idx
                        ref_num_candidates = len(Z_vector_components[0][_idx])
                        break
            if ref_num_candidates > 0:
                ad_selector = _infer_additional_selector(
                    additional_data[ref_idx], ref_num_candidates, resolve=selector_resolve
                )
            # 若 ad_selector 为 None，则后面取伴随数据会返回 None（宁缺毋滥）

    # --- 处理原点（按值排序，作为起点） ---
    origin_num_candidates = 0
    if Z_vector_components[0][origin] is not None:
        origin_num_candidates = len(Z_vector_components[0][origin])

    if origin_num_candidates == 0:
        Zg_out[origin] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        if additional_grouped is not None:
            additional_grouped[origin] = [None] * m
    else:
        # 候选索引按第一个分量的实值排序（复数按 .real）
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
                picked = _pick_additional_candidate_stable(ad, cand_idx, ad_selector)
                ordered_ad[s] = picked
            additional_grouped[origin] = ordered_ad

    assigned[origin] = True

    # --- 按字典序遍历网格其余点 ---
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

        # 组装候选向量
        candidates = []
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
                # 判断该流在前一点是否“全 NaN”
                row = v_prev_arr[s]
                prev_row_all_nan = True
                for val in row:
                    if not _is_nan_scalar(val):
                        prev_row_all_nan = False
                        break

                if not prev_row_all_nan:
                    for d in range(n_dims):
                        for k in range(num_vector_dims):
                            d_prev[s, d, k] = estimate_derivative_one_sided_vector_component(
                                Zg_out, assigned, neighbor, s, d, deltas[d], k, output_dtype
                            )

            C_nb = np.full((m, num_candidates), fill_value=nan_cost_penalty * 2, dtype=float)

            for s in range(m):
                # 再次判断该流在前一点是否“全 NaN”
                row = v_prev_arr[s]
                prev_all_nan = True
                for val in row:
                    if not _is_nan_scalar(val):
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

        # 匈牙利分配
        row_ind, col_ind = linear_sum_assignment(C_total)

        ordered = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        if additional_grouped is not None:
            ordered_ad = [None] * m
        for s_idx, c_idx in zip(row_ind, col_ind):
            if s_idx < m:
                ordered[s_idx, :] = candidates[c_idx]
                if additional_grouped is not None:
                    picked = _pick_additional_candidate_stable(ad_candidates, c_idx, ad_selector)
                    ordered_ad[s_idx] = picked

        Zg_out[idx] = ordered
        if additional_grouped is not None:
            additional_grouped[idx] = ordered_ad
        assigned[idx] = True

    # --------------------------------------------------------------
    # 【彻底修复版】后处理：每条流自动拆分（适用于任意维度网格）
    # --------------------------------------------------------------
    if auto_split_streams:
        # 1. 收集每条流的所有非 NaN 点
        stream_points = [[] for _ in range(m)]
        for idx in np.ndindex(*dims):
            for s in range(m):
                if Zg_out[idx] is None:
                    continue
                vec = Zg_out[idx][s]
                if vec is None or np.all([_is_nan_scalar(v) for v in vec]):
                    continue
                ad = additional_grouped[idx][s] if additional_grouped is not None else None
                stream_points[s].append((idx, vec.copy(), ad))

        # 2. 对每条流单独处理
        for s in range(m):
            points = stream_points[s]
            if len(points) < 3:  # 太少直接跳过
                continue

            # 计算该流内部所有“物理相邻”点的差值（不依赖字典序！）
            diffs = []
            for i, (idx1, vec1, _) in enumerate(points):
                for j in range(i + 1, len(points)):
                    idx2 = points[j][0]
                    dist = sum(abs(a - b) for a, b in zip(idx1, idx2))
                    if dist == 1:  # 真实网格邻居
                        diff = np.linalg.norm(vec1 - points[j][1])
                        diffs.append(diff)

            if len(diffs) == 0:
                continue

            diffs = np.array(diffs)
            median_diff = np.median(diffs)
            mad = median_abs_deviation(diffs, scale='normal')  # scale='normal' ≈ 1.4826 * MAD
            if mad < 1e-12:
                mad = 1e-12

            threshold = median_diff + mad_multiplier * mad  # 自适应阈值

            # 3. 构建该流的连通分量（真正的“连续段”）
            from collections import defaultdict
            graph = defaultdict(list)
            pos_to_i = {points[i][0]: i for i in range(len(points))}

            for i, (idx1, vec1, _) in enumerate(points):
                for d in range(n_dims):
                    for direction in [-1, 1]:
                        nidx_list = list(idx1)
                        if 0 <= idx1[d] + direction < dims[d]:
                            nidx_list[d] += direction
                            nidx = tuple(nidx_list)
                            if nidx in pos_to_i:
                                j = pos_to_i[nidx]
                                diff = np.linalg.norm(vec1 - points[j][1])
                                if diff <= threshold:  # 小于阈值才连边
                                    graph[i].append(j)
                                    graph[j].append(i)

            # 4. DFS 提取所有连通分量（子段）
            visited = [False] * len(points)
            segments = []
            for i in range(len(points)):
                if not visited[i]:
                    component = []
                    stack = [i]
                    visited[i] = True
                    while stack:
                        node = stack.pop()
                        component.append(node)
                        for nb in graph[node]:
                            if not visited[nb]:
                                visited[nb] = True
                                stack.append(nb)
                    segments.append([points[k] for k in component])

            # 5. 清空原流
            for idx, _, _ in points:
                Zg_out[idx][s] = np.full(num_vector_dims, np.nan, dtype=output_dtype)
                if additional_grouped is not None:
                    additional_grouped[idx][s] = None

            # 6. 把足够长的子段放回去（按长度降序，最长的一段优先占原 s）
            segments = [seg for seg in segments if len(seg) >= min_segment_points]
            segments.sort(key=len, reverse=True)

            target_stream = s
            for seg in segments:
                while target_stream < m:
                    # 检查该流是否完全空
                    empty = all(
                        Zg_out[idx][target_stream] is None or
                        np.all([_is_nan_scalar(v) for v in Zg_out[idx][target_stream]])
                        for idx in np.ndindex(*dims)
                    )
                    if empty:
                        for idx, vec, ad in seg:
                            Zg_out[idx][target_stream] = vec
                            if additional_grouped is not None:
                                additional_grouped[idx][target_stream] = ad
                        print(f"[Split] Stream {s} → new stream {target_stream} (length {len(seg)})")
                        target_stream += 1
                        break
                    target_stream += 1
                else:
                    print(f"[Discard] Segment of length {len(seg)} discarded (no empty stream slot)")
                    break
    # --------------------------------------------------------------

    if additional_grouped is None:
        return Zg_out
    return Zg_out, additional_grouped
