from typing import List

import numpy as np
import pandas as pd


def create_data_grid(
        df: pd.DataFrame,
        param_keys: list,
        z_key_lst: List[str],
        aggregator=None,
        deduplication=False
):
    """
    根据指定的参数组合 (param_keys) 和多目标数据列 (z_key_lst) 构造多维数据网格，
    每个参数组合对应的目标数据可能有多个，默认收集为多维列表，
    可通过 aggregator 参数对列表数据进行聚合处理。

    参数:
        df: 原始数据集 (DataFrame)，数据集由各参数组合生成，每一行对应一种组合下的观测值
        param_keys: 需要处理的参数名称列表，例如 ["w1 (nm)", "buffer (nm)", "h_grating (nm)"]
        z_key_lst: 包含最终结果数据的列名的列表，例如 ["特征频率 (THz)", 或其他需要后处理的结果]
        aggregator: 可选的聚合函数，接受多维列表作为输入，返回单一值，比如 np.mean、np.median 等
        deduplication: 可选的去重功能, 保证每一个单元格的数据组不重复

    返回:
        grid_coords: dict，存储每个参数对应的唯一取值（已排序），便于后续查找。例如：
                     {
                        "w1 (nm)": np.array([...]),
                        "buffer (nm)": np.array([...]),
                        "h_grating (nm)": np.array([...])
                     }
        Z: 多维 numpy 数组（dtype=object），维度顺序与 param_keys 相同，
           每个单元格存放该组合下包含 z_key_lst 中的所有 z_key 的值的多维列表（或者通过 aggregator 聚合后的值）
    """
    grid_coords = {}
    grid_shape = []

    # 为每个参数获取唯一且排序后的取值
    for key in param_keys:
        unique_vals = np.sort(df[key].unique())
        grid_coords[key] = unique_vals
        grid_shape.append(len(unique_vals))

    # 创建一个多维数组，每个位置存放一个多维列表（dtype=object）
    Z = np.empty(shape=grid_shape, dtype=object)
    for index in np.ndindex(*grid_shape):
        Z[index] = [[] for _ in range(len(z_key_lst))]

    # 遍历 DataFrame 中的每一行，将每行的多个 z_key 对应的值添加到相应位置的单元格的多维列表中
    for _, row in df.iterrows():
        indices = []
        for key in param_keys:
            # 定位当前值在唯一取值数组中的索引
            idx = np.where(grid_coords[key] == row[key])[0]
            if idx.size == 0:
                raise ValueError(f"数值 {row[key]} 在参数 {key} 中未找到匹配项。")
            indices.append(idx[0])
        # # 去重
        # if deduplication:
        #     Z[tuple(indices)] = list(set(Z[tuple(indices)]))
        for z_order, z_key in enumerate(z_key_lst):
            if deduplication and row[z_key] in Z[tuple(indices)][z_order]:
                continue
            # print(row[z_key])
            # 将目标数据添加到对应单元格的列表中
            Z[tuple(indices)][z_order].append(row[z_key])

    # 如果指定了聚合函数，则对每个单元格的数据进行聚合处理
    if aggregator is not None:
        for index in np.ndindex(*grid_shape):
            try:
                Z[index] = aggregator(Z[index])
            except Exception as e:
                raise ValueError(f"在索引 {index} 应用 aggregator 函数时出现错误: {e}")

    return grid_coords, Z


def query_data_grid(
        grid_coords: dict,
        Z: np.ndarray,
        query: dict
):
    """
    根据用户提供的一组参数数值 (query) 查找对应的结果数据，
    若某一单元格内的数据为列表，则直接返回该列表（或聚合后的单一值）。

    参数:
        grid_coords: 从 create_data_grid 返回的参数唯一取值字典
        Z: 多维数据数组 (对应 create_data_grid 返回的 Z)
        query: dict，键为参数名称，值为要查询的数值，例如：
               {"w1 (nm)": 215.00, "buffer (nm)": 1345.0, "h_grating (nm)": 113.50}

    返回:
        对应参数组合下的目标数据（可能为列表或单个聚合值）
    """
    indices = []
    for key, value in query.items():
        if key not in grid_coords:
            raise ValueError(f"参数 {key} 不在网格坐标中")
        idx = np.where(grid_coords[key] == value)[0]
        if idx.size == 0:
            raise ValueError(f"值 {value} 对应的参数 {key} 未找到。")
        indices.append(idx[0])

    return Z[tuple(indices)]

def compress_data_axis(coords, data, axis_key, aggregator=np.max, selection_range=None):
    """
    压缩数据数组的某一维度，对指定参数 axis_key 进行聚合计算。
    可以选择只对该轴中部分索引（例如第2到第5个）应用聚合函数。

    参数:
        coords: dict
            坐标字典，键为参数名，值为该参数的所有取值构成的数组，
            且顺序与 data 数组各维度顺序对应。
        data: np.ndarray
            多维数据数组，其各维度与 coords 中键的顺序相一致。
        axis_key: str
            要压缩的参数名称，比如 "a"。
        reduce_func: function, 可选
            用于在指定轴上进行聚合的函数，默认使用 np.max。
        selection_range: tuple 或 slice, 可选
            指定一个索引范围，只聚合该参数维度中对应的部分数据。
            例如 (1, 5) 表示只聚合第2到第5个（注意：Python 中索引从0开始）。
            如果为 None，则对整个轴进行聚合。

    返回:
        new_coords: dict
            去除了 axis_key 后的坐标字典。
        new_data: np.ndarray
            压缩后的数据数组，其维度比原数组少 1。
    """
    # 1. 获取参数字典的键列表，找出 axis_key 在哪个位置
    keys = list(coords.keys())
    if axis_key not in keys:
        raise KeyError(f"参数字典中不存在 key '{axis_key}'")
    axis_index = keys.index(axis_key)

    # 2. 根据 selection_range 判断在该维度上只选取子区间进行聚合
    if selection_range is not None:
        # 如果传入的是 tuple，则构造 slice 对象
        if isinstance(selection_range, tuple):
            slicer = slice(*selection_range)
        elif isinstance(selection_range, slice):
            slicer = selection_range
        else:
            raise TypeError("selection_range 应为 tuple 或 slice 类型")

        # 构造对 data 进行切片的切片元组：
        # 对于指定的 axis_index，使用 slicer，其它维度使用 slice(None)
        slicers = [slice(None)] * data.ndim
        slicers[axis_index] = slicer
        # 对选定部分数据应用切片
        data_selected = data[tuple(slicers)]
    else:
        data_selected = data

    # 3. 在选定的数据上沿 axis_index 维度进行聚合计算，
    # 注意：由于 data_selected 可能维度与 data 相同（即全部数据）或者该轴已被切片到部分数据，
    # 使用 reduce_func 并移除该维度得到压缩后的数据
    new_data = aggregator(data_selected, axis=axis_index)

    # 4. 在坐标字典中删除被压缩的参数 axis_key，
    # 同时，如果希望记录被选中那部分对应的坐标值，则可用如下方式（这里默认直接删除）
    new_coords = {k: v for k, v in coords.items() if k != axis_key}

    return new_coords, new_data


def group_solution(grid_coords, Z, freq_index=1):
    """
    Extracts the `freq_index`-th element from lists stored in a multi-dimensional array.

    Parameters:
        grid_coords (dict): Dictionary of grid coordinates.
        Z (np.ndarray): Multi-dimensional array with lists as elements.
        freq_index (int): Index of the frequency to extract (0-based).

    Returns:
        tuple: A tuple containing:
            - new_coords (dict): The unchanged grid coordinates.
            - Z_new (np.ndarray): A new array with the extracted frequency values.
    """
    if not isinstance(Z, np.ndarray) or Z.dtype != object:
        raise TypeError("Z must be a numpy array with dtype=object.")
    if not isinstance(freq_index, int) or freq_index < 0:
        raise ValueError("freq_index must be a non-negative integer.")

    # 2. Construct new coordinates
    new_coords = grid_coords

    # 3. Initialize the new array
    new_shape = Z.shape
    Z_new = np.zeros(shape=new_shape, dtype=complex)

    # 4. Iterate over all indices
    for idx in np.ndindex(*new_shape):
        list_A = Z[idx]

        if len(list_A) <= freq_index:
            raise IndexError(
                f"Index {idx} has a list of length {len(list_A)}, "
                f"which is insufficient to access element {freq_index + 1}."
            )

        Z_new[idx] = list_A[freq_index]

    return new_coords, Z_new


from typing import Optional, Tuple, Dict, Any, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable

GridCoords = Dict[str, np.ndarray]
ZArray = np.ndarray  # object 数组（你的 Z / Z_target 结构）

# -----------------------------
# 数据抽取 / 整理
# -----------------------------

def extract_basic_analysis_fields(
    additional_Z_grouped: ZArray,
    band_index: int,
    z_keys: Sequence[str],
    *,
    freq_key: Optional[str] = None,
    q_key: str = '品质因子 (1)',
    tanchi_key: str = 'tanchi (1)',
    phi_key: str = 'phi (rad)'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 additional_Z_grouped 中提取：phi, tanchi, log10(Q), freq(real)
    - 不再假设固定顺序，而是根据 z_keys 动态定位各列索引。
    - freq_key 缺省时会优先使用 '特征频率 (THz)'，若不存在则退回 '频率 (Hz)'.
    结构：additional_Z_grouped[i][j][band][k]，其中 k 对应 z_keys 的索引。
    """
    # 建立 key->idx 映射
    key_to_idx = {k: i for i, k in enumerate(z_keys)}

    # 确定频率字段
    if freq_key is None:
        if '特征频率 (THz)' in key_to_idx:
            freq_key = '特征频率 (THz)'
        elif '频率 (Hz)' in key_to_idx:
            freq_key = '频率 (Hz)'
        else:
            raise KeyError("z_keys 中未找到可用的频率字段：'特征频率 (THz)' 或 '频率 (Hz)'")

    required = [freq_key, q_key, tanchi_key, phi_key]
    for rk in required:
        if rk not in key_to_idx:
            raise KeyError(f"z_keys 缺少必要字段：{rk}")

    freq_idx = key_to_idx[freq_key]
    q_idx = key_to_idx[q_key]
    tanchi_idx = key_to_idx[tanchi_key]
    phi_idx = key_to_idx[phi_key]

    H, W = additional_Z_grouped.shape[:2]
    phi = np.zeros((H, W))
    tanchi = np.zeros((H, W))
    qlog = np.zeros((H, W))
    freq_real = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            band = additional_Z_grouped[i, j][band_index]
            freq = band[freq_idx]
            Q = band[q_idx]
            t = band[tanchi_idx]
            p = band[phi_idx]
            phi[i, j] = float(p)
            tanchi[i, j] = float(t)
            qlog[i, j] = float(np.log10(Q)) if Q > 0 else 0.0
            freq_real[i, j] = float(freq.real if isinstance(freq, complex) else freq)
    return phi, tanchi, qlog, freq_real

def _auto_norm(data: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> Normalize:
    """根据数据与可选的 vmin/vmax 生成 Normalize，自动忽略 NaN/inf。"""
    if vmin is None or vmax is None:
        finite = np.isfinite(data)
        if not finite.any():
            # 兜底，避免全 NaN 导致异常
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax
        else:
            if vmin is None:
                vmin = float(np.nanmin(data[finite]))
            if vmax is None:
                vmax = float(np.nanmax(data[finite]))
            if vmin == vmax:
                # 扩一点范围，避免除零
                eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-9
                vmin -= eps
                vmax += eps
    return Normalize(vmin=vmin, vmax=vmax, clip=True)


def plot_advanced_surface(
    ax: plt.Axes,
    mx: np.ndarray,
    my: np.ndarray,
    z1: np.ndarray,  # 高度
    z2: np.ndarray,  # 颜色值
    z3: Optional[np.ndarray] = None,  # 透明度值(可选；未给则全不透明)
    *,
    mapping: Dict[str, Any],
    elev: float = 30,
    azim: float = 25,
    font_size: int = 9,
    x_key: str = 'm1',
    y_key: str = 'm2',
    z_label: str = 'Frequency (normalized)',
    rstride: int = 1,
    cstride: int = 1,
) -> Tuple[plt.Axes, ScalarMappable]:
    """
    在同一张 3D 图上绘制一个带面：z1 控制高度，z2 控制颜色，z3 控制 alpha。
    参数
    ----
    mapping: 形如 {
        'cmap': 'hot',
        'z2': {'vmin': a, 'vmax': b},   # 可选；未给则自动取数据范围
        'z3': {'vmin': c, 'vmax': d},   # 可选；仅当传入 z3 时有意义；未给则自动 [min,max]
    }
    返回
    ----
    (ax, mappable_for_colorbar)
    """
    plt.rcParams.update({'font.size': font_size})

    # 网格，按 (i,j) 对齐
    Mx, My = np.meshgrid(mx, my, indexing='ij')
    assert z1.shape == z2.shape == Mx.shape, "z1/z2 与网格尺寸不一致"
    if z3 is not None:
        assert z3.shape == z1.shape, "z3 与网格尺寸不一致"

    # colormap & normalize
    cmap_name = mapping.get('cmap', 'hot')
    cmap = get_cmap(cmap_name)

    z2_cfg = mapping.get('z2', {})
    norm_z2 = _auto_norm(z2, z2_cfg.get('vmin'), z2_cfg.get('vmax'))

    if z3 is not None:
        z3_cfg = mapping.get('z3', {})
        norm_z3 = _auto_norm(z3, z3_cfg.get('vmin'), z3_cfg.get('vmax'))
        alphas = norm_z3(z3)  # 0~1
    else:
        alphas = np.ones_like(z2, dtype=float)

    # 生成 RGBA Facecolors
    colors = cmap(norm_z2(z2))
    # 写入 alpha 通道（保留每个面的透明度）
    colors = np.array(colors, copy=True)
    colors[..., 3] = np.clip(alphas, 0.0, 1.0)

    # 对无效数据设为全透明，避免出现杂乱三角面
    invalid = ~np.isfinite(z1) | ~np.isfinite(z2) | (~np.isfinite(z3) if z3 is not None else False)
    if invalid.any():
        colors[invalid] = (0, 0, 0, 0)

    # 绘制
    ax.plot_surface(Mx, My, z1, rstride=rstride, cstride=cstride, facecolors=colors)

    # 轴与视角
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_zlabel(z_label)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])

    # 为 colorbar 构造可复用的 mappable（只用于 z2 的颜色映射）
    mappable = ScalarMappable(norm=norm_z2, cmap=cmap)
    mappable.set_array([])  # 与 colorbar API 兼容

    return ax, mappable
