# eigtrack_blocks.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

# ---- project adapters ----
from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.process_multi_dim_params_space import create_data_grid, group_solution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian


# =============================================================================
# 1) Preprocess blocks
# =============================================================================
c_const = 299792458


def convert_complex_i_to_j(x: str) -> complex:
    return complex(x.replace("i", "j"))


def norm_freq(freq, period_multiplier: float):
    """freq / (c / period_multiplier)"""
    return freq / (c_const / period_multiplier)


def preprocess_columns_default(
    df: pd.DataFrame,
    period_nm: float,
    *,
    complex_freq_col: str = "特征频率 (THz)",
    hz_freq_col: str = "频率 (Hz)",
    m1_col: str = "m1",
    m2_col: str = "m2",
    k_col: str = "k",
    phi_col: str = "phi (rad)",
    m2_divisor: float = 2.414,
) -> pd.DataFrame:
    """
    与你脚本一致的预处理，但把列名参数化，便于跨项目复用。
    """
    out = df.copy()
    out[complex_freq_col] = (
        out[complex_freq_col]
        .apply(convert_complex_i_to_j)
        .apply(lambda f: norm_freq(f, period_nm * 1e-9 * 1e12))
    )
    out[hz_freq_col] = out[hz_freq_col].apply(
        lambda f: norm_freq(f, period_nm * 1e-9) if not pd.isna(f) else f
    )
    out[k_col] = out[m1_col] + out[m2_col] / m2_divisor
    out[phi_col] = out[phi_col].apply(lambda x: x % np.pi)
    return out


# =============================================================================
# 2) Plot blocks (要求：每次绘图都保存 temp.svg 并 show)
# =============================================================================
def setup_matplotlib_default(font_size: int = 9):
    plt.rcParams.update({"font.size": font_size})
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def save_png_and_temp_svg_and_show(output_dir: str, png_name: str, *, dpi_png: int = 200):
    """
    约定：在用户脚本里画完图后调用一次这个函数
    - 保存 output_dir/png_name
    - 原地保存 temp.svg
    - show
    - close
    """
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, png_name)
    plt.savefig(png_path, dpi=dpi_png, bbox_inches="tight")
    print("Saved plot to:", png_path)
    plt.savefig("temp.svg", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
    plt.close()


# =============================================================================
# 3) Frame (single-scan-value) processing blocks
# =============================================================================
def build_grid(df: pd.DataFrame, param_keys: list, z_keys: list, *, deduplication: bool = False):
    """
    Thin wrapper around create_data_grid.
    """
    return create_data_grid(df, param_keys, z_keys, deduplication=deduplication)


def filter_frame(
    grid_coords,
    Z,
    *,
    z_keys: list,
    fixed_params: dict,
    filter_conditions: dict,
):
    """
    Thin wrapper around advanced_filter_eigensolution.
    """
    return advanced_filter_eigensolution(
        grid_coords,
        Z,
        z_keys=z_keys,
        fixed_params=fixed_params,
        filter_conditions=filter_conditions,
    )


def adapt_Zfiltered_take_first(Z_filtered) -> np.ndarray:
    """
    复刻你脚本的结构适配：Z_filtered[i][0]
    放在一个函数里，方便以后项目里改结构只动这里。
    """
    Z_new = np.empty_like(Z_filtered, dtype=object)
    for i in range(Z_filtered.shape[0]):
        Z_new[i] = Z_filtered[i][0]
    return Z_new


def group_frame_one_sided_hungarian(
    Z_new: np.ndarray,
    *,
    deltas=(1e-3,),
    value_weights=None,
    deriv_weights=None,
    max_m: int = 8,
    auto_split_streams: bool = False,
):
    if value_weights is None:
        value_weights = np.array([[1.0]])
    if deriv_weights is None:
        deriv_weights = np.array([[1.0]])
    return group_vectors_one_sided_hungarian(
        [Z_new],
        deltas,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=max_m,
        auto_split_streams=auto_split_streams,
    )


def pick_band_from_grouped(
    new_coords,
    Z_grouped,
    *,
    freq_index: int = 0,
):
    """
    Thin wrapper around group_solution -> (coords_1d, Z_target_1d)
    """
    return group_solution(new_coords, Z_grouped, freq_index=freq_index)


# =============================================================================
# 4) Metric + peak detection blocks
# =============================================================================
def metric_Q_from_complex(Z_target_1d: np.ndarray) -> np.ndarray:
    """
    与你脚本一致：Q = Re/(2*Im), Im=0 -> nan（不取abs）
    """
    Z_arr = np.array(Z_target_1d, dtype=complex)
    re = np.real(Z_arr)
    im = np.imag(Z_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.where(im != 0, re / (2 * im), 1e15)
    return Q


def find_peaks_on_1d_metric(metric_1d: np.ndarray, *, height: float):
    """
    只做一件事：find_peaks(height=...)
    """
    y = np.nan_to_num(metric_1d, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    peaks, props = find_peaks(y, height=height)
    return np.asarray(peaks), props


def pack_peaks_records(
    peaks_idx: np.ndarray,
    *,
    coords_1d,
    Z_target_1d: np.ndarray,
    metric_1d: np.ndarray,
    k_key: str = "k",
):
    """
    输出与你脚本一致的 found 列表：{'k','freq','Q','index'}
    但字段名也允许你未来改（这里只保留最小必要：k,freq,metric,index）
    """
    Z_arr = np.array(Z_target_1d, dtype=complex)
    found = []
    for p in peaks_idx:
        k_val = coords_1d[k_key][p]
        freq_val = Z_arr[p] if p < len(Z_arr) else np.nan
        m_val = metric_1d[p] if p < len(metric_1d) else np.nan
        found.append(
            {"k": float(k_val), "freq": complex(freq_val), "metric": float(m_val), "index": int(p)}
        )
    return found


# =============================================================================
# 5) Tracking blocks (Hungarian)
# =============================================================================
def hungarian_track_by_distance(
    scan_values_sorted: np.ndarray,
    peaks_by_scan: dict,
    *,
    get_point_coord=lambda peak: peak["k"],
    max_distance: float = 0.1,
    fill_value=np.nan,
):
    """
    通用 Hungarian tracking：
    - peaks_by_scan[scan] -> list of peak dict (至少能 get_point_coord)
    - cost = abs(ref - cur)
    - gate: cost <= max_distance
    - missing filled with fill_value
    结构与输出对齐你的脚本（buffers/ks/freqs/Qs），但内部更通用。
    """
    tracks = []

    if len(scan_values_sorted) == 0:
        return tracks

    first = float(scan_values_sorted[0])
    for p in peaks_by_scan.get(first, []):
        tracks.append(
            {
                "buffers": [first],
                "ks": [p.get("k", fill_value)],
                "freqs": [p.get("freq", fill_value)],
                "metrics": [p.get("metric", fill_value)],
            }
        )

    for i in range(1, len(scan_values_sorted)):
        cur = float(scan_values_sorted[i])
        cur_peaks = peaks_by_scan.get(cur, [])

        if len(cur_peaks) == 0:
            for tr in tracks:
                tr["buffers"].append(cur)
                tr["ks"].append(fill_value)
                tr["freqs"].append(fill_value)
                tr["metrics"].append(fill_value)
            continue

        if len(tracks) == 0:
            for p in cur_peaks:
                tracks.append(
                    {
                        "buffers": [cur],
                        "ks": [p.get("k", fill_value)],
                        "freqs": [p.get("freq", fill_value)],
                        "metrics": [p.get("metric", fill_value)],
                    }
                )
            continue

        # last valid ref coord
        ref_coords = []
        for tr in tracks:
            vals = np.array(tr["ks"], dtype=float)
            if np.all(np.isnan(vals)):
                ref = np.nan
            else:
                ref = float(vals[~np.isnan(vals)][-1])
            ref_coords.append(ref)

        cur_coords = np.array([get_point_coord(p) for p in cur_peaks], dtype=float)

        dist = np.full((len(ref_coords), len(cur_coords)), 1e6, dtype=float)
        for r, ref in enumerate(ref_coords):
            if np.isnan(ref):
                continue
            dist[r, :] = np.abs(cur_coords - ref)

        row_idx, col_idx = linear_sum_assignment(dist)
        assigned_cols = set()

        for r, c in zip(row_idx, col_idx):
            if dist[r, c] <= max_distance:
                p = cur_peaks[c]
                tracks[r]["buffers"].append(cur)
                tracks[r]["ks"].append(p.get("k", fill_value))
                tracks[r]["freqs"].append(p.get("freq", fill_value))
                tracks[r]["metrics"].append(p.get("metric", fill_value))
                assigned_cols.add(c)
            else:
                tracks[r]["buffers"].append(cur)
                tracks[r]["ks"].append(fill_value)
                tracks[r]["freqs"].append(fill_value)
                tracks[r]["metrics"].append(fill_value)

        for c, p in enumerate(cur_peaks):
            if c in assigned_cols:
                continue
            new_tr = {"buffers": [], "ks": [], "freqs": [], "metrics": []}
            for j in range(i):
                new_tr["buffers"].append(float(scan_values_sorted[j]))
                new_tr["ks"].append(fill_value)
                new_tr["freqs"].append(fill_value)
                new_tr["metrics"].append(fill_value)
            new_tr["buffers"].append(cur)
            new_tr["ks"].append(p.get("k", fill_value))
            new_tr["freqs"].append(p.get("freq", fill_value))
            new_tr["metrics"].append(p.get("metric", fill_value))
            tracks.append(new_tr)

    # normalize
    n = len(scan_values_sorted)
    for tr in tracks:
        while len(tr["buffers"]) < n:
            tr["buffers"].append(float(scan_values_sorted[len(tr["buffers"])]))
            tr["ks"].append(fill_value)
            tr["freqs"].append(fill_value)
            tr["metrics"].append(fill_value)

    return tracks


# =============================================================================
# 6) Table/IO blocks
# =============================================================================
def peaks_by_scan_to_df(peaks_by_scan: dict, *, scan_col_name: str = "buffer") -> pd.DataFrame:
    """
    把 {scan: [peak,...]} 转成 DataFrame（对齐你 bic_peaks_raw.csv 的列）
    """
    rows = []
    for scan, found in peaks_by_scan.items():
        if len(found) == 0:
            rows.append(
                {scan_col_name: scan, "peak_index": None, "k": np.nan, "freq_real": np.nan, "freq_imag": np.nan, "Q": np.nan}
            )
        else:
            for f in found:
                rows.append(
                    {
                        scan_col_name: scan,
                        "peak_index": f.get("index", None),
                        "k": f.get("k", np.nan),
                        "freq_real": np.real(f.get("freq", np.nan)),
                        "freq_imag": np.imag(f.get("freq", np.nan)),
                        # 为了兼容你当前 CSV 列名，这里写回 Q
                        "Q": f.get("metric", np.nan),
                    }
                )
    df = pd.DataFrame(rows).sort_values([scan_col_name, "k"])
    return df


def tracks_to_df(tracks: list, scan_values_sorted: np.ndarray, *, scan_col_name: str = "buffer") -> pd.DataFrame:
    """
    把 tracks(list of dict) 转成 DataFrame（对齐你 bic_tracks_raw.csv 的列）
    """
    rows = []
    for tid, tr in enumerate(tracks):
        for i, scan in enumerate(scan_values_sorted):
            fr = tr["freqs"][i]
            rows.append(
                {
                    "track_id": tid,
                    scan_col_name: float(scan),
                    "k": tr["ks"][i],
                    "freq_real": np.nan if fr is np.nan else np.real(fr),
                    "freq_imag": np.nan if fr is np.nan else np.imag(fr),
                    "Q": tr["metrics"][i],
                }
            )
    return pd.DataFrame(rows)


def save_df_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print("Saved:", path)


def select_tracks_present_at_first_scan_sorted_by_k(
    df_tracks: pd.DataFrame,
    *,
    scan_col: str = "buffer",
    k_col: str = "k",
    n: int = 3,
):
    scans = np.sort(df_tracks[scan_col].unique())
    first = scans[0]
    present = df_tracks[(df_tracks[scan_col] == first) & (~df_tracks[k_col].isna())]
    order = present.sort_values(k_col)["track_id"].tolist()
    print("Initial tracks present at first buffer (sorted by k):", order)
    if len(order) == 0:
        return list(range(min(n, df_tracks["track_id"].nunique())))
    return order[:n]
