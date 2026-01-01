import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from core.advanced_data_analysis.eigtrack_toolbox import (
    setup_matplotlib_default, preprocess_columns_default,
    build_grid, filter_frame, adapt_Zfiltered_take_first,
    group_frame_one_sided_hungarian, pick_band_from_grouped,
    metric_Q_from_complex, find_peaks_on_1d_metric, pack_peaks_records,
    hungarian_track_by_distance,
    peaks_by_scan_to_df, tracks_to_df, save_df_csv,
    select_tracks_present_at_first_scan_sorted_by_k,
    save_png_and_temp_svg_and_show,
)

# ---- parameters (与你当前脚本一致) ----
# data_path = 'data/FP_PhC_Rect-diff_FP-14eigenband-400P-200T-0.6d-0.1k.csv'
# data_path = 'data/FP_PhC_Rect-diff_FP-14eigenband-400P-200T-0.6d-0.1k-(around_merging).csv'
data_path = 'data/FP_PhC_Rect-diff_FP-14eigenband-400P-200T-0.6d-0.1k-(around_merging1).csv'
period = 400
BIC_Q_threshold = 1e5
max_match_k_distance = 0.1
output_dir = './rsl/bic_scan'
os.makedirs(output_dir, exist_ok=True)

# buffer_values = np.linspace(180, 185.5, int((185.5-180)/0.5+1))  # space 0.5
# buffer_values = np.linspace(183, 186, int((186-183)/0.5+1))  # space 0.5
# buffer_values = np.linspace(185.5, 186.5, int((186.5-185.5)/0.1+1))  # space 0.1
buffer_values = np.linspace(185.5, 186.0, int((186.0-185.5)/0.1+1))  # space 0.1
buffer_space = buffer_values[1] - buffer_values[0]

param_keys = ["k", "buffer (nm)"]
z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "fake_factor (1)", "频率 (Hz)"]
filter_conditions = {
    "fake_factor (1)": {"<": 1},
    "频率 (Hz)": {">": 0.4, "<": 0.5},
}

setup_matplotlib_default(font_size=9)

# ---- load + preprocess ----
df = pd.read_csv(data_path, sep='\t')
df = preprocess_columns_default(df, period_nm=period)

# ---- scan (自由组合：你也可以把 build_grid 移出循环做缓存) ----
results_by_buffer = {}

for buf in buffer_values:
    print(f"Processing buffer = {buf} nm ...")
    try:
        grid_coords, Z = build_grid(df, param_keys, z_keys, deduplication=False)

        new_coords, Z_filtered, _min_lens = filter_frame(
            grid_coords, Z,
            z_keys=z_keys,
            fixed_params={"buffer (nm)": float(buf)},
            filter_conditions=filter_conditions,
        )

        Z_new = adapt_Zfiltered_take_first(Z_filtered)

        Z_grouped = group_frame_one_sided_hungarian(
            Z_new,
            deltas=(1e-3,),
            value_weights=np.array([[1.0]]),
            deriv_weights=np.array([[1.0]]),
            max_m=8,
            auto_split_streams=False,
        )

        coords_1d, Z_target_1d = pick_band_from_grouped(new_coords, Z_grouped, freq_index=5)

        Q = metric_Q_from_complex(Z_target_1d)

        peaks_idx, _ = find_peaks_on_1d_metric(Q, height=BIC_Q_threshold)

        # 如果peaks_idx中没有Γ点，则强制添加Γ点（k=0处）
        if 80 not in peaks_idx and 0 in coords_1d["k"]:
            gamma_idx = np.where(coords_1d["k"] == 0)[0][0]
            peaks_idx = np.append(peaks_idx, gamma_idx)
            peaks_idx = np.sort(peaks_idx)

        found = pack_peaks_records(
            peaks_idx,
            coords_1d=coords_1d,
            Z_target_1d=Z_target_1d,
            metric_1d=Q,
            k_key="k",
        )

        # # show Q
        # fig, ax = plt.subplots(figsize=(2, 1))
        # ax.plot(coords_1d["k"], Q, 'k+')
        # # mark peaks
        # if len(peaks_idx) > 0:
        #     ax.plot(coords_1d["k"][peaks_idx], Q[peaks_idx], marker="x", linestyle="", color="red", markersize=8, label="peaks")
        #     # ax.legend()
        # ax.set_xlabel("k (2π/P)")
        # ax.set_ylabel("Q")
        # # ax.set_yscale("log")
        # ax.set_title(f"L = {buf} nm")
        # ax.grid(True)
        # plt.savefig('temp.svg', format='svg', bbox_inches='tight', transparent=True)
        # plt.show()

        # 为了完全对齐你当前 CSV 字段名，把 metric 写回 Q 语义
        results_by_buffer[float(buf)] = [
            {"k": r["k"], "freq": r["freq"], "Q": r["metric"], "metric": r["metric"], "index": r["index"]}
            for r in found
        ]

    except Exception as e:
        print(f"  ERROR processing buffer {buf}: {e}")
        results_by_buffer[float(buf)] = []

# ---- export peaks csv ----
df_peaks = peaks_by_scan_to_df(results_by_buffer, scan_col_name="buffer")
save_df_csv(df_peaks, os.path.join(output_dir, "bic_peaks_raw.csv"))

# ---- tracking ----
buffers_sorted = np.sort(np.array(list(results_by_buffer.keys()), dtype=float))
tracks = hungarian_track_by_distance(
    buffers_sorted,
    results_by_buffer,
    get_point_coord=lambda p: p["k"],
    max_distance=max_match_k_distance,
    fill_value=np.nan,
)

df_tracks = tracks_to_df(tracks, buffers_sorted, scan_col_name="buffer")
save_df_csv(df_tracks, os.path.join(output_dir, "bic_tracks_raw.csv"))

selected_track_ids = select_tracks_present_at_first_scan_sorted_by_k(
    df_tracks, scan_col="buffer", k_col="k", n=3
)

# ---- plots (每次都 temp.svg + show) ----

# k vs buffer
plt.figure(figsize=(2, 1))
colors = ["r", "gray", "b"]
for idx, tid in enumerate(selected_track_ids):
    d = df_tracks[df_tracks["track_id"] == tid].sort_values("buffer")
    plt.plot(d["buffer"]/period, d["k"], marker="o", color=colors[idx % len(colors)])
plt.xlabel("L/P")
plt.ylabel(r"k (2$\pi$/P)")
plt.grid(True)
save_png_and_temp_svg_and_show(output_dir, "k_vs_buffer.png", dpi_png=200)

# freq vs buffer
plt.figure(figsize=(2, 1))
for idx, tid in enumerate(selected_track_ids):
    d = df_tracks[df_tracks["track_id"] == tid].sort_values("buffer")
    plt.plot(d["buffer"]/period, d["freq_real"], marker="o", color=colors[idx % len(colors)])
plt.xlabel("L/P")
plt.ylabel("Re(f) (c/P)")
plt.grid(True)
save_png_and_temp_svg_and_show(output_dir, "freq_vs_buffer.png", dpi_png=200)

# ratio plot (完全照你当前版本：硬取 track_id==0/1/2，并覆盖 freq_vs_buffer.png)
plt.figure(figsize=(1.25, 2.5))
d0 = df_tracks[df_tracks["track_id"] == 0].sort_values("buffer")
d1 = df_tracks[df_tracks["track_id"] == 1].sort_values("buffer")
d2 = df_tracks[df_tracks["track_id"] == 2].sort_values("buffer")
plt.plot(d0["buffer"]/period, (d2["freq_real"].values-d1["freq_real"].values)/(d0["freq_real"].values-d1["freq_real"].values),
         marker="o", color="orange")
plt.plot(d0["buffer"]/period, np.abs(d2["k"].values-d1["k"].values)/np.abs(d0["k"].values-d1["k"].values),
         marker="o", color="green")
plt.xlabel("L/P")
plt.ylabel("")
plt.ylim(None, 1)
plt.grid(True)
save_png_and_temp_svg_and_show(output_dir, "freq_vs_buffer.png", dpi_png=200)

# k diff
if len(selected_track_ids) >= 2:
    plt.figure(figsize=(2, 1))
    colors2 = ["r", "b"]
    tid_ref = selected_track_ids[1]
    for order, idx in enumerate([0, 2]):
        if idx >= len(selected_track_ids):
            continue
        tid = selected_track_ids[idx]
        a = df_tracks[df_tracks["track_id"] == tid_ref].sort_values("buffer")
        b = df_tracks[df_tracks["track_id"] == tid].sort_values("buffer")
        kd = np.array(a["k"].values) - np.array(b["k"].values)
        plt.plot(buffers_sorted/period, np.abs(kd)/(buffer_space/period), marker="o", color=colors2[order % len(colors2)])
    plt.xlabel("L/P")
    plt.ylabel(r"|Δk|/ΔL (2$\pi$/P²)")
    plt.grid(True)
    save_png_and_temp_svg_and_show(output_dir, "k_diff_vs_buffer.png", dpi_png=200)

# freq diff
if len(selected_track_ids) >= 2:
    plt.figure(figsize=(2, 1))
    colors2 = ["r", "b"]
    tid_ref = selected_track_ids[1]
    for order, idx in enumerate([0, 2]):
        if idx >= len(selected_track_ids):
            continue
        tid = selected_track_ids[idx]
        a = df_tracks[df_tracks["track_id"] == tid_ref].sort_values("buffer")
        b = df_tracks[df_tracks["track_id"] == tid].sort_values("buffer")
        fd = np.array(a["freq_real"].values) - np.array(b["freq_real"].values)
        plt.plot(buffers_sorted/period, np.abs(fd)/(buffer_space/period), marker="o", color=colors2[order % len(colors2)])
    plt.xlabel("L/P")
    plt.ylabel(r"|Δf|/ΔL (c/P²)")
    plt.grid(True)
    save_png_and_temp_svg_and_show(output_dir, "freq_diff_vs_buffer.png", dpi_png=200)

print("All done. Results and plots are in:", output_dir)
