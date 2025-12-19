# bic_buffer_scan.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import os

# ---- 请确保下面这些模块路径在你的 PYTHONPATH 中 ----
from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.process_multi_dim_params_space import *
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian

# 如果你的包里函数名或位置不同，请相应修改导入
# -------------------------------------------------------
fs = 9
plt.rcParams.update({'font.size': fs})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ---- 用户可调整参数 ----
data_path = 'data/FP_PhC-diff_FP-detailed-14eigenband-strB.csv'
period = 450  # nm
BIC_Q_threshold = 1e5  # Q 阈值
min_peak_prominence = None  # 可根据需要调整 find_peaks 的参数
max_match_k_distance = 0.1  # 匹配阈值（k的单位与网格相同），超出则认为无法匹配
output_dir = './rsl/bic_scan'
os.makedirs(output_dir, exist_ok=True)

# 给定的 buffer 值数组（使用你提供的）
buffer_values = np.array([245., 245.5, 246., 246.5, 247., 247.5, 248., 248.5, 249., 249.5, 250.])
buffer_space = buffer_values[1] - buffer_values[0]
# buffer_values = np.array([242.5, 243. , 243.5, 244. , 244.5, 245. ,
#                           245.5, 246. ,])
# buffer_values = np.array([245., 250.])

# 加载数据并进行与原脚本一致的预处理
df = pd.read_csv(data_path, sep='\t')

# helper: convert and normalize (与你的原脚本保持一致)
c_const = 299792458


def convert_complex(freq_str):
    return complex(freq_str.replace('i', 'j'))


def norm_freq(freq, period_multiplier):
    # freq can be complex or float
    return freq / (c_const / period_multiplier)


# 处理列（保持你原脚本做法）
df["特征频率 (THz)"] = df["特征频率 (THz)"].apply(convert_complex).apply(lambda f: norm_freq(f, period * 1e-9 * 1e12))
df["频率 (Hz)"] = df["频率 (Hz)"].apply(lambda f: norm_freq(f, period * 1e-9) if not pd.isna(f) else f)
df["k"] = df["m1"] + df["m2"] / 2.414
df["phi (rad)"] = df["phi (rad)"].apply(lambda x: x % np.pi)

# 如果你需要 sp/p 识别函数，复制你原本的那段（此处略去，假定无须）
# df["sp_polar_show"] = recognize_sp(...)

# 准备输出结构
results_by_buffer = {}  # buffer -> list of dicts: {'k':..., 'freq':..., 'Q':..., 'peak_idx':...}
all_buffer_kgrid = None  # 保存 grid coords for last run（可选）

# 扫描每个 buffer
for buf in buffer_values:
    print(f"Processing buffer = {buf} nm ...")
    try:
        # create grid (注意：create_data_grid 需要的参数要与项目一致)
        param_keys = ["k", "buffer (nm)"]
        z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "fake_factor (1)", "频率 (Hz)"]
        grid_coords, Z = create_data_grid(df, param_keys, z_keys, deduplication=False)

        # filter by buffer
        new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
            grid_coords, Z,
            z_keys=z_keys,
            fixed_params={'buffer (nm)': float(buf)},
            filter_conditions={
                "fake_factor (1)": {"<": 1},
                "频率 (Hz)": {"<": 0.6, ">": 0.50},
            }
        )

        # 根据你的处理方式提取 Z_new（与你原脚本一致）
        # 你原代码里 Z_filtered 是一个 shape=(N,...) 的 object array，并把第一个元素取出
        Z_new = np.empty_like(Z_filtered, dtype=object)
        for i in range(Z_filtered.shape[0]):
            Z_new[i] = Z_filtered[i][0]  # 取第一个列表元素（如你的脚本）

        # 可视检查：不绘制图以节省时间
        # 现在做 grouping（使用和你原脚本一致的参数）
        deltas3 = (1e-3,)
        value_weights = np.array([[1, ]])
        deriv_weights = np.array([[1, ]])
        Z_grouped = group_vectors_one_sided_hungarian(
            [Z_new], deltas3,
            value_weights=value_weights,
            deriv_weights=deriv_weights,
            max_m=8,
            auto_split_streams=False
        )

        # 得到第 n 个频带
        new_coords_after_group, Z_target = group_solution(new_coords, Z_grouped, freq_index=0)
        # Z_target 是复数组，每个位置有一个复数（或 nan/None）

        # 计算 Qfactors（与原脚本一致）
        # 注意：如果 Im=0，可能导致 inf；先屏蔽掉 nan/inf
        Z_arr = np.array(Z_target, dtype=complex)
        re = np.real(Z_arr)
        im = np.imag(Z_arr)
        with np.errstate(divide='ignore', invalid='ignore'):
            Qfactors = np.where(im != 0, re / (2 * im), np.nan)

        # find peaks on Qfactors
        # convert to 1D numeric array; get indices where Q > 0 (or finite)
        qvals_for_peaks = np.nan_to_num(Qfactors, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        # find_peaks requires 1D; ensure qvals_for_peaks is 1D
        peaks, props = find_peaks(qvals_for_peaks, height=BIC_Q_threshold)
        peaks = np.asarray(peaks)

        # store found peaks: k, freq, Q
        found = []
        for p in peaks:
            k_val = new_coords_after_group['k'][p]
            freq_val = Z_arr[p] if p < len(Z_arr) else np.nan
            Qval = Qfactors[p] if p < len(Qfactors) else np.nan
            found.append({'k': float(k_val), 'freq': complex(freq_val), 'Q': float(Qval), 'index': int(p)})
        results_by_buffer[float(buf)] = found
        all_buffer_kgrid = new_coords_after_group['k']  # 保存最后一次用于参考
    except Exception as e:
        print(f"  ERROR processing buffer {buf}: {e}")
        results_by_buffer[float(buf)] = []

# 将每个 buffer 的结果写成 CSV（方便检查）
rows = []
for buf, found in results_by_buffer.items():
    if len(found) == 0:
        rows.append(
            {'buffer': buf, 'peak_index': None, 'k': np.nan, 'freq_real': np.nan, 'freq_imag': np.nan, 'Q': np.nan})
    else:
        for f in found:
            rows.append({'buffer': buf, 'peak_index': f['index'], 'k': f['k'], 'freq_real': np.real(f['freq']),
                         'freq_imag': np.imag(f['freq']), 'Q': f['Q']})
df_peaks = pd.DataFrame(rows).sort_values(['buffer', 'k'])
df_peaks.to_csv(os.path.join(output_dir, 'bic_peaks_raw.csv'), index=False)
print("Saved raw peaks to:", os.path.join(output_dir, 'bic_peaks_raw.csv'))

# ------------------------------
# 跨-buffer 追踪（tracking）
# ------------------------------
# 我们把每个 buffer 的峰视作一帧，用匈牙利算法在相邻帧间匹配（最小化 k 距离）
buffers_sorted = np.sort(np.array(list(results_by_buffer.keys())))
tracks = []  # 每个 track 是 dict: {'buffers':[], 'ks':[], 'freqs':[], 'Qs':[]}

# 初始化第一帧的 tracks
first_buf = buffers_sorted[0]
first_peaks = results_by_buffer[float(first_buf)]
for p in first_peaks:
    tracks.append({'buffers': [first_buf], 'ks': [p['k']], 'freqs': [p['freq']], 'Qs': [p['Q']]})

# 遍历后续帧
for i in range(1, len(buffers_sorted)):
    cur_buf = buffers_sorted[i]
    prev_buf = buffers_sorted[i - 1]
    cur_peaks = results_by_buffer[float(cur_buf)]
    prev_peaks = results_by_buffer[float(prev_buf)]

    if len(cur_peaks) == 0:
        # 这一帧没有峰：所有现有轨迹加入空值占位（表示该轨迹在此处消失）
        for tr in tracks:
            tr['buffers'].append(cur_buf)
            tr['ks'].append(np.nan)
            tr['freqs'].append(np.nan)
            tr['Qs'].append(np.nan)
        continue

    # 构造距离矩阵：以当前已有轨迹的最新已知 k（从最后非 nan 值取）为行，cur_peaks 的 k 为列
    # 若 tracks 为空（例如第一帧无 peaks），直接把 cur_peaks 新建为 tracks
    if len(tracks) == 0:
        for p in cur_peaks:
            tracks.append({'buffers': [cur_buf], 'ks': [p['k']], 'freqs': [p['freq']], 'Qs': [p['Q']]})
        continue

    # 获取 tracks 当前参考 k（最后一个非 nan 值）
    track_ref_ks = []
    for tr in tracks:
        # 找到最后一个非 nan k
        kvals = np.array(tr['ks'])
        if np.all(np.isnan(kvals)):
            refk = np.nan
        else:
            refk = float(kvals[~np.isnan(kvals)][-1])
        track_ref_ks.append(refk)

    cur_ks = np.array([p['k'] for p in cur_peaks])
    # 对于无法匹配（refk nan），我们设置一个大值距离
    distance = np.full((len(track_ref_ks), len(cur_ks)), fill_value=1e6, dtype=float)
    for r, refk in enumerate(track_ref_ks):
        if np.isnan(refk):
            continue
        for c, ck in enumerate(cur_ks):
            distance[r, c] = abs(refk - ck)

    # 使用 Hungarian 最优分配
    row_idx, col_idx = linear_sum_assignment(distance)
    assigned_cols = set()
    for r, c in zip(row_idx, col_idx):
        if distance[r, c] <= max_match_k_distance:
            # assign this cur peak to track r
            p = cur_peaks[c]
            tracks[r]['buffers'].append(cur_buf)
            tracks[r]['ks'].append(p['k'])
            tracks[r]['freqs'].append(p['freq'])
            tracks[r]['Qs'].append(p['Q'])
            assigned_cols.add(c)
        else:
            # cannot match -> append nan to that track
            tracks[r]['buffers'].append(cur_buf)
            tracks[r]['ks'].append(np.nan)
            tracks[r]['freqs'].append(np.nan)
            tracks[r]['Qs'].append(np.nan)

    # any unassigned current peaks -> create new tracks
    for c_idx, p in enumerate(cur_peaks):
        if c_idx not in assigned_cols:
            # create new track with previous frames filled with nan to align lengths
            new_tr = {'buffers': [], 'ks': [], 'freqs': [], 'Qs': []}
            # add previous buffer placeholders for alignment
            for _ in range(i):  # i frames already passed
                new_tr['buffers'].append(buffers_sorted[_])
                new_tr['ks'].append(np.nan)
                new_tr['freqs'].append(np.nan)
                new_tr['Qs'].append(np.nan)
            # append current
            new_tr['buffers'].append(cur_buf)
            new_tr['ks'].append(p['k'])
            new_tr['freqs'].append(p['freq'])
            new_tr['Qs'].append(p['Q'])
            tracks.append(new_tr)

# 最后 normalize 每个 track 的长度（若有短的加尾部 nan）
n_frames = len(buffers_sorted)
for tr in tracks:
    while len(tr['buffers']) < n_frames:
        # append missing frames at end
        tr['buffers'].append(buffers_sorted[len(tr['buffers'])])
        tr['ks'].append(np.nan)
        tr['freqs'].append(np.nan)
        tr['Qs'].append(np.nan)

# 现在得到若干 tracks，挑出我们关心的前三条主轨迹（按第一个 buffer 中 k 的排序，若不存在则忽略）
# 先构造一个 dataframe 方便输出
track_rows = []
for ti, tr in enumerate(tracks):
    for idx_frame, buf in enumerate(buffers_sorted):
        track_rows.append({
            'track_id': ti,
            'buffer': buf,
            'k': tr['ks'][idx_frame],
            'freq_real': np.nan if tr['freqs'][idx_frame] is np.nan else np.real(tr['freqs'][idx_frame]),
            'freq_imag': np.nan if tr['freqs'][idx_frame] is np.nan else np.imag(tr['freqs'][idx_frame]),
            'Q': tr['Qs'][idx_frame]
        })
df_tracks = pd.DataFrame(track_rows)
df_tracks.to_csv(os.path.join(output_dir, 'bic_tracks_raw.csv'), index=False)
print("Saved raw tracks to:", os.path.join(output_dir, 'bic_tracks_raw.csv'))

# 为了方便展示，挑出在首个 buffer（buffers_sorted[0]）存在峰的那些 track，并按 k 排序
initial_buf = buffers_sorted[0]
initial_present = df_tracks[(df_tracks['buffer'] == initial_buf) & (~df_tracks['k'].isna())]
init_track_order = initial_present.sort_values('k')['track_id'].tolist()
print("Initial tracks present at first buffer (sorted by k):", init_track_order)

# 取前三条（或少于3条时取现有条数）
selected_track_ids = init_track_order[:3] if len(init_track_order) > 0 else list(range(min(3, len(tracks))))

# ------------------------------
# 绘图：k vs buffer, freq vs buffer, k-difference between track pairs
# ------------------------------
plt.figure(figsize=(2, 1))
# colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
colors = ['gray', 'r', 'b']
for idx, tid in enumerate(selected_track_ids):
    df_tid = df_tracks[df_tracks['track_id'] == tid].sort_values('buffer')
    plt.plot(df_tid['buffer']/period, df_tid['k'], marker='o', label=f"track {tid}", color=colors[idx % len(colors)])
    # # 标注拟合斜率
    # mask = ~df_tid['k'].isna()
    # if mask.sum() >= 2:
    #     coef = np.polyfit(df_tid['buffer'].values[mask], df_tid['k'].values[mask], 1)
    #     slope = coef[0]
    #     plt.text(df_tid['buffer'].values[mask].mean(), df_tid['k'].values[mask].mean(),
    #              f"s={slope:.3e}", color=colors[idx % len(colors)])
plt.xlabel('L/P')
plt.ylabel('k (2$\pi$/P)')
# plt.title('BIC tracks: k vs buffer')
# plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'k_vs_buffer.png'), dpi=200, bbox_inches='tight')
print("Saved k vs buffer plot to:", os.path.join(output_dir, 'k_vs_buffer.png'))
plt.savefig('temp.svg', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

plt.figure(figsize=(2, 1))
for idx, tid in enumerate(selected_track_ids):
    df_tid = df_tracks[df_tracks['track_id'] == tid].sort_values('buffer')
    plt.plot(df_tid['buffer']/period, df_tid['freq_real'], marker='o', label=f"track {tid}", color=colors[idx % len(colors)])
plt.xlabel('L/P')
plt.ylabel('Re(f) (c/P)')
# plt.title('BIC tracks: frequency vs buffer')
# plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'freq_vs_buffer.png'), dpi=200, bbox_inches='tight')
print("Saved freq vs buffer plot to:", os.path.join(output_dir, 'freq_vs_buffer.png'))
plt.savefig('temp.svg', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

plt.figure(figsize=(1.25, 3-0.25))
df_tid0 = df_tracks[df_tracks['track_id'] == 0].sort_values('buffer')
df_tid1 = df_tracks[df_tracks['track_id'] == 1].sort_values('buffer')
df_tid2 = df_tracks[df_tracks['track_id'] == 2].sort_values('buffer')
print((df_tid1['freq_real']-df_tid0['freq_real'])/(df_tid2['freq_real']-df_tid0['freq_real']))
plt.plot(df_tid0['buffer']/period, (df_tid1['freq_real'].values-df_tid0['freq_real'].values)/(df_tid2['freq_real'].values-df_tid0['freq_real'].values), marker='o', label=f"track a", color='orange')
plt.plot(df_tid0['buffer']/period, (df_tid1['k'].values-df_tid0['k'].values)/(df_tid2['k'].values-df_tid0['k'].values), marker='o', label=f"track b", color='green')
plt.xlabel('L/P')
plt.ylabel('')
# plt.title('BIC tracks: frequency vs buffer')
# plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'freq_vs_buffer.png'), dpi=200, bbox_inches='tight')
print("Saved freq vs buffer plot to:", os.path.join(output_dir, 'freq_vs_buffer.png'))
plt.savefig('temp.svg', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

colors = ['r', 'b']
# 如果已选择至少两条轨迹，绘制它们之间的 freq 差随 buffer 的变化（展示合并趋势）
if len(selected_track_ids) >= 2:
    plt.figure(figsize=(2, 1))
    tid_ref = selected_track_ids[1]
    for order, idx in enumerate([0, 2]):
        # df_tid = df_tracks[df_tracks['track_id'] == idx].sort_values('buffer')
        # plt.plot(df_tid['buffer'], df_tid['k'], marker='o', label=f"track {idx}", color=colors[idx % len(colors)])
        tid = selected_track_ids[idx]
        d0 = df_tracks[df_tracks['track_id'] == tid_ref].sort_values('buffer')
        d1 = df_tracks[df_tracks['track_id'] == tid].sort_values('buffer')
        # 对应 buffer 下 k 的差（若任一 nan -> nan）
        kd = np.array(d0['k'].values) - np.array(d1['k'].values)
        plt.plot(buffers_sorted/period, np.abs(kd)/(buffer_space/period), marker='o', label=f"|track {tid_ref} - track {tid}|",
                 color=colors[order % len(colors)])
    plt.xlabel('L/P')
    plt.ylabel('|Δk|/ΔL (2$\pi$/P²)')
    # plt.title('k difference between tracks (merging behavior)')
    # plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'k_diff_vs_buffer.png'), dpi=200, bbox_inches='tight')
    print("Saved k-difference plot to:", os.path.join(output_dir, 'k_diff_vs_buffer.png'))
    plt.savefig('temp.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

# 如果已选择至少两条轨迹，绘制它们之间的 freq 差随 buffer 的变化（展示合并趋势）
if len(selected_track_ids) >= 2:
    plt.figure(figsize=(2, 1))
    tid_ref = selected_track_ids[1]
    for order, idx in enumerate([0, 2]):
        tid = selected_track_ids[idx]
        d0 = df_tracks[df_tracks['track_id'] == tid_ref].sort_values('buffer')
        d1 = df_tracks[df_tracks['track_id'] == tid].sort_values('buffer')
        # 对应 buffer 下 freq 的差（若任一 nan -> nan）
        kd = np.array(d0['freq_real'].values) - np.array(d1['freq_real'].values)
        plt.plot(buffers_sorted/period, np.abs(kd)/(buffer_space/period), marker='o', label=f"|track {tid_ref} - track {tid}|",
                 color=colors[order % len(colors)])
    plt.xlabel('L/P')
    plt.ylabel('|Δf|/ΔL (c/P²)')
    # plt.title('freq difference between tracks (merging behavior)')
    # plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'freq_diff_vs_buffer.png'), dpi=200, bbox_inches='tight')
    print("Saved freq-difference plot to:", os.path.join(output_dir, 'freq_diff_vs_buffer.png'))
    plt.savefig('temp.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

print("All done. Results and plots are in:", output_dir)
