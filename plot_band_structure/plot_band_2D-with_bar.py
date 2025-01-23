import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline


def main(freq_data_path, Q_data_path):
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 28

    fig, ax = plt.subplots(figsize=(4, 8))

    # 加载频率和 Q 数据
    freq = np.loadtxt(freq_data_path)[:, 1]
    Q = np.loadtxt(Q_data_path)[:, 1]

    period = 61
    freq_part = freq.reshape(-1, period).T  # 假设 `freq` 代表 zone folding 数据
    Q_part = Q.reshape(-1, period).T
    sorted_freq_part = np.sort(freq_part, axis=1)
    sorted_Q_part = np.sort(Q_part, axis=1)

    freq_df = pd.DataFrame(freq_part)
    Q_df = pd.DataFrame(Q_part)
    freq_result = {}
    Q_rsl = {}
    for i, freq_part in enumerate(freq_df):
        # freq_df[freq_part] = freq_df[freq_part].replace(0, np.nan)
        mask = freq_df[freq_part].notna()
        freq_result[i] = freq_df[freq_part]
        Q_rsl[i] = Q_df[freq_part]*mask

    ax.set_ylim(0.95e14, 1.40e14)
    ax.set_xlim(0, 1)

    # 绘制带误差棒的图形
    for i in range(8):
        ax.errorbar(np.linspace(0, 1, len(freq_result[i])),
                    freq_result[i],
                    yerr=1/Q_rsl[i]*1e14*5e-1,
                    fmt='o',
                    label=f'Series {i+1}',
                    linewidth=1,
                    markersize=3,
                    capsize=2,
                    capthick=1,
                    alpha=0.8)
        # plt.show()

    # 去除坐标轴标签和刻度
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # 去除边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(f'../rsl/band_2D_with_error_bars+{freq_data_path.split("/")[-1].split(".")[0]}.png',
                dpi=300,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
    plt.show()


if __name__ == '__main__':
    # 修改此处为实际的文件路径
    main('../data/VBG-band2D-freq-Gamma_M-0.12.txt', '../data/VBG-band2D-Q-Gamma_M-0.12.txt')
    main('../data/VBG-band2D-freq-Gamma_X-0.12.txt', '../data/VBG-band2D-Q-Gamma_X-0.12.txt')
