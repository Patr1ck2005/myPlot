import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import make_interp_spline

from utils.utils import clear_ax_ticks


def main(freq, x, val=None, condition=None, xlim=None, plot_ylim=None, selection_ylim=None, selection=None, save_name='default'):
    # plt.style.use('classic')
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 18

    fig, ax = plt.subplots(figsize=(4, 4))

    if condition:
        # 按照验证值和条件进行过滤
        _val_mask = condition(val)
        freq[_val_mask] = 0
    if selection_ylim:
        # 对于频率大小进行过滤
        _mask = (freq < selection_ylim[0]) | (freq > selection_ylim[1])
        freq[_mask] = 0

    period = len(x)
    freq_part = freq.reshape(-1, period).T
    sorted_freq_part = np.sort(freq_part, axis=1)

    freq_df = pd.DataFrame(freq_part)
    freq_result = {}
    Q_rsl = {}
    for i, freq_part in enumerate(freq_df):
        # freq_df[freq_part] = freq_df[freq_part].replace(0, np.nan)
        mask = freq_df[freq_part].notna()
        freq_result[i] = freq_df[freq_part]

    if plot_ylim:
        ax.set_ylim(plot_ylim[0], plot_ylim[-1])
    if xlim:
        ax.set_xlim(xlim[0], xlim[-1])
    else:
        ax.set_xlim(x[0], x[-1])

    # ax.xaxis.set_major_locator(MaxNLocator(4))  # X轴最多刻度
    ax.yaxis.set_major_locator(MaxNLocator(4))  # Y轴最多刻度

    # 绘制图形
    for i in selection:
        i -= 1
        ax.errorbar(x,
                    freq_result[i],
                    fmt='-',
                    label=f'Series {i+1}',
                    linewidth=3,
                    markersize=5,
                    fillstyle='none',
                    capsize=2,
                    capthick=1,
                    alpha=0.8)
        # plt.show()

    # clear_ax_ticks(ax)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.savefig(f'./rsl/plot_from_comsol/band_2D_general+{save_name.split("/")[-1].split(".")[0]}.png',
                dpi=300,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
    plt.savefig(f'./rsl/plot_from_comsol/band_2D_general+{save_name.split("/")[-1].split(".")[0]}.svg',
                dpi=300,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
    plt.show()


