import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import make_interp_spline

from utils.utils import clear_ax_ticks


def main(freq, Q, x, xlim, plot_ylim, selection_ylim=None, selection=None, save_name='default'):
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 18

    fig, ax = plt.subplots(figsize=(4, 4))

    if selection_ylim:
        # 对于频率大小进行过滤
        _mask = (freq < selection_ylim[0]) | (freq > selection_ylim[1])
        freq[_mask] = 0
        Q[_mask] = 0

    period = len(x)
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

    ax.set_ylim(plot_ylim[0], plot_ylim[-1])
    ax.set_xlim(xlim[0], xlim[-1])
    # ax.set_ylim(1e10, 1e12)
    # ax.set_xlim(1e-5, 1e-3)

    # ax.set_xticks([0, 0.33, 0.66, 1])
    # ax.set_xticklabels([0, 0.005, 0.005, 0.015])

    # ax.xaxis.set_major_locator(MaxNLocator(4))  # X轴最多刻度
    ax.yaxis.set_major_locator(MaxNLocator(4))  # Y轴最多刻度

    # 绘制带误差棒的图形
    for i in selection:
        i -= 1
        ax.errorbar(abs(x),
                    abs(freq_result[i]),
                    # yerr=1/Q_rsl[i]*1e14*5e-1,
                    fmt='-o',
                    label=f'Series {i+1}',
                    linewidth=3,
                    markersize=5,
                    fillstyle='none',
                    capsize=2,
                    capthick=1,
                    alpha=0.8)
        # plt.show()

    # clear_ax_ticks(ax)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'./rsl/band_2D_with_error_bars+{save_name.split("/")[-1].split(".")[0]}.png',
                dpi=300,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
    plt.savefig(f'./rsl/band_2D_with_error_bars+{save_name.split("/")[-1].split(".")[0]}.svg',
                dpi=300,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
    plt.show()


if __name__ == '__main__':
    # 修改此处为实际的文件路径
    # main('./data/VBG-band2D-freq-Gamma_M-0.12.txt', './data/VBG-band2D-Q-Gamma_M-0.12.txt')  # period = 61
    # main('./data/VBG-band2D-freq-Gamma_X-0.12.txt', './data/VBG-band2D-Q-Gamma_X-0.12.txt')
    # 加载频率和 Q 数据
    freq = np.loadtxt('./data/3EP_noslab/perturbations/3EP-band2D-freq.txt')[:, 1]-7.120e13
    Q = np.loadtxt('./data/3EP_noslab/perturbations/3EP-band2D-Q.txt')[:, 1]
    main(
        freq,
        Q,
        x=np.linspace(0.0055, 0.0070, 151) - 0.006285,
        # ylim=[7.0e13-7.12e13, 7.22e13-7.12e13],
        xlim=[1e-5, 1e-3],
        plot_ylim=[1e10, 1e12],
        selection=[1, 2, 3],
    )
    ifreq = np.loadtxt('./data/3EP_noslab/perturbations/3EP-band2D-ifreq.txt')[:, 1]/2/np.pi+1.22e13/2/np.pi
    main(
        ifreq,
        Q,
        x=np.linspace(0.0055, 0.0070, 151)-0.006285,
        # xlim=[1e-5, 1e-3],
        # ylim=[-2*1e13+1.2e13, 0+1.2e13],
        # plot_ylim=[1e10, 1e12],
        selection=[1, 2, 3]
    )
    # main(
    #     'data/3EP_noslab/0~3e-2k/3EP-band2D-freq.txt',
    #     'data/3EP_noslab/0~3e-2k/3EP-band2D-Q.txt',
    #     ylim=[6.5e13, 7.7e13],
    #     period=601,
    #     selection=[1, 2, 3]
    # )
    # main(
    #     'data/3EP_noslab/0~3e-2k/3EP-band2D-ifreq.txt',
    #     'data/3EP_noslab/0~3e-2k/3EP-band2D-Q.txt',
    #     [-3 * 1e13, 0],
    #     period=601,
    #     selection=[1, 2, 3]
    # )
