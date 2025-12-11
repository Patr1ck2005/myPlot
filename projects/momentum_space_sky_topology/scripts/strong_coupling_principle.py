import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from advance_plot_styles.line_plot import plot_line_advanced

# ===================== 物理模型部分 =====================

# 初始参数（除 k 外的所有参数都做成滑动条）
alpha0 = 1.0      # ω_r = alpha * k^2
c0 = 0.5          # γ_r = c * k^2
omega_e0 = 1.0    # 常数共振频率
gamma_e0 = 0.25   # 常数衰减
kappa0 = 0.3      # 耦合系数 κ

# k 空间
k_min, k_max = -1.5, 1.5
num_k = 400
k_array = np.linspace(k_min, k_max, num_k)


def bands(k_array, alpha, c, omega_e, gamma_e, kappa):
    """对一组 k 计算 2x2 非厄米哈密顿量的本征值，返回形状 (2, N)"""
    vals = np.zeros((2, len(k_array)), dtype=complex)
    hybridization = np.zeros((2, len(k_array)), dtype=float)
    for i, k in enumerate(k_array):
        omega_r = alpha * k**2
        gamma_r = c * k**2
        H = np.array([
            [omega_r - 1j * gamma_r, kappa],
            [kappa, omega_e - 1j * gamma_e]
        ], dtype=complex)
        # w = np.linalg.eigvals(H)
        w, v = np.linalg.eig(H)
        # w = w[np.argsort(np.abs(w)+w.real)]  # 按实部排序
        # v = v[:, np.argsort(np.abs(w)+w.real)]
        vals[:, i] = w
        # hybridization 定义为本征矢的两个分量模的归一化占比
        hybridization[:, i] = np.abs(v[0, :])**2 / (np.abs(v[0, :])**2+np.abs(v[1, :])**2)
    return vals, hybridization


# ===================== 初始计算并绘图 =====================
vals0, hybrid0 = bands(k_array, alpha0, c0, omega_e0, gamma_e0, kappa0)

fig, ax = plt.subplots(figsize=(7, 5))
# 给 5 个滑动条预留空间
plt.subplots_adjust(left=0.12, bottom=0.38)


def draw_dispersion(ax, vals, color_data=None):
    """用 plot_line_advanced 画两条能带，线的“厚度”由虚部决定。"""
    ax.clear()

    # 颜色范围在两条能带之间统一
    # vmin, vmax = color_data.min(), color_data.max()
    vmin, vmax = 0, 1

    y_min = vals.real.min() - 0.2 * (vals.real.max() - vals.real.min()) - 1
    y_max = vals.real.max() + 0.2 * (vals.real.max() - vals.real.min()) + 1
    if color_data is None:
        color_data = vals.imag
        imag_all = vals.imag.flatten()
        vmin, vmax = imag_all.min(), imag_all.max()
    for band_idx in range(2):
        re_part = vals[band_idx].real
        im_part = vals[band_idx].imag   # 用作宽度 & 颜色（虚部）
        plot_line_advanced(
            ax, k_array, re_part,
            z2=im_part,      # 决定填充宽度 → 厚度 ~ Im
            z3=color_data[band_idx],      # 决定颜色
            index=band_idx,
            enable_fill=True,
            enable_dynamic_color=True,
            gradient_fill=False,
            # cmap='magma_r',
            # cmap='RdBu',
            cmap='coolwarm_r',
            default_color='k',
            scale=1,
            alpha_line=1,
            alpha_fill=1,
            linewidth_base=1,
            global_color_vmin=vmin,
            global_color_vmax=vmax,
            add_colorbar=False
        )
        plot_line_advanced(
            ax, k_array, re_part,
            z2=None,      # 决定填充宽度 → 厚度 ~ Im
            z3=None,      # 决定颜色
            index=band_idx,
            enable_fill=False,
            enable_dynamic_color=False,
            gradient_fill=False,
            default_color='k',
            scale=1,
            alpha_line=1,
            alpha_fill=1,
            linewidth_base=1,
            global_color_vmin=vmin,
            global_color_vmax=vmax,
            add_colorbar=False
        )
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_title("Dispersion of H(k)\n(thickness & color ∝ Im eigenvalue)")
    ax.grid(True, alpha=0.3)

# 先画一次
draw_dispersion(ax, vals0, hybrid0)

# ===================== 滑动条 =====================
ax_kappa   = plt.axes([0.15, 0.25, 0.7, 0.03])
ax_alpha   = plt.axes([0.15, 0.20, 0.7, 0.03])
ax_c       = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_omega_e = plt.axes([0.15, 0.10, 0.7, 0.03])
ax_gamma_e = plt.axes([0.15, 0.05, 0.7, 0.03])

slider_kappa   = Slider(ax_kappa,   r"$\kappa$",    0.0, 1.5,  valinit=kappa0)
slider_alpha   = Slider(ax_alpha,   r"$\alpha$",    0.0, 2.0,  valinit=alpha0)
slider_c       = Slider(ax_c,       r"$c$",         0.0, 2.0,  valinit=c0)
slider_omega_e = Slider(ax_omega_e, r"$\omega_e$",  0.0, 2.0,  valinit=omega_e0)
slider_gamma_e = Slider(ax_gamma_e, r"$\gamma_e$",  0.0, 0.5,  valinit=gamma_e0)

def update(val):
    alpha   = slider_alpha.val
    c       = slider_c.val
    omega_e = slider_omega_e.val
    gamma_e = slider_gamma_e.val
    kappa   = slider_kappa.val

    vals, hybrid = bands(k_array, alpha, c, omega_e, gamma_e, kappa)
    draw_dispersion(ax, vals, hybrid)
    fig.canvas.draw_idle()
    plt.savefig("strong_coupling_dispersion.png", dpi=300)

for s in [slider_kappa, slider_alpha, slider_c, slider_omega_e, slider_gamma_e]:
    s.on_changed(update)

plt.show()
