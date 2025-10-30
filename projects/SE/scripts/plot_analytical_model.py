# -*- coding: utf-8 -*-
"""
Modular plotting for Ptot/Prad with split k-axis panel and k-integrated curves.

Figure 1:
- x: k in [-1, 1], y: omega
- Left half (k in [-1, 0]): Hamiltonian band structure (from H0)
- Right half (k in [0, 1]): Density map of chosen power (Ptot or Prad)

Figure 2:
- k-integrated power vs omega for multiple gamma values (colored by a colormap)

All plot labels/titles are in English; comments are in Chinese for readability.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Callable, Tuple

from plot_3D.advance_plot_styles.line_plot import plot_line_advanced

# ===================== 1) 参数与模式 =====================

# fontsize
fs = 12
plt.rcParams.update({'font.size': fs})


@dataclass
class SystemParams:
    omega0: float = 0.0
    delta: float = 0.0
    gamma0: float = 1e-3
    d1: float = 1.0
    d2: float = 0.0


@dataclass
class PlotParams:
    # 图1：右半密度图的采样分辨率与范围
    k_right_min: float = 0.0
    k_right_max: float = 1.0
    k_right_samples: int = 128

    omega_min: float = -0.5  # 相对 omega0 偏移已在主程设置
    omega_max: float = 0.5
    omega_samples: int = 512

    # 图2：k 积分范围与 ω 取样
    k_int_range: Tuple[float, float] = (-10.0, 10.0)
    # k_int_range: Tuple[float, float] = (-.5, .5)
    omega_int_min: float = -0.1
    omega_int_max: float = 0.1
    omega_int_samples: int = 129

    # 颜色与风格
    cmap_density: str = "magma"
    cmap_lines: str = "viridis"


# 模式：'Ptot' or 'Prad'
MODE_PTOT = "Ptot"
MODE_PRAD = "Prad"


# ===================== 2) 物理核心函数 =====================

def calculate_Heff(omega: float, k: float, current_gamma: float, sp: SystemParams):
    """计算有效哈密顿量 Heff 与辐射耦合矩阵 KKT。"""
    omega0, delta, gamma0, d1, d2 = sp.omega0, sp.delta, sp.gamma0, sp.d1, sp.d2

    H0 = np.array([
        [omega0 + delta, k],
        [k, omega0 - delta]
    ], dtype=complex)

    GammaAbs = np.array([
        [2 * gamma0, 0],
        [0, 2 * gamma0]
    ], dtype=complex)

    KKT = np.array([
        [0, 0],
        [0, 2 * current_gamma]
    ], dtype=complex)

    Heff = H0 - (1j / 2) * GammaAbs - (1j / 2) * KKT
    return Heff, KKT


def calculate_GMatrix(omega: float, k: float, current_gamma: float, sp: SystemParams):
    """计算格林函数 G(omega, k)."""
    Heff, _ = calculate_Heff(omega, k, current_gamma, sp)
    Ginv = omega * np.identity(2, dtype=complex) - Heff
    try:
        return np.linalg.inv(Ginv)
    except np.linalg.LinAlgError:
        return np.full((2, 2), np.nan, dtype=complex)


def calculate_Prad(omega: float, k: float, current_gamma: float, sp: SystemParams):
    """辐射功率 Prad(omega, k) = d^† G^† KKT G d"""
    _, KKT = calculate_Heff(omega, k, current_gamma, sp)
    G = calculate_GMatrix(omega, k, current_gamma, sp)
    d = np.array([sp.d1, sp.d2], dtype=complex)
    val = np.conjugate(d).T @ np.conjugate(G).T @ KKT @ G @ d
    return np.real(val)


def calculate_Ptot(omega: float, k: float, current_gamma: float, sp: SystemParams):
    """总功率 Ptot(omega, k) = d^† G^† [ -Im(Heff)/2 ] G d"""
    Heff, _ = calculate_Heff(omega, k, current_gamma, sp)
    G = calculate_GMatrix(omega, k, current_gamma, sp)
    d = np.array([sp.d1, sp.d2], dtype=complex)
    kernel = -np.imag(Heff) * 2
    val = np.conjugate(d).T @ np.conjugate(G).T @ kernel @ G @ d
    return np.real(val)


def get_power_func(mode: str) -> Callable:
    """根据模式选择功率函数。"""
    if mode == MODE_PTOT:
        return calculate_Ptot
    elif mode == MODE_PRAD:
        return calculate_Prad
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ===================== 3) 能带（左半） =====================

def band_eigenvalues_H0(k_vals: np.ndarray, sp: SystemParams) -> Tuple[np.ndarray, np.ndarray]:
    """返回 H0 的两条能带本征值 ω(k)。"""
    # H0 的解析本征值：omega0 ± sqrt(delta^2 + k^2)
    omega0, delta = sp.omega0, sp.delta
    gap = np.sqrt(delta ** 2 + k_vals ** 2)
    band_plus = omega0 + gap
    band_minus = omega0 - gap
    return band_plus, band_minus


# === 新增：用 Heff 的本征值（取实部）作为能带 ===
def band_eigenvalues_Heff(k_vals: np.ndarray, current_gamma: float, sp: SystemParams):
    """
    返回 Heff 的两条能带：Re(eigvals(Heff(k)))。
    注意：Heff 与 omega 无关（按当前定义），仅依赖 k 与 current_gamma。
    """
    band_plus = []
    band_minus = []
    for kk in k_vals:
        Heff, _ = calculate_Heff(omega=sp.omega0, k=kk, current_gamma=current_gamma, sp=sp)
        vals = np.linalg.eigvals(Heff)
        # # 两个本征值按实部大小排序，方便一致性
        # vals_sorted = np.sort(np.real(vals))
        # band_minus.append(vals_sorted[0])
        # band_plus.append(vals_sorted[1])
        # # 两个本征值按实部大小排序，但是贮存复数
        # vals_sorted = np.sort(vals)
        vals_sorted = vals
        band_minus.append(vals_sorted[0])
        band_plus.append(vals_sorted[1])
    return np.array(band_plus), np.array(band_minus)


# ===================== 4) 网格与积分 =====================

def compute_density_grid(power_fn: Callable, gamma_for_density: float,
                         sp: SystemParams, pp: PlotParams):
    """构建右半区密度图的 (omega, k) 网格与 Z 值。"""
    k_grid = np.linspace(pp.k_right_min, pp.k_right_max, pp.k_right_samples)
    omega_grid = np.linspace(sp.omega0 + pp.omega_min, sp.omega0 + pp.omega_max, pp.omega_samples)

    Z = np.zeros((omega_grid.size, k_grid.size))
    for i, om in enumerate(omega_grid):
        for j, kk in enumerate(k_grid):
            Z[i, j] = power_fn(om, kk, gamma_for_density, sp)
    return k_grid, omega_grid, Z


def integrate_over_k(power_fn: Callable, gamma_val: float,
                     sp: SystemParams, pp: PlotParams,
                     omega_array: np.ndarray):
    """对每个 ω，在 k∈[kmin, kmax] 上数值积分 power_fn(ω,k)。"""
    kmin, kmax = pp.k_int_range
    results = []
    for om in omega_array:
        res, _ = quad(lambda k_: power_fn(om, k_, gamma_val, sp), kmin, kmax)
        results.append(res)
    return np.array(results)


# ===================== 5) 绘图封装 =====================

def plot_figure_1(mode: str,
                  sp: SystemParams,
                  pp: PlotParams,
                  gamma_for_density: float = 1.0,
                  gamma_for_bands: float = None,  # <== 新增：能带使用的 γ，可与右半不同
                  show: bool = True):
    power_fn = get_power_func(mode)

    # 右半密度
    k_right, omega_grid, Z = compute_density_grid(power_fn, gamma_for_density, sp, pp)

    # 左半能带（Heff）
    if gamma_for_bands is None:
        gamma_for_bands = gamma_for_density  # 默认与右半一致
    k_left = np.linspace(-1.0, 0, 300)
    band_plus, band_minus = band_eigenvalues_Heff(k_left, gamma_for_bands, sp)
    original_band_plus, original_band_minus = band_eigenvalues_H0(k_left, sp)

    # 绘图（其余保持你现有版本即可）
    fig, ax = plt.subplots(figsize=(4, 3))
    # mesh = ax.pcolormesh(k_right, omega_grid, Z, cmap=pp.cmap_density, shading='auto')
    im = ax.imshow(Z, extent=[pp.k_right_min, pp.k_right_max, sp.omega0 + pp.omega_min, sp.omega0 + pp.omega_max],
                   origin='lower', cmap=pp.cmap_density, aspect='auto')
    cbar = fig.colorbar(im, ax=ax)

    ax.plot(k_left, original_band_plus, label=r'Band + (Re eig Heff)', color='gray', alpha=0.5)
    ax.plot(k_left, original_band_minus, label=r'Band - (Re eig Heff)', color='gray', alpha=0.5)

    ax.plot(k_left, band_plus, label=r'Band + (Re eig Heff)', color='red')
    ax.plot(k_left, band_minus, label=r'Band - (Re eig Heff)', color='blue')

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    # ax.set_xlabel("k")
    # ax.set_ylabel("Frequency ω")
    # ax.set_title(
    #     f"Split Panel: Bands (left, Re eig Heff) & {mode} density (right)\n"
    #     f"(ω0={sp.omega0}, δ={sp.delta}, γ0={sp.gamma0}, γ_density={gamma_for_density}, "
    #     f"γ_bands={gamma_for_bands}, d=({sp.d1}, {sp.d2}))"
    # )
    # ax.legend(loc="best")

    if show:
        # plt.tight_layout()
        plt.savefig('temp1.svg', transparent=True, bbox_inches='tight')
        plt.show()
    return fig, ax


def plot_band(mode: str,
              sp: SystemParams,
              pp: PlotParams,
              gamma_for_density: float = 1.0,
              gamma_for_bands: float = None,  # <== 新增：能带使用的 γ，可与右半不同
              show: bool = True):
    power_fn = get_power_func(mode)

    # 右半密度
    k_right, omega_grid, Z = compute_density_grid(power_fn, gamma_for_density, sp, pp)

    # 左半能带（Heff）
    if gamma_for_bands is None:
        gamma_for_bands = gamma_for_density  # 默认与右半一致
    k_space = np.linspace(-1.0, 1, 512)
    band_plus, band_minus = band_eigenvalues_Heff(k_space, gamma_for_bands, sp)
    original_band_plus, original_band_minus = band_eigenvalues_H0(k_space, sp)

    # 绘图（其余保持你现有版本即可）
    fig, ax = plt.subplots(figsize=(5, 3))

    plot_params = {
        'enable_fill': True,
        'gradient_fill': True,
        'gradient_direction': 'z3',
        'cmap': 'magma',
        'alpha': 1,
        'alpha_fill': 0.5,
        'legend': False,
        'edge_color': 'none',
        'title': False,
        'scale': 1,
        'add_colorbar': False,
        'global_color_vmin': 0,
        'global_color_vmax': 0.5,
    }
    plot_line_advanced(ax, k_space, z1=band_plus.real, z2=-band_plus.imag, z3=-band_plus.imag, default_color='red',
                       **plot_params)
    plot_line_advanced(ax, k_space, z1=band_minus.real, z2=-band_minus.imag, z3=-band_minus.imag, default_color='blue',
                       **plot_params)

    ax.plot(k_space, original_band_plus, label=r'Band + (Re eig Heff)', color='gray', alpha=1)
    ax.plot(k_space, original_band_minus, label=r'Band - (Re eig Heff)', color='gray', alpha=1)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    # ax.set_xlim([-.5, .5])
    # ax.set_ylim([-.5, .5])

    if show:
        # plt.tight_layout()
        plt.savefig('temp0.svg', transparent=True, bbox_inches='tight')
        plt.show()
    return fig, ax


def plot_spectrum(mode: str,
                  sp: SystemParams,
                  pp: PlotParams,
                  gamma_for_density: float = 1.0,
                  gamma_for_bands: float = None,  # <== 新增：能带使用的 γ，可与右半不同
                  show: bool = True):
    power_fn = get_power_func(mode)

    # 右半密度
    k_right, omega_grid, Z = compute_density_grid(power_fn, gamma_for_density, sp, pp)

    # 绘图（其余保持你现有版本即可）
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(Z, extent=[pp.k_right_min, pp.k_right_max, sp.omega0 + pp.omega_min, sp.omega0 + pp.omega_max],
                   origin='lower', cmap=pp.cmap_density, aspect='auto')
    cbar = fig.colorbar(im, ax=ax)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-0.5, 0.5])

    if show:
        plt.savefig('temp1.svg', transparent=True, bbox_inches='tight')
        plt.show()
    return fig, ax


def plot_figure_2(mode: str,
                  sp: SystemParams,
                  pp: PlotParams,
                  gamma_values=(0.01, 0.1, 0.5, 1.0, 2.0),
                  show: bool = True):
    """
    图2：对 k∈[-1,1] 积分后的功率随 ω 的曲线（多条曲线用同一色图采样的颜色）。
    """
    power_fn = get_power_func(mode)
    omega_array = np.linspace(sp.omega0 + pp.omega_int_min,
                              sp.omega0 + pp.omega_int_max,
                              pp.omega_int_samples)

    # # 用色图为多条曲线赋色
    # cmap = plt.cm.get_cmap(pp.cmap_lines, len(gamma_values))

    # # 根据曲线的gamma_value为其赋色
    cmap = plt.get_cmap(pp.cmap_lines)
    norm = plt.Normalize(-max(gamma_values), max(gamma_values))
    cmap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, ax = plt.subplots(figsize=(2, 3))
    for idx, gval in enumerate(gamma_values):
        y = integrate_over_k(power_fn, gval, sp, pp, omega_array)
        # ax.plot(omega_array, y, label=rf'$\gamma$ = {gval}', color=cmap(idx))
        ax.plot(omega_array, y, label=rf'$\gamma$ = {gval}', color=cmap.to_rgba(gval))
        A = np.sqrt(sp.gamma0*(sp.gamma0+gval))
        analytical_BIC = abs(np.pi/omega_array*(np.sqrt(omega_array**2+1j*gval*omega_array)+np.sqrt(omega_array**2-1j*gval*omega_array)))
        analytical_QGM = abs(0*np.pi*sp.gamma0/omega_array/gval*(np.sqrt(omega_array**2+1j*gval*omega_array)+np.sqrt(omega_array**2-1j*gval*omega_array))+
                             np.pi*omega_array*(1/np.sqrt(omega_array**2+1j*gval*omega_array)+1/np.sqrt(omega_array**2-1j*gval*omega_array)))
        ax.plot(omega_array, analytical_QGM, label=rf'$\gamma$ = {gval}', color=cmap.to_rgba(gval))

    # ax.set_xlabel("Frequency ω")
    # ax.set_ylabel(f"k-integrated {mode}")
    # ax.set_title(f"k-integrated {mode} vs ω (k∈[{pp.k_int_range[0]},{pp.k_int_range[1]}])")
    # ax.grid(True, alpha=0.3)
    # ax.legend(loc="best")

    # 添加colorbar
    # cbar = fig.colorbar(cmap, ax=ax)

    if show:
        # plt.tight_layout()
        plt.savefig('temp2.svg', transparent=True, bbox_inches='tight')
        plt.show()
    return fig, ax


def plot_figure_3(mode: str,
                  sp: SystemParams,
                  pp: PlotParams,
                  omega_values,
                  gamma_value=0.1,  # <== 这里固定一个 gamma
                  show: bool = True):
    power_fn = get_power_func(mode)

    # # 用色图为多条曲线赋色
    # cmap = plt.cm.get_cmap(pp.cmap_lines, len(gamma_values))

    # # 根据曲线的gamma_value为其赋色
    cmap = plt.get_cmap(pp.cmap_lines)
    norm = plt.Normalize(-max(omega_values)*2, max(omega_values))
    cmap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, ax = plt.subplots(figsize=(2, 3))
    k_range_values = np.linspace(0, 1, 101)

    for omega in omega_values:
        y_list = []
        for idx, k_range in enumerate(k_range_values):
            pp.k_int_range = (-k_range, k_range)
            y = integrate_over_k(power_fn, gamma_value, sp, pp, [omega])
            y_list.append(y[0]/k_range/2)  # 平均到k_range
        ax.plot(k_range_values, y_list, '-', label=rf'$\omega$ = {omega}', color=cmap.to_rgba(omega))

    if show:
        # plt.legend()
        plt.savefig('temp3.svg', transparent=True, bbox_inches='tight')
        plt.show()
    return fig, ax


# ===================== 6) 主程序（可直接运行） =====================

if __name__ == "__main__":
    # --- 固定系统参数 ---
    sp = SystemParams(
        omega0=0.0,
        delta=0.0,
        gamma0=1e-3,
        d1=0.0,
        d2=1.0
    )

    # --- 绘图控制参数 ---
    pp = PlotParams(
        # 图1右半密度图范围（相对 omega0 的 ±0.5 ）
        omega_min=-0.5,
        omega_max=0.5,
        omega_samples=512 + 1,
        k_right_samples=128 + 1,

        # 图2的 ω 扫描范围（相对 omega0 的 ±0.1）
        # omega_int_min = 0.17,
        # omega_int_max =  0.23,
        # omega_int_min = -0.5,
        # omega_int_max = 0.5,
        omega_int_min=-0.50,
        omega_int_max=0.50,
        omega_int_samples=128*4+1,

        # 色图
        cmap_density="magma",
        # cmap_lines   = "viridis",
        # cmap_lines="Reds",
        cmap_lines   = "Blues",
    )

    # 选择模式：MODE_PTOT 或 MODE_PRAD
    mode = MODE_PTOT  # 改成 MODE_PRAD 即可切换
    # mode = MODE_PRAD  # 改成 MODE_PRAD 即可切换

    # 图1：右半密度所用 gamma
    # gamma_for_density = 0.5
    gamma_for_density = 0.1

    # 图2：多条曲线的 gamma 列表
    # gamma_values = [0.01, 0.1, 0.5, 1.0, 2.0]
    gamma_values = [0.5]

    # 图3：多条曲线的 k 列表
    k_range_values = [0.01, 0.1, 0.5, 1.0, 2.0]

    # 图3：多条曲线的 omega 列表
    omega_values = [-0.4, -0.2, 0.4]

    # --- 绘制图 ---
    # plot_band(mode, sp, pp, gamma_for_density, show=True)
    # plot_spectrum(mode, sp, pp, gamma_for_density, show=True)
    # plot_figure_1(mode, sp, pp, gamma_for_density, show=True)
    plot_figure_2(mode, sp, pp, gamma_values, show=True)
    # plot_figure_3(mode, sp, pp, omega_values, show=True)
