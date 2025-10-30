# plot_critical_couplings.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from physics_core import SystemParams, NonHermitianTwoLevel

# ===== 基础参数（除非某组覆盖）=====
PARAMS = {
    "omega0": 0.4413,
    "delta":  0.00585,
    "gamma0": 1.2e-4,
    "d1":     0.0,
    "d2":     1.0,
    "gamma_rad": 3.5e-4,
    "dispersion_v": 0.29,
}

# ===== 频率扫描设置（单位与 omega 一致）=====
OMEGA_RANGE = {
    "min": 0.430,   # 根据需要调整
    "max": 0.442,   # 根据需要调整
    "num": 1201,    # 采样点数
}

# ===== 扫描组：每组“一起”指定 (gamma_rad, omega0, delta) =====
# key 是图例上的组名；只写需要覆盖的那三项，其它用 PARAMS 默认值。
GROUPS = {
}

# 读取数据来设置GROUPS
lossless_data_path = r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\eigens\3fold-QGM-kloss0-delta_space.pkl"
lossly_data_path = r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\eigens\3fold-QGM-kloss1e-3-delta_space.pkl"
import pickle
with open(lossless_data_path, 'rb') as f:
    data_lossless = pickle.load(f)
    x_vals = data_lossless['x_vals']
    complx_eigenfreq_lossless = data_lossless['subs'][0]  # 取第一个子图数据
with open(lossly_data_path, 'rb') as f:
    data_lossly = pickle.load(f)
    x_vals = data_lossly['x_vals']
    complx_eigenfreq_lossy = data_lossly['subs'][0]  # 取第一个子图数据
# 计算 gamma_rad
for i, x in enumerate(x_vals):
    omega0 = 0.4413
    delta = omega0 - float(complx_eigenfreq_lossy[i].real)  # delta = Re(omega_lossy) - omega0
    omega1 = float(complx_eigenfreq_lossy[i].real)
    gamma0 = float(complx_eigenfreq_lossy[i].imag-complx_eigenfreq_lossless[i].imag)
    gamma_rad = float(complx_eigenfreq_lossless[i].imag)  # gamma_rad = gamma_total - gamma0
    label = f"x={x:.3f}"
    GROUPS[label] = {
        "omega0": omega0,
        "delta": delta,  # 保持不变
        "gamma_rad": gamma_rad,
    }


# ===== 其它设置 =====
K_FIXED = 0.0                      # k = 0
SAVE_SVG = "k0_groups_Prad_Ptot.svg"
FIGSIZE = (5, 3)
fs = 14  # 字体大小
plt.rcParams.update({'font.size': fs})


def compute_curves_for_group(label: str,
                             base_params: dict,
                             group_overrides: dict,
                             omega_grid: np.ndarray):
    """返回 (Prad(omega), Ptot(omega)) 两条曲线（均为 shape=(n_omega,)）"""
    # 组合参数：组内覆盖
    p = dict(base_params)
    p.update(group_overrides)

    # 构建系统
    sp = SystemParams(
        omega0=p["omega0"],
        delta=p["delta"],
        gamma0=p["gamma0"],
        d1=p["d1"],
        d2=p["d2"],
        dispersion_v=p["dispersion_v"],
    )
    model = NonHermitianTwoLevel(sp)
    gamma_rad = float(p["gamma_rad"])

    # 使用你的快速网格函数（对 k=0，只需 1 列）
    if hasattr(model, "power_on_grid_fast"):
        k_vals = np.array([K_FIXED], dtype=float)
        # Prad
        Z_prad = model.power_on_grid_fast("Prad", gamma_rad, k_vals, omega_grid)
        # Ptot
        Z_ptot = model.power_on_grid_fast("Ptot", gamma_rad, k_vals, omega_grid)
        prad_curve = Z_prad[:, 0]
        ptot_curve = Z_ptot[:, 0]
    else:
        # 回退（极少用到）：逐点算
        prad_curve = np.array([model.prad(om, K_FIXED, gamma_rad) for om in omega_grid], dtype=float)
        ptot_curve = np.array([model.ptot(om, K_FIXED, gamma_rad) for om in omega_grid], dtype=float)

    return prad_curve, ptot_curve


def figure_1():
    # 频率网格
    wmin, wmax, n = OMEGA_RANGE["min"], OMEGA_RANGE["max"], int(OMEGA_RANGE["num"])
    omega_grid = np.linspace(wmin, wmax, n, dtype=float)

    # 画图（单图叠加）
    plt.figure(figsize=FIGSIZE)

    curves_info = []  # 记录供必要时后处理
    for idx, (label, overrides) in enumerate(GROUPS.items()):
        prad_curve, ptot_curve = compute_curves_for_group(label, PARAMS, overrides, omega_grid)
        plt.plot(omega_grid, ptot_curve, '-', label=f"{label}  Ptot", color='gray', zorder=0, alpha=0.5)
        plt.plot(omega_grid, prad_curve, '-',  label=f"{label}  Prad", color='black', zorder=10)

    # plt.title(f"k = {K_FIXED}  |  Overlaid Prad & Ptot vs f")
    plt.xlabel("f (c/P)")
    # plt.ylabel("Power")

    # 保存为 SVG
    plt.savefig(SAVE_SVG, format="svg", bbox_inches="tight", transparent=True, dpi=300)
    print(f"Saved: {os.path.abspath(SAVE_SVG)}")
    plt.close()

def figure_2():
    sp_BIC = SystemParams(
        omega0=0,
        delta=0,
        gamma0=0.1,
        d1=1,
        d2=0,
        dispersion_v=1,
    )
    sp_QGM = SystemParams(
        omega0=0,
        delta=0,
        gamma0=0.1,
        d1=0,
        d2=1,
        dispersion_v=1,
    )
    model_BIC = NonHermitianTwoLevel(sp_BIC)
    model_QGM = NonHermitianTwoLevel(sp_QGM)
    intergrated_Prad_BIC_lst = []
    intergrated_Prad_QGM_lst = []
    Prad_QGM_lst = []
    gamma_rad_array = np.linspace(0, 1, 64)
    for gamma_rad in gamma_rad_array:
        # omega = 0, k = 0
        intergrated_Prad_BIC_lst.append(model_BIC.integrate_over_k(model_BIC.prad, gamma_rad=gamma_rad, k_range=(-10, 10), omega_array=[0]))
        # intergrated_Prad_QGM_lst.append(model_QGM.integrate_over_k(model_QGM.prad, gamma_rad=gamma_rad, k_range=(-5, 5), omega_array=[0]))
        intergrated_Prad_QGM_lst.append(model_QGM.integrate_over_k(model_QGM.prad, gamma_rad=gamma_rad, k_range=(-10, 10), omega_array=[0]))
        Prad_QGM_lst.append(model_QGM.prad(omega=0, k=0, gamma_rad=gamma_rad))
    fig, ax = plt.subplots(figsize=(4.5, 2))
    ax.plot(gamma_rad_array, intergrated_Prad_BIC_lst, '-', label=f"Integrated Prad of BIC", color='red')
    ax.plot(gamma_rad_array, intergrated_Prad_QGM_lst, '-', label=f"Integrated Prad of QGM", color='blue')
    # ax.set_ylabel("Integrated $P_{rad}$")
    # ax.set_xlabel("$\gamma$")
    ax_twin = ax.twinx()
    ax_twin.plot(gamma_rad_array, Prad_QGM_lst, '--',  label=f"Prad of QGM", color='gray')
    # ax_twin.set_ylabel("$P_{rad}$")
    plt.savefig("integrated_Prad_vs_gamma_rad.svg", format="svg", bbox_inches="tight", transparent=True, dpi=300)
    plt.show()



if __name__ == "__main__":
    # figure_1()
    figure_2()
