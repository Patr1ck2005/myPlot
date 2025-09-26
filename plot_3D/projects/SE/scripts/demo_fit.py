# quick_fit_viewer.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pickle
import os

# ===== 你的物理内核 =====
from physics_core import SystemParams, NonHermitianTwoLevel, get_power_function


# =============== 数据加载（按你给的函数） ===============
def load_data(data_path):
    with open(data_path, 'rb') as f:
        raw_dataset = pickle.load(f)
    x_vals = raw_dataset.get('x_vals', np.array([]))  # k 轴？
    y_vals = raw_dataset.get('y_vals', np.array([]))  # freq 轴？
    subs = raw_dataset.get('subs', [])
    print(f"Pickle基础提取: x_shape={x_vals.shape}, subs_len={len(subs)} 🔍")
    z_data = subs[0].real  # (k_space, freq_space)
    return x_vals, y_vals, z_data


# =============== 计算模型网格（与数据网格同形状） ===============
def compute_model_on_axes(sp, gamma_rad, k_grid, omega_grid, mode, dispersion_v=1.0):
    # 保持向后兼容：SystemParams 若无 dispersion_v，这里已在你的代码里兜底
    try:
        sp = SystemParams(omega0=sp.omega0, delta=sp.delta, gamma0=sp.gamma0,
                          d1=sp.d1, d2=sp.d2, dispersion_v=dispersion_v)
    except TypeError:
        pass
    model = NonHermitianTwoLevel(sp)
    if not hasattr(model.sp, "dispersion_v"):
        setattr(model.sp, "dispersion_v", float(dispersion_v))

    # === NEW: 超快向量化路径 ===
    Z = model.power_on_grid_fast(mode, gamma_rad, np.asarray(k_grid, float), np.asarray(omega_grid, float))
    return Z



# =============== 主程序 ===============
def main(data_path):
    # 1) 加载数据
    k_vals_raw, freq_vals_raw, z_data_raw = load_data(data_path)
    # (k, freq) -> (freq, k)，并做简单归一化
    denom = np.max(z_data_raw) if np.max(z_data_raw) != 0 else 1.0
    dataZ = (z_data_raw.T.copy()) / denom
    k_grid = np.array(k_vals_raw, dtype=float)
    omega_grid = np.array(freq_vals_raw, dtype=float)

    # 形状兜底（最近邻）
    if dataZ.shape != (omega_grid.size, k_grid.size):
        print("⚠️ 发现形状与轴长度不匹配，进行简易重采样（最近邻）……")
        tgt = np.zeros((omega_grid.size, k_grid.size), dtype=float)
        si, sj = dataZ.shape
        for i in range(tgt.shape[0]):
            ii = int(round(i * (si - 1) / max(tgt.shape[0] - 1, 1)))
            for j in range(tgt.shape[1]):
                jj = int(round(j * (sj - 1) / max(tgt.shape[1] - 1, 1)))
                tgt[i, j] = dataZ[ii, jj]
        dataZ = tgt

    # 2) 初始参数（延用你的起点，更贴近当前数据）
    params = {
        "omega0": 0.4413,
        "delta":  0.00585,
        "gamma0": 1.2e-4,
        "d1":     1.0,
        "d2":     0.0,
        "gamma_rad": 3.5e-4,
        "dispersion_v": 0.29,
    }
    mode = ["Prad"]
    # 切片索引初值：取中间
    k_idx = [k_grid.size // 2]
    w_idx = [omega_grid.size // 2]

    # 3) 画布与总布局（顶部三图；底部：左/中为切片，右为控制面板）
    plt.rcParams['figure.figsize'] = (14, 8)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.25, 1.0],  # 顶部更高
        width_ratios=[1.0, 1.0, 1.0]
    )

    # 顶部热图
    ax_data  = fig.add_subplot(gs[0, 0])
    ax_model = fig.add_subplot(gs[0, 1])
    ax_diff  = fig.add_subplot(gs[0, 2])

    # 底部：两幅切片图 + 控制面板
    ax_line1 = fig.add_subplot(gs[1, 0])
    ax_line2 = fig.add_subplot(gs[1, 1])
    ax_ctrls = fig.add_subplot(gs[1, 2])
    ax_ctrls.set_title("Controls", pad=10)
    ax_ctrls.set_xticks([]); ax_ctrls.set_yticks([])
    for spine in ax_ctrls.spines.values():
        spine.set_alpha(0.3)

    # 4) 初次计算模型
    sp = SystemParams(omega0=params["omega0"], delta=params["delta"],
                      gamma0=params["gamma0"], d1=params["d1"], d2=params["d2"])
    Zm = compute_model_on_axes(sp, params["gamma_rad"], k_grid, omega_grid, mode[0],
                               dispersion_v=params["dispersion_v"])
    Zm_norm = Zm / (np.max(Zm) if np.max(Zm) != 0 else 1.0)
    diffZ = Zm_norm - dataZ

    # 统一色标（0~1），跟你保持一致
    vmin, vmax = 0.0, 1.0

    im_data  = ax_data.imshow(
        dataZ, origin="lower", aspect="auto",
        extent=[k_grid.min(), k_grid.max(), omega_grid.min(), omega_grid.max()],
        vmin=vmin, vmax=vmax, interpolation="none"
    )
    ax_data.set_title("Data (heatmap)")
    ax_data.set_xlabel("k"); ax_data.set_ylabel("omega")

    im_model = ax_model.imshow(Zm_norm, origin="lower", aspect="auto",
                               extent=[k_grid.min(), k_grid.max(), omega_grid.min(), omega_grid.max()],
                               vmin=vmin, vmax=vmax, interpolation="none")
    ax_model.set_title("Model (heatmap)")
    ax_model.set_xlabel("k"); ax_model.set_ylabel("omega")

    im_diff  = ax_diff.imshow(diffZ, origin="lower", aspect="auto",
                              extent=[k_grid.min(), k_grid.max(), omega_grid.min(), omega_grid.max()], interpolation="none")
    ax_diff.set_title("Difference (Model - Data)")
    ax_diff.set_xlabel("k"); ax_diff.set_ylabel("omega")

    fig.colorbar(im_data,  ax=ax_data,  fraction=0.046, pad=0.04)
    fig.colorbar(im_model, ax=ax_model, fraction=0.046, pad=0.04)
    fig.colorbar(im_diff,  ax=ax_diff,  fraction=0.046, pad=0.04)

    # 5) 控制面板内创建滑块/按钮/单选（相对 ax_ctrls 的位置自动摆放）
    #    ——— 将 ax_ctrls 的 bbox 作为参考系，等距垂直堆叠控件 ———
    def add_in_panel(y_rel_top, height, kind="slider", label="", vmin=0, vmax=1, val=0.0, choices=None):
        """
        在 ax_ctrls 内按相对坐标添加控件。
        y_rel_top: [0,1] 控件顶部相对位置（1=顶部，0=底部）
        height:    控件相对高度（相对 ax_ctrls 高度）
        kind:      'slider' | 'radio' | 'button'
        """
        panel = ax_ctrls.get_position()
        left   = panel.x0 + 0.06 * panel.width
        width  = panel.width * 0.88
        top    = panel.y0 + y_rel_top * panel.height
        bottom = top - height * panel.height
        rect = [left, bottom, width, height * panel.height]

        if kind == "slider":
            ax = fig.add_axes(rect)
            return Slider(ax, label, vmin, vmax, valinit=val)
        elif kind == "radio":
            ax = fig.add_axes(rect)
            rb = RadioButtons(ax, choices or ("Prad", "Ptot"), active=0)
            return rb
        elif kind == "button":
            ax = fig.add_axes(rect)
            return Button(ax, label)
        else:
            raise ValueError("Unknown control kind")

    # 控件竖向布局参数
    ROW = 0.065  # 单个控件相对高度
    GAP = 0.012  # 控件之间相对间距

    y_top = 0.96  # 从面板顶部往下排

    # 模式选择（占两行高度）
    r_mode = add_in_panel(y_top, ROW*1.6, kind="radio", label="", choices=("Prad", "Ptot"))
    y_top -= (ROW*1.6 + GAP)

    # 六个物理参数滑块
    s_omega0 = add_in_panel(y_top, ROW, "slider", "omega0", params["omega0"]*0.8, params["omega0"]*1.2, params["omega0"])
    y_top -= (ROW + GAP)

    # delta 更紧凑些（易调细节），如需恢复可改回 0~1
    s_delta  = add_in_panel(y_top, ROW, "slider", "delta", params["delta"]*0.5, params["delta"]*2, params["delta"])
    y_top -= (ROW + GAP)

    s_gamma0 = add_in_panel(y_top, ROW, "slider", "gamma0", params["gamma0"]*0.5, params["gamma0"]*2, params["gamma0"])
    y_top -= (ROW + GAP)

    s_d1     = add_in_panel(y_top, ROW, "slider", "d1", -2.0, 2.0, params["d1"])
    y_top -= (ROW + GAP)

    s_d2     = add_in_panel(y_top, ROW, "slider", "d2", -2.0, 2.0, params["d2"])
    y_top -= (ROW + GAP)

    # gamma_rad 初期范围略收紧，细调更顺手（如需宽可改成 1e-5~2.0）
    s_grad   = add_in_panel(y_top, ROW, "slider", "gamma_rad", params["gamma_rad"]*0.5, params["gamma_rad"]*2, params["gamma_rad"])
    y_top -= (ROW + GAP)

    # dispersion_v 常用在 0~1（如需更宽可改回 0~5）
    s_v      = add_in_panel(y_top, ROW, "slider", "dispersion_v", params["dispersion_v"]*0.5, params["dispersion_v"]*2, params["dispersion_v"])
    y_top -= (ROW + GAP)

    # 切片索引滑块（各占一行）
    s_kcut = add_in_panel(y_top, ROW, "slider", "k-index", 0, k_grid.size - 1, k_idx[0])
    s_kcut.valstep = 1
    y_top -= (ROW + GAP)

    s_wcut = add_in_panel(y_top, ROW, "slider", "omega-index", 0, omega_grid.size - 1, w_idx[0])
    s_wcut.valstep = 1
    y_top -= (ROW + GAP)

    # 刷新按钮
    b_refresh = add_in_panel(y_top, ROW*0.9, "button", "Refresh")
    y_top -= (ROW*0.9 + GAP)

    # 6) 切片初绘
    oi = np.clip(w_idx[0], 0, omega_grid.size - 1)
    kj = np.clip(k_idx[0], 0, k_grid.size - 1)

    ax_line1.plot(k_grid, dataZ[oi, :], label="Data")
    ax_line1.plot(k_grid, Zm_norm[oi, :], label="Model")
    ax_line1.set_title(f"k-cut @ omega={omega_grid[oi]:.4g}")
    ax_line1.set_xlabel("k"); ax_line1.set_ylabel("Power"); ax_line1.legend()

    ax_line2.plot(omega_grid, dataZ[:, kj], label="Data")
    ax_line2.plot(omega_grid, Zm_norm[:, kj], label="Model")
    ax_line2.set_title(f"omega-cut @ k={k_grid[kj]:.4g}")
    ax_line2.set_xlabel("omega"); ax_line2.set_ylabel("Power"); ax_line2.legend()

    # 7) 更新函数
    def update(_evt=None):
        params["omega0"]    = float(s_omega0.val)
        params["delta"]     = float(s_delta.val)
        params["gamma0"]    = float(s_gamma0.val)
        params["d1"]        = float(s_d1.val)
        params["d2"]        = float(s_d2.val)
        params["gamma_rad"] = float(s_grad.val)
        params["dispersion_v"] = float(s_v.val)
        mode[0]             = r_mode.value_selected
        k_idx[0]            = int(s_kcut.val)
        w_idx[0]            = int(s_wcut.val)

        sp = SystemParams(omega0=params["omega0"], delta=params["delta"],
                          gamma0=params["gamma0"], d1=params["d1"], d2=params["d2"])
        Zm = compute_model_on_axes(sp, params["gamma_rad"], k_grid, omega_grid, mode[0],
                                   dispersion_v=params["dispersion_v"])
        Zm_norm = Zm / (np.max(Zm) if np.max(Zm) != 0 else 1.0)
        diffZ = Zm_norm - dataZ

        im_model.set_data(Zm_norm)
        im_diff.set_data(diffZ)

        # 更新切片
        ax_line1.cla(); ax_line2.cla()
        oi = np.clip(w_idx[0], 0, omega_grid.size - 1)
        kj = np.clip(k_idx[0], 0, k_grid.size - 1)

        ax_line1.plot(k_grid, dataZ[oi, :], label="Data")
        ax_line1.plot(k_grid, Zm_norm[oi, :], label="Model")
        ax_line1.set_title(f"k-cut @ omega={omega_grid[oi]:.4g}")
        ax_line1.set_xlabel("k"); ax_line1.set_ylabel("Power"); ax_line1.legend()

        ax_line2.plot(omega_grid, dataZ[:, kj], label="Data")
        ax_line2.plot(omega_grid, Zm_norm[:, kj], label="Model")
        ax_line2.set_title(f"omega-cut @ k={k_grid[kj]:.4g}")
        ax_line2.set_xlabel("omega"); ax_line2.set_ylabel("Power"); ax_line2.legend()

        fig.canvas.draw_idle()

    # 绑定事件
    for sl in (s_omega0, s_delta, s_gamma0, s_d1, s_d2, s_grad, s_v, s_kcut, s_wcut):
        sl.on_changed(update)
    r_mode.on_clicked(update)
    b_refresh.on_clicked(update)

    # 初次更新
    update()
    plt.suptitle("Quick Trend-First Viewer (drag sliders to adjust)")
    plt.show()


if __name__ == "__main__":
    # data_path = r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\PL\3fold_PL_QGM-rad.pkl"
    data_path = r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\PL\3fold_PL_BIC-rad.pkl"
    if not os.path.exists(data_path):
        print("请修改 data_path 为你的 pickle 文件路径。")
    main(data_path)
