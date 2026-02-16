import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------------
# 模型：用复函数 f 的零点/绕行编码 BIC 与拓扑荷
# ----------------------------
# z = kx + i ky
# f = z^3 * (conj(z)^2 - d^2)
#  - z^3 给 Γ 点 +3 底座电荷
#  - conj(z)^2 - d^2 给两颗简单零点，每个 -1（位置由 d 决定）
#
# d(t):
#  t<0: d = real -> 两颗 -1 在 kx 轴, (±d, 0) 并向 Γ 合并
#  Case1 t>=0: d=0 -> 两颗 -1 不再出现，Γ 等效电荷变 +1 并保持
#  Case2 t>0: d = i*real -> 两颗 -1 在 ky 轴, (0, ±|d|) 重新分裂
#
# 庞加莱球轨迹：由 Jones 向量 (Ex,Ey) -> Stokes (s1,s2,s3)
# 为避免轨迹退化到赤道，引入相对相位 delta(k) 使 S3 != 0


import numpy as np

def d_from_t(t, case, d0=0.6):
    if t < 0:
        return d0 * np.sqrt(-t)      # real: two -1 on kx-axis
    else:
        if case == 1:
            return 0.0               # no re-emergence on ky
        else:
            return 1j * d0 * np.sqrt(t)  # imaginary: re-split on ky-axis

def A_of_t(t, case, tau=0.35):
    # transient merging disturbance decays only for Case 1 after merging
    if case == 1 and t >= 0:
        return np.exp(-t / tau)
    return 1.0

def compute_fields(kx, ky, t, case,
                   d0=0.6, delta0=2.0, kscale=1.2,
                   gamma0=1e-3, beta=5e-3, m=3,
                   alpha=2.0, eta=0.45, tau=0.35):
    """
    Returns:
      psi: polarization director angle (for quiver field)
      s1,s2,s3: normalized Stokes (Poincaré sphere)
      S0: intensity-like quantity (for masking arrows)
      Qeq: effective Q from the NEW loss decomposition model
    """

    # --- (I) Topology / polarization field: keep your original toy field ---
    d = d_from_t(t, case, d0=d0)
    z = kx + 1j * ky
    f = (z ** 3) * (np.conj(z) ** 2 - d ** 2)

    cx, cy = np.real(f), np.imag(f)

    # relative phase to lift the Poincaré trajectory off equator
    delta = delta0 * (kx * ky) / (kscale ** 2)
    Ex = cx.astype(np.complex128)
    Ey = cy.astype(np.complex128) * np.exp(1j * delta)

    S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    S1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = 2 * np.imag(Ex * np.conj(Ey))

    eps = 1e-12
    s1 = S1 / (S0 + eps)
    s2 = S2 / (S0 + eps)
    s3 = S3 / (S0 + eps)

    psi = 0.5 * np.arctan2(S2, S1)

    # --- (II) Effective Q model: NEW (matches your Case-1 observations) ---
    k2 = kx*kx + ky*ky

    # background loss (independent of merging)
    gamma_bg = gamma0 * (1.0 + 0.0 * k2)   # keep it simple; can add anisotropy if needed

    # Gamma "base" BIC channel (high-Q at Gamma)
    gamma_G = beta * (k2 ** m)

    # merging-induced transient disturbance (must vanish on ky-axis: kx=0)
    # choose d_real = |d| (works for both branches)
    dmag = np.abs(d)
    gamma_merge = (kx*kx) * ( (kx*kx - dmag*dmag)**2 + (eta*eta)*(ky*ky) )

    A = A_of_t(t, case, tau=tau)

    gamma_total = gamma_bg + gamma_G + alpha * A * gamma_merge
    Qeq = 1.0 / (gamma_total + eps)

    return psi, s1, s2, s3, S0, Qeq



def bic_points(t, case, d0=0.6):
    pts = []

    # Γ 点等效电荷：由 f 的因子结构决定
    # Case1: t<0 时 Γ 是 +3；t>=0 后两颗 -1 并入 Γ，Γ 变 +1 并保持
    # Case2: t=0 临界时 Γ 等效 +1；t!=0 时两颗 -1 不在 Γ，Γ 显现 +3
    if case == 1:
        qg = 3 if t < 0 else 1
    else:
        qg = 1 if abs(t) < 1e-6 else 3
    pts.append((0.0, 0.0, qg, "Γ"))

    # 两颗 -1
    d = d_from_t(t, case, d0=d0)
    if t < 0:
        dd = float(np.real(d))
        pts.append((+dd, 0.0, -1, "BIC"))
        pts.append((-dd, 0.0, -1, "BIC"))
    else:
        if case == 2 and np.imag(d) != 0:
            ee = float(np.imag(d))
            pts.append((0.0, +ee, -1, "BIC"))
            pts.append((0.0, -ee, -1, "BIC"))
    return pts


# ----------------------------
# 主程序：三幅图 + UI
# ----------------------------
def main():
    # k 空间范围与网格
    kmax = 1.2
    n = 121
    kx = np.linspace(-kmax, kmax, n)
    ky = np.linspace(-kmax, kmax, n)
    KX, KY = np.meshgrid(kx, ky)

    # 绕 Γ 圆周（用于庞加莱轨迹）
    Rcirc = 0.85
    nphi = 500
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=True)
    kx_c = Rcirc * np.cos(phi)
    ky_c = Rcirc * np.sin(phi)

    # 初始状态
    t0 = -0.8
    case0 = 1

    # 计算初始场
    psi, s1, s2, s3, S0, Qeq = compute_fields(
        KX, KY, t0, case0, d0=0.6, delta0=2.0, kscale=kmax
    )

    # mask：接近 BIC 的点（强度太小）不画 director
    mask = S0 < 1e-10
    U = np.cos(psi)
    V = np.sin(psi)
    U = np.where(mask, np.nan, U)
    V = np.where(mask, np.nan, V)

    # Q 热图用对数更稳定：log10 Q
    logQ = np.log10(Qeq)
    # 可视化截断，避免无穷大把颜色压扁
    logQ_clip = np.clip(logQ, 0, 8)

    # Figure Layout: 3行2列（第3行给UI）
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[10, 10, 1.2], width_ratios=[1.05, 1.0])

    ax_pol = fig.add_subplot(gs[0, 0])               # 偏振 director field
    ax_q   = fig.add_subplot(gs[1, 0])               # Q 热图
    ax_s   = fig.add_subplot(gs[0:2, 1], projection='3d')  # 庞加莱球（跨两行）

    # ------------------ (A) 偏振场 ------------------
    ax_pol.set_title("k 空间偏振场（director field）")
    ax_pol.set_xlabel(r"$k_x$")
    ax_pol.set_ylabel(r"$k_y$")
    ax_pol.set_xlim(-kmax, kmax)
    ax_pol.set_ylim(-kmax, kmax)
    ax_pol.set_aspect('equal', 'box')
    ax_pol.axhline(0, lw=0.8)
    ax_pol.axvline(0, lw=0.8)

    # director 用 quiver（稀疏显示）
    step = 4
    qv = ax_pol.quiver(
        KX[::step, ::step], KY[::step, ::step],
        U[::step, ::step], V[::step, ::step],
        pivot='mid', angles='xy', scale_units='xy', scale=18,
        width=0.003
    )

    # 绕 Γ 圆周
    circle_line, = ax_pol.plot(kx_c, ky_c, lw=1.2)

    # BIC 标注（偏振图）
    bic_scatter_pol = ax_pol.scatter([], [], s=90)
    bic_texts_pol = []

    # ------------------ (B) Q 热图 ------------------
    ax_q.set_title(r"等效 $Q$ 因子热图（显示 $\log_{10} Q_{\rm eq}$，并截断到[0,8]）")
    ax_q.set_xlabel(r"$k_x$")
    ax_q.set_ylabel(r"$k_y$")
    ax_q.set_xlim(-kmax, kmax)
    ax_q.set_ylim(-kmax, kmax)
    ax_q.set_aspect('equal', 'box')
    ax_q.axhline(0, lw=0.8)
    ax_q.axvline(0, lw=0.8)

    im = ax_q.imshow(
        logQ_clip,
        origin="lower",
        extent=[-kmax, kmax, -kmax, kmax],
        interpolation="nearest"
    )
    cb = fig.colorbar(im, ax=ax_q, fraction=0.046, pad=0.02)
    cb.set_label(r"$\log_{10} Q_{\rm eq}$ (clipped)")

    # BIC 标注（Q图）
    bic_scatter_q = ax_q.scatter([], [], s=90)
    bic_texts_q = []

    # ------------------ (C) 庞加莱球 + 方向箭头 ------------------
    ax_s.set_title("绕 Γ 圆周的偏振庞加莱球轨迹（箭头示方向）")
    ax_s.set_xlabel("S1")
    ax_s.set_ylabel("S2")
    ax_s.set_zlabel("S3")
    ax_s.set_xlim(-1, 1)
    ax_s.set_ylim(-1, 1)
    ax_s.set_zlim(-1, 1)
    ax_s.set_box_aspect((1, 1, 1))

    # 球面 wireframe
    uu = np.linspace(0, 2*np.pi, 30)
    vv = np.linspace(0, np.pi, 18)
    UU, VV = np.meshgrid(uu, vv)
    Xs = np.cos(UU) * np.sin(VV)
    Ys = np.sin(UU) * np.sin(VV)
    Zs = np.cos(VV)
    ax_s.plot_wireframe(Xs, Ys, Zs, rstride=2, cstride=2, linewidth=0.4, alpha=0.6)

    # 初始轨迹
    _, s1c, s2c, s3c, S0c, Qc = compute_fields(kx_c, ky_c, t0, case0, d0=0.6, delta0=2.0, kscale=kmax)
    traj_line, = ax_s.plot(s1c, s2c, s3c, lw=2.0)
    start_pt = ax_s.scatter([s1c[0]], [s2c[0]], [s3c[0]], s=40)

    # 轨迹方向箭头（3D quiver，采样若干点）
    traj_quiver = None

    def build_traj_arrows(x, y, z, n_arrows=12):
        """沿轨迹放置 n_arrows 个箭头，方向取切向量。"""
        # 取均匀索引（避开最后一个点）
        idx = np.linspace(0, len(x) - 2, n_arrows, dtype=int)
        xs = x[idx]
        ys = y[idx]
        zs = z[idx]

        dx = x[idx + 1] - x[idx]
        dy = y[idx + 1] - y[idx]
        dz = z[idx + 1] - z[idx]

        # 归一化切向量，避免箭头长短差异
        norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
        dxn, dyn, dzn = dx / norm, dy / norm, dz / norm

        # length 是统一比例
        return ax_s.quiver(xs, ys, zs, dxn, dyn, dzn, length=0.18, arrow_length_ratio=0.35, linewidth=1.2)

    traj_quiver = build_traj_arrows(s1c, s2c, s3c, n_arrows=12)

    # ------------------ UI：滑块 + 单选 ------------------
    ui_ax = fig.add_subplot(gs[2, :])
    ui_ax.axis("off")

    slider_ax = fig.add_axes([0.12, 0.05, 0.62, 0.035])
    t_slider = Slider(slider_ax, "t", -1.0, 1.0, valinit=t0, valstep=0.01)

    radio_ax = fig.add_axes([0.78, 0.02, 0.20, 0.12])
    radio = RadioButtons(radio_ax, ("Case 1: 合并后不再出现", "Case 2: 合并后在ky轴分开"), active=0)

    def redraw_bic_labels(ax, scatter, texts_list, t, case):
        for tx in texts_list:
            tx.remove()
        texts_list.clear()

        pts = bic_points(t, case, d0=0.6)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scatter.set_offsets(np.c_[xs, ys])

        for (x, y, q, tag) in pts:
            texts_list.append(ax.text(x + 0.03, y + 0.03, f"{tag}  q={q:+d}", fontsize=10))

    redraw_bic_labels(ax_pol, bic_scatter_pol, bic_texts_pol, t0, case0)
    redraw_bic_labels(ax_q,   bic_scatter_q,   bic_texts_q,   t0, case0)

    def update(_):
        nonlocal traj_quiver

        case = 1 if radio.value_selected.startswith("Case 1") else 2
        t = t_slider.val

        # --- 更新偏振 director field ---
        psi, s1, s2, s3, S0, Qeq = compute_fields(
            KX, KY, t, case, d0=0.6, delta0=2.0, kscale=kmax
        )
        mask = S0 < 1e-10
        U = np.cos(psi)
        V = np.sin(psi)
        U = np.where(mask, np.nan, U)
        V = np.where(mask, np.nan, V)
        qv.set_UVC(U[::step, ::step], V[::step, ::step])

        # --- 更新 Q 热图 ---
        logQ = np.log10(Qeq)
        logQ_clip = np.clip(logQ, 0, 8)
        im.set_data(logQ_clip)

        # --- 更新 BIC 标签 ---
        redraw_bic_labels(ax_pol, bic_scatter_pol, bic_texts_pol, t, case)
        redraw_bic_labels(ax_q,   bic_scatter_q,   bic_texts_q,   t, case)

        # --- 更新庞加莱球轨迹 + 箭头方向 ---
        _, s1c, s2c, s3c, S0c, Qc = compute_fields(
            kx_c, ky_c, t, case, d0=0.6, delta0=2.0, kscale=kmax
        )
        traj_line.set_data(s1c, s2c)
        traj_line.set_3d_properties(s3c)
        start_pt._offsets3d = ([s1c[0]], [s2c[0]], [s3c[0]])

        # 删除旧箭头并重建
        if traj_quiver is not None:
            traj_quiver.remove()
        traj_quiver = build_traj_arrows(s1c, s2c, s3c, n_arrows=12)

        fig.canvas.draw_idle()

    t_slider.on_changed(update)
    radio.on_clicked(update)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
