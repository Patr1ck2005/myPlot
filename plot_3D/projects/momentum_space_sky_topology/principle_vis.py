import numpy as np
import matplotlib.pyplot as plt

from utils.functions import skyrmion_density, skyrmion_number

# 你的“物理基底”：V 点分裂成两 C 点
def jones_field_base(kx, ky, kxc=0.0, kyc=0.0, amp=1.0):
    EL = (kx - kxc) + 1j*(ky - kyc)
    ER = (kx + kxc) - 1j*(ky - kyc)
    Ex = amp*(EL + ER)/np.sqrt(2.0)
    Ey = amp*1j*(EL - ER)/np.sqrt(2.0)
    return Ex, Ey

# 从 Jones 计算 Stokes 与角度
def _stokes_angles(Ex, Ey, eps=1e-12):
    S0 = (np.abs(Ex)**2 + np.abs(Ey)**2) + eps
    S1 = (np.abs(Ex)**2 - np.abs(Ey)**2)
    S2 = 2.0*np.real(Ex*np.conj(Ey))
    S3 = 2.0*np.imag(Ex*np.conj(Ey))
    phi = 0.5*np.arctan2(S2, S1)                              # 取向角
    chi = 0.5*np.arcsin(np.clip(S3/S0, -1.0, 1.0))            # 椭圆率角
    return S0, S1, S2, S3, phi, chi

# 用 (S0, phi, chi) 重建 Jones（单位相位不重要）
def _jones_from_phi_chi(S0, phi, chi):
    # 经典参数化：见任意偏振学教材（保持与上面 phi/chi 定义一致）
    cφ, sφ = np.cos(phi), np.sin(phi)
    cχ, sχ = np.cos(chi), np.sin(chi)
    Ex_hat = cφ*cχ - 1j*sφ*sχ
    Ey_hat = sφ*cχ + 1j*cφ*sχ
    amp = np.sqrt(np.maximum(S0, 0.0))
    return amp*Ex_hat, amp*Ey_hat

# —— 你要的“平方/立方并保符号”的 jones_field ——
def jones_field(kx, ky, kxc=0.0, kyc=0.0, amp=1.0, p=1.0):
    """
    先用物理模型生成 (Ex,Ey)，再把 s=S3/S0 做非线性：s -> sign(s)*|s|^p，
    并据此仅修改椭圆率角 chi，保持 phi 与 S0 不变，最后重建 Jones。
    - p=2（平方保符号）、p=3（立方）等均可，p=1 即原始。
    """
    Ex0, Ey0 = jones_field_base(kx, ky, kxc=kxc, kyc=kyc, amp=amp)
    S0, S1, S2, S3, phi, chi = _stokes_angles(Ex0, Ey0)

    s = np.clip(S3/S0, -1.0, 1.0)
    s_new = np.sign(s) * np.power(np.abs(s), p)               # 结果目标
    chi_new = 0.5*np.arcsin(np.clip(s_new, -1.0, 1.0))        # 由 s_new 反推新椭圆率

    Ex, Ey = _jones_from_phi_chi(S0, phi, chi_new)            # 保持 S0 与 phi
    return Ex, Ey

def stokes_from_jones(Ex, Ey, eps=1e-12):
    S0 = (np.abs(Ex)**2 + np.abs(Ey)**2) + eps
    S1 = (np.abs(Ex)**2 - np.abs(Ey)**2)
    S2 = 2.0*np.real(Ex*np.conj(Ey))
    S3 = 2.0*np.imag(Ex*np.conj(Ey))
    # 角度：phi（取向角，周期 π），chi（椭圆率角，[-pi/4, pi/4]）
    phi = 0.5*np.arctan2(S2, S1)               # [-pi/2, pi/2]
    chi = 0.5*np.arcsin(np.clip(S3/S0, -1.0, 1.0))
    return S0, S1, S2, S3, phi, chi

# ---------- 画图函数（均接收 ax 并返回 ax，便于链式） ----------
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from matplotlib import cm

def _wrap_angle_pi(phi):
    """把取向角规约到 (-pi/2, pi/2]，避免跨分支时跳变过大。"""
    return (phi + np.pi/2) % np.pi - np.pi/2

def plot_polarization_ellipses(
    ax, kx, ky, Ex, Ey,
    step=(6, 6),            # 栅格抽样步长（x方向、y方向）
    scale=None,             # 椭圆“直径”基准（数据坐标）；默认 0.8*min(dx,dy)
    cmap='RdBu',            # 颜色图：按 S3/S0 上色
    clim=(-1.0, 1.0),       # 颜色范围（vmin, vmax），默认映射到 [-1, 1]
    lw=1.2, alpha=0.9, zorder=2
):
    """
    在给定 Axes 上以贴片方式绘制偏振椭圆场。
    - 输入为 Jones 场 Ex, Ey（与现有程序保持一致）
    - 椭圆几何来自 (phi, chi) 的标准定义：
        phi = 0.5*atan2(S2, S1)      线偏振取向角（mod π）
        chi = 0.5*arcsin(S3/S0)      椭圆率角（∈ [-pi/4, pi/4]）
      轴比 b/a = |tan(chi)|
    - 颜色：按 S3/S0 用 RdBu 映射（红/蓝表示手性）
    返回 ax（可链式）。
    """
    # ---- Stokes 与角度 ----
    S0 = (np.abs(Ex)**2 + np.abs(Ey)**2)
    S1 = (np.abs(Ex)**2 - np.abs(Ey)**2)
    S2 = 2.0*np.real(Ex*np.conj(Ey))
    S3 = 2.0*np.imag(Ex*np.conj(Ey))

    phi = 0.5*np.arctan2(S2, S1)
    phi = _wrap_angle_pi(phi)
    chi = 0.5*np.arcsin(np.clip(S3/(S0 + 1e-12), -1.0, 1.0))   # 椭圆率角

    # ---- 网格轴向（假设 kx,ky 来自 meshgrid，且规则网格）----
    # 用第一行/列抽出坐标轴刻度
    xg = kx[0, :]
    yg = ky[:, 0]
    # 网格间距与默认 scale
    dx = np.min(np.diff(xg)) if xg.size > 1 else 1.0
    dy = np.min(np.diff(yg)) if yg.size > 1 else 1.0
    if scale is None:
        scale = 0.8 * min(dx, dy)

    sx, sy = step if isinstance(step, (tuple, list)) else (step, step)
    xs = range(0, xg.size, max(1, int(sx)))
    ys = range(0, yg.size, max(1, int(sy)))

    # ---- 构建贴片 ----
    patches = []
    color_vals = []
    for j in ys:       # 注意：行是 y 索引
        for i in xs:   # 列是 x 索引
            phi_ij = phi[j, i]
            chi_ij = chi[j, i]
            # 椭圆几何：a 为主轴直径，b 为副轴直径
            a = scale
            b = scale * abs(np.tan(chi_ij))
            b = max(b, 1e-3*scale)  # 下限，避免完全退化

            e = Ellipse((xg[i], yg[j]), width=a, height=b,
                        angle=np.degrees(phi_ij))
            patches.append(e)

            # 颜色按 S3/S0
            color_vals.append(np.clip(S3[j, i]/(S0[j, i] + 1e-12), -1.0, 1.0))

    pc = PatchCollection(patches, linewidths=lw, alpha=alpha, zorder=zorder)
    cmap_obj = cm.get_cmap(cmap)
    vmin, vmax = clim
    normed = (np.asarray(color_vals) - vmin) / (vmax - vmin + 1e-12)
    facecolors = cmap_obj(np.clip(normed, 0, 1))
    pc.set_facecolor(facecolors)
    pc.set_edgecolor(facecolors)

    ax.add_collection(pc)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title('Polarization ellipse field (colored by $S_3/S_0$, RdBu)')
    return ax


def imshow_phi(ax, phi, extent=None, vmin=-np.pi/2, vmax=np.pi/2, **imshow_kwargs):
    """
    可视化取向角 phi（[-pi/2, pi/2]），可能存在分支切换。
    extent: 对应 imshow 的坐标范围
    允许通过 **imshow_kwargs 传入插值等参数；函数返回 ax 以支持链式。
    """
    im = ax.imshow(phi, origin='lower', extent=extent, vmin=vmin, vmax=vmax, **imshow_kwargs)
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title(r'Orientation angle $\phi$')
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax

def imshow_S3(ax, S3, S0=None, extent=None, vmin=-1.0, vmax=1.0, **imshow_kwargs):
    """
    可视化 S3 或 S3/S0（默认归一化到 [-1,1]）。
    extent: 对应 imshow 的坐标范围
    """
    if S0 is not None:
        val = np.clip(S3/(S0 + 1e-12), -1.0, 1.0)
    else:
        # 假定 S3 已在 [-1,1]
        val = np.clip(S3, -1.0, 1.0)
    im = ax.imshow(val, origin='lower', extent=extent, vmin=vmin, vmax=vmax, **imshow_kwargs)
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title(r'$S_3$ (helicity)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax

# ---------- 示例：从 V 点到两 C 点的分裂 ----------
if __name__ == "__main__":
    # 网格（k 空间）
    n = 201
    kmax = 5.0
    dk = 2.0*kmax/(n-1)
    x = np.linspace(-kmax, kmax, n)
    y = np.linspace(-kmax, kmax, n)
    kx, ky = np.meshgrid(x, y, indexing='xy')
    extent = [x.min(), x.max(), y.min(), y.max()]

    # 调节分裂位移： (kxc, kyc)
    kxc = 0.4   # C 点分别在 (+kxc, kyc) 与 (-kxc, kyc)
    kyc = 0.1

    # 生成 Jones 场
    Ex, Ey = jones_field(kx, ky, kxc=kxc, kyc=kyc, amp=1.0)
    S0, S1, S2, S3, phi, chi = stokes_from_jones(Ex, Ey)

    # 绘图：三种图，互不干扰，也支持你按需链式调用
    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2,2,1)
    plot_polarization_ellipses(ax1, kx, ky, Ex, Ey, step=6, scale=0.1)

    ax2 = fig.add_subplot(2,2,2)
    imshow_phi(ax2, phi, extent=extent, interpolation='nearest', cmap='hsv')

    ax3 = fig.add_subplot(2,2,3)
    imshow_S3(ax3, S3, S0=S0, extent=extent, interpolation='nearest', cmap='RdBu')

    # 在图上标出两处 C 点位置（仅作为参考标记）
    for ax in [ax1, ax2, ax3]:
        ax.plot([+kxc, -kxc], [kyc, kyc], 'ko', ms=0)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(r'V-point ($q=1$) $\to$ two C-points ($q=1/2$) splitting at $\,(k_x,k_y)=(\pm k_{x,c},\,k_{y,c})$', y=0.98)
    plt.tight_layout()
    plt.show()

    left_regime_mask = kx < 0
    nsk = skyrmion_density(S1/S0, S2/S0, S3/S0)
    nsk_left = nsk * left_regime_mask
     # 绘制斯格明子密度
    fig = plt.figure(figsize=(6,5))
    # plt.imshow(nsk, origin='lower', extent=extent, cmap='viridis')
    plt.imshow(nsk_left, origin='lower', cmap='viridis')
    plt.colorbar(label='Skyrmion density')
    plt.title('Skyrmion density')
    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$k_y$')
    plt.show()
    print(nsk_left.sum())
    sk_num = skyrmion_number(nsk_left, 1, 1)
