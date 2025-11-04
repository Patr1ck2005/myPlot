import numpy as np
from matplotlib import pyplot as plt

from advance_plot_styles.polar_plot import *
from utils.advanced_color_mapping import map_s1s2s3_color
# 可选：你已有的工具
from utils.functions import skyrmion_density, skyrmion_number

# =========================
# 物理基底与辅助函数
# =========================

def jones_field_base(kx, ky, kxc=0.0, kyc=0.0, amp=1.0):
    """V 点（q=1）沿动量空间分裂成两 C 点（q=1/2）的最简物理基底。"""
    EL = (kx - kxc) + 1j*(ky - kyc)
    ER = (kx + kxc) - 1j*(ky - kyc)
    Ex = amp*(EL + ER)/np.sqrt(2.0)
    Ey = amp*1j*(EL - ER)/np.sqrt(2.0)
    return Ex, Ey

def _stokes_angles(Ex, Ey, eps=1e-12):
    S0 = (np.abs(Ex)**2 + np.abs(Ey)**2) + eps
    S1 = (np.abs(Ex)**2 - np.abs(Ey)**2)
    S2 = 2.0*np.real(Ex*np.conj(Ey))
    S3 = 2.0*np.imag(Ex*np.conj(Ey))
    phi = 0.5*np.arctan2(S2, S1)                              # 取向角
    chi = 0.5*np.arcsin(np.clip(S3/S0, -1.0, 1.0))            # 椭圆率角
    return S0, S1, S2, S3, phi, chi

def _jones_from_phi_chi(S0, phi, chi):
    """由 (S0,phi,chi) 重建 Jones. 相对整体相位无关紧要。"""
    cφ, sφ = np.cos(phi), np.sin(phi)
    cχ, sχ = np.cos(chi), np.sin(chi)
    Ex_hat = cφ*cχ - 1j*sφ*sχ
    Ey_hat = sφ*cχ + 1j*cφ*sχ
    amp = np.sqrt(np.maximum(S0, 0.0))
    return amp*Ex_hat, amp*Ey_hat

def stokes_from_jones(Ex, Ey, eps=1e-12):
    S0 = (np.abs(Ex)**2 + np.abs(Ey)**2) + eps
    S1 = (np.abs(Ex)**2 - np.abs(Ey)**2)
    S2 = 2.0*np.real(Ex*np.conj(Ey))
    S3 = 2.0*np.imag(Ex*np.conj(Ey))
    phi = 0.5*np.arctan2(S2, S1)
    chi = 0.5*np.arcsin(np.clip(S3/S0, -1.0, 1.0))
    return S0, S1, S2, S3, phi, chi

# —— “从结果出发”的幂映射 jones_field ——
def jones_field(kx, ky, kxc=0.0, kyc=0.0, amp=1.0, p=1.0):
    """
    先生成 (Ex,Ey)，再把 s=S3/S0 做非线性 s -> sgn(s)*|s|^p，
    仅修改椭圆率角 chi，保持 phi 与 S0 不变，最后重建 Jones。
    p=2（平方）、p=3（立方）等；p=1 为原始。
    """
    Ex0, Ey0 = jones_field_base(kx, ky, kxc=kxc, kyc=kyc, amp=amp)
    S0, S1, S2, S3, phi, chi = _stokes_angles(Ex0, Ey0)

    s = np.clip(S3/S0, -1.0, 1.0)
    s_new = np.sign(s) * np.power(np.abs(s), p)
    chi_new = 0.5*np.arcsin(np.clip(s_new, -1.0, 1.0))

    Ex, Ey = _jones_from_phi_chi(S0, phi, chi_new)
    return Ex, Ey


# =========================
# 示例（每个图独立）
# =========================
if __name__ == "__main__":

    # 网格
    n = 201
    kmax = 2.0
    x = np.linspace(-kmax, kmax, n)
    y = np.linspace(-kmax, kmax, n)
    kx, ky = np.meshgrid(x, y, indexing='ij')
    extent = [x.min(), x.max(), y.min(), y.max()]

    # C 点位置
    kxc, kyc = 0.4, 0.1

    # 生成 Jones（可通过 p 控制 S3/S0 的陡峭度）
    Ex, Ey = jones_field(kx, ky, kxc=kxc, kyc=kyc, amp=1.0, p=1.0)
    S0, S1, S2, S3, phi, chi = stokes_from_jones(Ex, Ey)
    s1, s2, s3 = S1/(S0 + 1e-12), S2/(S0 + 1e-12), S3/(S0 + 1e-12)

    # 1) 椭圆贴片（独立成图）
    fig1, ax1 = plt.subplots(figsize=(3, 3))
    plot_polarization_ellipses(ax1, kx, ky, s1, s2, s3, S0=S0, step=(10,10), scale=0.15)
    ax1.plot([+kxc, -kxc], [kyc, kyc], 'k.', ms=6)  # 标记 C 点
    ax1.set_title('Polarization ellipses')
    plt.savefig("temp.svg", bbox_inches='tight', transparent=True)
    plt.show()

    #
    fig1, ax1 = plt.subplots(figsize=(3, 3))
    rbg = map_s1s2s3_color(s1, s2, s3)
    ax1.imshow(rbg, origin='lower', extent=extent)
    ax1.set_title('Hyper colormap')
    plt.savefig("temp.svg", bbox_inches='tight', transparent=True)
    plt.show()

    # 2) phi 的 imshow（独立成图）
    fig2, ax2 = plt.subplots(figsize=(3, 3))
    imshow_phi(ax2, phi, extent=extent, interpolation='nearest', cmap='hsv')
    ax2.plot([+kxc, -kxc], [kyc, kyc], 'k.', ms=6)
    plt.savefig("temp.svg", bbox_inches='tight', transparent=True)
    plt.show()

    # 3) S3/S0 的 imshow（独立成图）
    fig3, ax3 = plt.subplots(figsize=(3, 3))
    imshow_s3(ax3, s3, S0=S0, extent=extent, interpolation='nearest', cmap='RdBu')
    ax3.plot([+kxc, -kxc], [kyc, kyc], 'k.', ms=6)
    plt.savefig("temp.svg", bbox_inches='tight', transparent=True)
    plt.show()

    # 4) 斯格明子纹理（箭头）（独立成图）
    fig4, ax4 = plt.subplots(figsize=(3, 3))
    plot_skyrmion_quiver(ax4, kx, ky, s1, s2, s3, S0=S0, step=(10,10),
                         normalize=True, cmap='RdBu', clim=(-1,1),
                         quiver_scale=None, width=0.010)
    ax4.set_xlim(extent[0], extent[1]); ax4.set_ylim(extent[2], extent[3])
    ax4.plot([+kxc, -kxc], [kyc, kyc], 'k.', ms=6)
    plt.savefig("temp.svg", bbox_inches='tight', transparent=True)
    plt.show()

    # 5) 投到 Poincaré 球面（独立成图）
    fig5 = plt.figure(figsize=(6, 6))
    ax5 = fig5.add_subplot(111, projection='3d')
    plot_on_poincare_sphere(ax5, s1, s2, s3, S0=S0, step=(1,1),
                            c_by='s3', cmap='RdBu', clim=(-1,1),
                            s=8, alpha=0.9, sphere_style='wire')
    plt.savefig("temp.svg", bbox_inches='tight', transparent=True)
    plt.show()

    # 6) 斯格明子密度/数（示例：左半平面），独立成图
    left_mask = (kx < 0)
    nsk = skyrmion_density(s1, s2, s3)   # 你已有的工具函数
    nsk_left = nsk * left_mask

    fig6, ax6 = plt.subplots(figsize=(6,5))
    im = ax6.imshow(nsk_left.T, origin='lower', extent=extent, cmap='viridis')
    plt.colorbar(im, ax=ax6, shrink=0.8, label='Skyrmion density')
    ax6.set_title('Skyrmion density (left half-plane)')
    ax6.set_xlabel(r'$k_x$'); ax6.set_ylabel(r'$k_y$')
    plt.savefig("temp.svg", bbox_inches='tight', transparent=True)
    plt.show()

    print("Sum of n_sk (left):", float(nsk_left.sum()))
    # 这里的格点间距在 skyrmion_number 中由你自定义的函数内部处理；
    # 若需要明确步长，可在你的 skyrmion_number API 中传入 dkx,dky。
    # 例如：sk_num = skyrmion_number(nsk_left, dkx, dky)
    # 你当前示例用 (1,1)：保持一致
    sk_num = skyrmion_number(nsk_left, 1, 1)
    print("Skyrmion number (left):", sk_num)

    plt.show()