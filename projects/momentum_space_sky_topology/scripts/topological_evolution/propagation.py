import numpy as np
from matplotlib import pyplot as plt

from advance_plot_styles.polar_plot import *
from utils.advanced_color_mapping import map_s1s2s3_color
from utils.functions import skyrmion_density, skyrmion_number

# ============================================================
# 你原来的 Jones/Stokes 相关函数：原样保留（或直接 import）
# ============================================================

def envelope_gaussian(kx, ky, kw=1.0, m=1, eps=0):
    r2 = kx*kx + ky*ky
    # exp(-(r/kw)^(2m)) = exp(-(r2/kw^2)^m)
    return np.exp(-np.power(r2/(kw*kw), m))+eps

def jones_field_base(kx, ky, kxc=0.0, kyc=0.0, amp=1.0, eps=0):
    EL = (kx - kxc) + 1j*(ky - kyc)
    ER = (kx + kxc) - 1j*(ky - kyc)
    Ex = amp*(EL + ER)/np.sqrt(2.0)
    Ey = amp*1j*(EL - ER)/np.sqrt(2.0)
    return Ex+eps, Ey+eps

def stokes_from_jones(Ex, Ey, eps=1e-12):
    S0 = (np.abs(Ex)**2 + np.abs(Ey)**2) + eps
    S1 = (np.abs(Ex)**2 - np.abs(Ey)**2)
    S2 = 2.0*np.real(Ex*np.conj(Ey))
    S3 = 2.0*np.imag(Ex*np.conj(Ey))
    phi = 0.5*np.arctan2(S2, S1)
    chi = 0.5*np.arcsin(np.clip(S3/S0, -1.0, 1.0))
    return S0, S1, S2, S3, phi, chi

def jones_field(kx, ky, kxc=0.0, kyc=0.0, amp=1.0, p=1.0):
    Ex0, Ey0 = jones_field_base(kx, ky, kxc=kxc, kyc=kyc, amp=amp)
    return Ex0, Ey0


# ============================================================
# 完全归一化角谱传播：u,v,ζ
# ============================================================

def transfer_normalized(u, v, zeta):
    """
    u=kx/k0, v=ky/k0, zeta=k0*z
    w=kz/k0=sqrt(1-u^2-v^2)
    H=exp(i*w*zeta)
    """
    w = np.sqrt((1.0 - (u*u + v*v)) + 0j)
    return np.exp(1j * w * zeta)

def propagate_kspace_normalized(Ex_uv_0, Ey_uv_0, u, v, zeta):
    H = transfer_normalized(u, v, zeta)
    return Ex_uv_0 * H, Ey_uv_0 * H


# ============================================================
# FFT 坐标：u -> x̃=k0 x
# ============================================================

def ugrid_to_xtilde(u_axis):
    n = u_axis.size
    du = float(u_axis[1] - u_axis[0])
    dx_tilde = 2.0*np.pi / (n * du)
    x_tilde = (np.arange(n) - n//2) * dx_tilde
    return x_tilde, dx_tilde

def ifft2_centered(F):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F)))

def uv_to_xy_tilde(Ex_uv, Ey_uv):
    Ex = ifft2_centered(Ex_uv)
    Ey = ifft2_centered(Ey_uv)
    return Ex, Ey


# ============================================================
# 通用：从 Jones 得到 (S0,s1,s2,s3,phi)
# ============================================================

def stokes_normalized_from_jones(Ex, Ey):
    S0, S1, S2, S3, phi, chi = stokes_from_jones(Ex, Ey)
    s1 = S1 / S0
    s2 = S2 / S0
    s3 = S3 / S0
    return S0, s1, s2, s3, phi, chi


# ============================================================
# 主程序：每个 ζ 同时画 动量空间 + 实空间
# ============================================================

if __name__ == "__main__":

    # ---------- 1) 动量空间网格：u,v ----------
    n = 201*3
    umax = 2.0
    u1 = np.linspace(-umax, umax, n)
    v1 = np.linspace(-umax, umax, n)
    u, v = np.meshgrid(u1, v1, indexing="ij")
    extent_uv = [u1.min(), u1.max(), v1.min(), v1.max()]

    # C 点位置（归一化波矢坐标）
    # uc, vc = 0, 0
    uc, vc = 0.04, 0.01

    # ---------- 2) 初始 Jones：在 (u,v) ----------
    Ex_uv_0, Ey_uv_0 = jones_field(u, v, kxc=uc, kyc=vc, amp=1.0, p=1.0)
    # ues envelope_gaussian
    env = envelope_gaussian(u, v, kw=0.2, m=1)
    Ex_uv_0 *= env
    Ey_uv_0 *= env

    fig, ax = plt.subplots(figsize=(5,4))
    # imshow Ex.real
    im = ax.imshow(np.real(Ex_uv_0).T, origin="lower", extent=extent_uv, cmap="viridis")
    ax.set_title(r"Initial $E_x(u,v)$ Real Part")
    ax.set_xlabel("u"); ax.set_ylabel("v")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.show()
    fig, ax = plt.subplots(figsize=(5,4))
    # imshow Ey.real
    im = ax.imshow(np.real(Ey_uv_0).T, origin="lower", extent=extent_uv, cmap="viridis")
    ax.set_title(r"Initial $E_x(u,v)$ Real Part")
    ax.set_xlabel("u"); ax.set_ylabel("v")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.show()



    # ---------- 3) 实空间坐标：x̃,ỹ ----------
    x_tilde, dx_tilde = ugrid_to_xtilde(u1)
    y_tilde, dy_tilde = ugrid_to_xtilde(v1)
    X_t, Y_t = np.meshgrid(x_tilde, y_tilde, indexing="ij")
    extent_xy = [x_tilde.min(), x_tilde.max(), y_tilde.min(), y_tilde.max()]

    # ---------- 4) 传播距离：ζ ----------
    zeta_list = [0.0, 5.0, 10.0, 20.0]

    for zeta in zeta_list:

        # ===== A) 传播后的动量空间 Jones =====
        Ex_uv, Ey_uv = propagate_kspace_normalized(Ex_uv_0, Ey_uv_0, u, v, zeta=zeta)
        S0_uv, s1_uv, s2_uv, s3_uv, phi_uv, chi_uv = stokes_normalized_from_jones(Ex_uv, Ey_uv)

        # ===== B) 对应的实空间 Jones =====
        Ex_xy, Ey_xy = uv_to_xy_tilde(Ex_uv, Ey_uv)
        S0_xy, s1_xy, s2_xy, s3_xy, phi_xy, chi_xy = stokes_normalized_from_jones(Ex_xy, Ey_xy)

        # =====================================================
        # Figure 1：动量空间 vs 实空间 的强度 + Hyper colormap（2x2）
        # =====================================================
        fig1, axes = plt.subplots(2, 2, figsize=(9, 7))

        # (u,v) 强度
        I_uv = (np.abs(Ex_uv)**2 + np.abs(Ey_uv)**2)
        im00 = axes[0, 0].imshow(I_uv.T, origin="lower", extent=extent_uv, cmap="viridis")
        axes[0, 0].plot([+uc, -uc], [vc, vc], "k.", ms=6)
        axes[0, 0].set_title(rf"$|E(u,v)|^2$  at $\zeta={zeta:g}$")
        axes[0, 0].set_xlabel("u"); axes[0, 0].set_ylabel("v")
        plt.colorbar(im00, ax=axes[0, 0], shrink=0.8)

        # (u,v) hyper
        rgb_uv = map_s1s2s3_color(s1_uv, s2_uv, s3_uv)
        axes[0, 1].imshow(rgb_uv, origin="lower", extent=extent_uv)
        axes[0, 1].plot([+uc, -uc], [vc, vc], "k.", ms=6)
        axes[0, 1].set_title("Hyper colormap in (u,v)")
        axes[0, 1].set_xlabel("u"); axes[0, 1].set_ylabel("v")

        # (x̃,ỹ) 强度
        I_xy = (np.abs(Ex_xy)**2 + np.abs(Ey_xy)**2)
        im10 = axes[1, 0].imshow(I_xy.T, origin="lower", extent=extent_xy, cmap="viridis")
        axes[1, 0].set_title(r"$|E(\tilde{x},\tilde{y})|^2$")
        axes[1, 0].set_xlabel(r"$\tilde{x}=k_0 x$")
        axes[1, 0].set_ylabel(r"$\tilde{y}=k_0 y$")
        plt.colorbar(im10, ax=axes[1, 0], shrink=0.8)

        # (x̃,ỹ) hyper
        rgb_xy = map_s1s2s3_color(s1_xy, s2_xy, s3_xy)
        axes[1, 1].imshow(rgb_xy, origin="lower", extent=extent_xy)
        axes[1, 1].set_title(r"Hyper colormap in $(\tilde{x},\tilde{y})$")
        axes[1, 1].set_xlabel(r"$\tilde{x}=k_0 x$")
        axes[1, 1].set_ylabel(r"$\tilde{y}=k_0 y$")

        plt.tight_layout()
        plt.show()

        # =====================================================
        # Figure 2：动量空间 vs 实空间 的 phi / s3（2x2）
        # 复用你已有的 imshow_phi / imshow_s3
        # =====================================================
        fig2, axes = plt.subplots(2, 2, figsize=(9, 7))

        # (u,v) phi
        imshow_phi(axes[0, 0], phi_uv, extent=extent_uv, interpolation="nearest", cmap="hsv")
        axes[0, 0].plot([+uc, -uc], [vc, vc], "k.", ms=6)
        axes[0, 0].set_title(r"$\phi(u,v)$")
        axes[0, 0].set_xlabel("u"); axes[0, 0].set_ylabel("v")

        # (u,v) s3
        imshow_s3(axes[0, 1], s3_uv, S0=S0_uv, extent=extent_uv, interpolation="nearest", cmap="RdBu")
        axes[0, 1].plot([+uc, -uc], [vc, vc], "k.", ms=6)
        axes[0, 1].set_title(r"$s_3(u,v)$")
        axes[0, 1].set_xlabel("u"); axes[0, 1].set_ylabel("v")

        # (x̃,ỹ) phi
        imshow_phi(axes[1, 0], phi_xy, extent=extent_xy, interpolation="nearest", cmap="hsv")
        axes[1, 0].set_title(r"$\phi(\tilde{x},\tilde{y})$")
        axes[1, 0].set_xlabel(r"$\tilde{x}=k_0 x$")
        axes[1, 0].set_ylabel(r"$\tilde{y}=k_0 y$")

        # (x̃,ỹ) s3
        imshow_s3(axes[1, 1], s3_xy, S0=S0_xy, extent=extent_xy, interpolation="nearest", cmap="RdBu")
        axes[1, 1].set_title(r"$s_3(\tilde{x},\tilde{y})$")
        axes[1, 1].set_xlabel(r"$\tilde{x}=k_0 x$")
        axes[1, 1].set_ylabel(r"$\tilde{y}=k_0 y$")

        plt.suptitle(rf"Stokes maps at $\zeta={zeta:g}$", y=1.02)
        plt.tight_layout()
        plt.show()

        # =====================================================
        # Figure 3：动量空间 vs 实空间 的偏振椭圆（并排 1x2）
        # =====================================================
        fig3, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        plot_polarization_ellipses(
            axes[0], u, v, s1_uv, s2_uv, s3_uv, S0=S0_uv,
            step=(10, 10), scale=0.05
        )
        axes[0].plot([+uc, -uc], [vc, vc], "k.", ms=6)
        axes[0].set_title(r"Ellipses in $(u,v)$")
        axes[0].set_xlim(extent_uv[0], extent_uv[1])
        axes[0].set_ylim(extent_uv[2], extent_uv[3])

        plot_polarization_ellipses(
            axes[1], X_t, Y_t, s1_xy, s2_xy, s3_xy, S0=S0_xy,
            step=(10, 10), scale=10
        )
        axes[1].set_title(r"Ellipses in $(\tilde{x},\tilde{y})$")
        axes[1].set_xlim(extent_xy[0], extent_xy[1])
        axes[1].set_ylim(extent_xy[2], extent_xy[3])

        plt.suptitle(rf"Polarization ellipses at $\zeta={zeta:g}$", y=1.02)
        plt.tight_layout()
        plt.show()

        # =====================================================
        # Figure 4：动量空间 vs 实空间 的 skyrmion density（并排 1x2）
        # =====================================================
        nsk_uv = skyrmion_density(s1_uv, s2_uv, s3_uv)
        nsk_xy = skyrmion_density(s1_xy, s2_xy, s3_xy)

        fig4, axes = plt.subplots(1, 2, figsize=(10, 4.2))

        imL = axes[0].imshow(nsk_uv.T, origin="lower", extent=extent_uv, cmap="viridis")
        axes[0].set_title(r"$n_{sk}(u,v)$")
        axes[0].set_xlabel("u"); axes[0].set_ylabel("v")
        plt.colorbar(imL, ax=axes[0], shrink=0.8)

        imR = axes[1].imshow(nsk_xy.T, origin="lower", extent=extent_xy, cmap="viridis")
        axes[1].set_title(r"$n_{sk}(\tilde{x},\tilde{y})$")
        axes[1].set_xlabel(r"$\tilde{x}$"); axes[1].set_ylabel(r"$\tilde{y}$")
        plt.colorbar(imR, ax=axes[1], shrink=0.8)

        plt.suptitle(rf"Skyrmion density at $\zeta={zeta:g}$", y=1.02)
        plt.tight_layout()
        plt.show()

        # （可选）给一个简单的数值输出：例如实空间左半平面 x̃<0 的 skyrmion number
        left_mask = (X_t < 0)
        nsk_left = nsk_xy * left_mask
        sk_num_left = skyrmion_number(nsk_left, 1, 1)
        print(rf"[zeta={zeta:g}] skyrmion_number(x̃<0) =", sk_num_left)

    plt.show()
