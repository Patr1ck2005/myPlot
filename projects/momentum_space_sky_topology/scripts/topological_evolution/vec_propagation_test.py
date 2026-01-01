import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


# ============================================================
# 辅助函数：Stokes 参数、Jones 矢量转换
# ============================================================
def stokes_from_jones(Ex, Ey, eps=1e-12):
    S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + eps
    S1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = 2 * np.imag(Ex * np.conj(Ey))
    phi = 0.5 * np.arctan2(S2, S1)
    chi = 0.5 * np.arcsin(np.clip(S3 / S0, -1.0, 1.0))
    return S0, S1, S2, S3, phi, chi


def stokes_normalized_from_jones(Ex, Ey):
    S0, S1, S2, S3, phi, chi = stokes_from_jones(Ex, Ey)
    s1 = S1 / S0
    s2 = S2 / S0
    s3 = S3 / S0
    return S0, s1, s2, s3, phi, chi


# ============================================================
# 动量空间拉盖尔-高斯模式 (LG01, l=1, p=0)
# ============================================================
def laguerre_gaussian_polar(kx, ky, l=1, p=0, w0=1.0, k0=1.0):
    r2 = kx ** 2 + ky ** 2
    r = np.sqrt(r2)
    phi = np.arctan2(ky, kx)
    rho = r / w0
    Lpl = 1  # 对于 p=0，L_p^|l| = 1
    gaussian = np.exp(-r2 / (2 * w0 ** 2))
    amplitude = (np.sqrt(2) * rho) ** np.abs(l) * Lpl * gaussian
    phase = np.exp(1j * l * phi)
    return amplitude * phase


# ============================================================
# 径向偏振矢量光束在动量空间的 Jones 矢量
# ============================================================
def radial_vector_beam_uv(u, v, w0=0.5):
    """
    径向偏振：E ∝ (x̂ cosφ + ŷ sinφ) * LG01
    在动量空间中，u = kx/k0, v = ky/k0
    """
    phi = np.arctan2(v, u)
    LG = laguerre_gaussian_polar(u, v, l=1, w0=w0)

    # 径向偏振矢量：(cosφ, sinφ)
    Ex = LG * np.cos(phi)
    Ey = LG * np.sin(phi)
    return Ex, Ey


# ============================================================
# 归一化角谱传播
# ============================================================
def transfer_normalized(u, v, zeta):
    w = np.sqrt(1.0 - (u ** 2 + v ** 2) + 0j)
    return np.exp(1j * w * zeta)


def propagate_kspace_normalized(Ex_uv_0, Ey_uv_0, u, v, zeta):
    H = transfer_normalized(u, v, zeta)
    return Ex_uv_0 * H, Ey_uv_0 * H


# ============================================================
# FFT 坐标转换：u -> x̃ = k0 x
# ============================================================
def ugrid_to_xtilde(u_axis):
    n = u_axis.size
    du = float(u_axis[1] - u_axis[0])
    dx_tilde = 2.0 * np.pi / (n * du)
    x_tilde = (np.arange(n) - n // 2) * dx_tilde
    return x_tilde, dx_tilde


def ifft2_centered(F):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F)))


def uv_to_xy_tilde(Ex_uv, Ey_uv):
    Ex = ifft2_centered(Ex_uv)
    Ey = ifft2_centered(Ey_uv)
    return Ex, Ey


# ============================================================
# 简单可视化函数（可替换为您的 advance_plot_styles）
# ============================================================
def imshow_phi(ax, phi, extent, cmap='hsv'):
    phi_norm = (phi + np.pi) % (2 * np.pi)  # 0到2π
    im = ax.imshow(phi_norm.T, origin='lower', extent=extent, cmap=cmap)
    plt.colorbar(im, ax=ax, shrink=0.8)


def imshow_s3(ax, s3, extent, cmap='RdBu'):
    im = ax.imshow(s3.T, origin='lower', extent=extent, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)


def hyper_color_map(s1, s2, s3):
    """简单版：Poincaré球超色图"""
    theta = np.arccos(s3)
    phi = np.arctan2(s2, s1)
    rgb = np.zeros((*s1.shape, 3))
    rgb[..., 0] = np.sin(theta) * np.cos(phi) * 0.5 + 0.5
    rgb[..., 1] = np.sin(theta) * np.sin(phi) * 0.5 + 0.5
    rgb[..., 2] = np.cos(theta) * 0.5 + 0.5
    return np.clip(rgb, 0, 1)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 动量空间网格
    n = 801
    umax = 5
    u1 = np.linspace(-umax, umax, n)
    v1 = np.linspace(-umax, umax, n)
    u, v = np.meshgrid(u1, v1, indexing='ij')
    extent_uv = [u1.min(), u1.max(), v1.min(), v1.max()]

    # 实空间网格
    x_tilde, dx_tilde = ugrid_to_xtilde(u1)
    y_tilde, dy_tilde = ugrid_to_xtilde(v1)
    X, Y = np.meshgrid(x_tilde, y_tilde, indexing='ij')
    extent_xy = [x_tilde.min(), x_tilde.max(), y_tilde.min(), y_tilde.max()]

    # 构建初始动量空间径向偏振 LG01 光束
    w0 = 0.1  # 控制束腰宽度
    Ex_uv_0, Ey_uv_0 = radial_vector_beam_uv(u, v, w0=w0)

    # imshow Ex.real
    plt.figure(figsize=(6, 5))
    plt.imshow(Ex_uv_0.real.T, origin='lower', extent=extent_uv, cmap='RdBu')
    plt.colorbar(label='Re(Ex)')
    plt.title('Initial Ex (u,v)')
    plt.xlabel('u = kx/k0')
    plt.ylabel('v = ky/k0')
    plt.show()

    # imshow Ey.real
    plt.figure(figsize=(6, 5))
    plt.imshow(Ey_uv_0.real.T, origin='lower', extent=extent_uv, cmap='RdBu')
    plt.colorbar(label='Re(Ex)')
    plt.title('Initial Ey (u,v)')
    plt.xlabel('u = kx/k0')
    plt.ylabel('v = ky/k0')
    plt.show()


    # 传播距离
    zeta_list = [0.0, 5.0, 10.0, 20.0]

    for zeta in zeta_list:
        print(f"\n传播距离 ζ = {zeta:.1f}")

        # 传播
        Ex_uv, Ey_uv = propagate_kspace_normalized(Ex_uv_0, Ey_uv_0, u, v, zeta)
        Ex_xy, Ey_xy = uv_to_xy_tilde(Ex_uv, Ey_uv)

        # Stokes 参数（归一化）
        _, s1_uv, s2_uv, s3_uv, phi_uv, _ = stokes_normalized_from_jones(Ex_uv, Ey_uv)
        _, s1_xy, s2_xy, s3_xy, phi_xy, _ = stokes_normalized_from_jones(Ex_xy, Ey_xy)

        # 图1：强度 + Hyper colormap
        fig1, axes = plt.subplots(2, 2, figsize=(12, 9))
        I_uv = np.abs(Ex_uv) ** 2 + np.abs(Ey_uv) ** 2
        I_xy = np.abs(Ex_xy) ** 2 + np.abs(Ey_xy) ** 2

        axes[0, 0].imshow(I_uv.T, origin='lower', extent=extent_uv, cmap='inferno')
        axes[0, 0].set_title(f'|E|^2 (u,v)  ζ={zeta:.1f}')
        axes[0, 1].imshow(hyper_color_map(s1_uv, s2_uv, s3_uv), origin='lower', extent=extent_uv)
        axes[0, 1].set_title('Hyper colormap (u,v)')
        axes[1, 0].imshow(I_xy.T, origin='lower', extent=extent_xy, cmap='inferno')
        axes[1, 0].set_title(f'|E|^2 (x̃,ỹ)')
        axes[1, 1].imshow(hyper_color_map(s1_xy, s2_xy, s3_xy), origin='lower', extent=extent_xy)
        axes[1, 1].set_title('Hyper colormap (x̃,ỹ)')

        for ax in axes.flat:
            ax.set_xlabel('u' if 'u' in ax.get_title() else 'x̃')
            ax.set_ylabel('v' if 'u' in ax.get_title() else 'ỹ')
        plt.tight_layout()
        plt.show()

        # 图2：phi 和 s3
        fig2, axes = plt.subplots(2, 2, figsize=(12, 9))
        imshow_phi(axes[0, 0], phi_uv, extent_uv)
        axes[0, 0].set_title(f'φ(u,v)  ζ={zeta:.1f}')
        imshow_s3(axes[0, 1], s3_uv, extent_uv)
        axes[0, 1].set_title(f's₃(u,v)')
        imshow_phi(axes[1, 0], phi_xy, extent_xy)
        axes[1, 0].set_title(f'φ(x̃,ỹ)')
        imshow_s3(axes[1, 1], s3_xy, extent_xy)
        axes[1, 1].set_title(f's₃(x̃,ỹ)')
        plt.tight_layout()
        plt.show()


        # 图3：Skyrmion density
        def skyrmion_density(s1, s2, s3):
            # 简单近似：∇×s · s
            ds1_dx = np.gradient(s1, axis=0)
            ds1_dy = np.gradient(s1, axis=1)
            ds2_dx = np.gradient(s2, axis=0)
            ds2_dy = np.gradient(s2, axis=1)
            ds3_dx = np.gradient(s3, axis=0)
            ds3_dy = np.gradient(s3, axis=1)
            nsk = (s1 * (ds2_dy - ds3_dy) - s2 * (ds1_dx - ds3_dx) + s3 * (ds1_dy - ds2_dx))
            return nsk


        nsk_xy = skyrmion_density(s1_xy, s2_xy, s3_xy)
        fig4 = plt.figure(figsize=(6, 5))
        plt.imshow(nsk_xy.T, origin='lower', extent=extent_xy, cmap='seismic', vmin=-5, vmax=5)
        plt.colorbar(label='n_sk')
        plt.title(f'Skyrmion density (x̃,ỹ)  ζ={zeta:.1f}')
        plt.xlabel('x̃ = k₀x');
        plt.ylabel('ỹ = k₀y')
        plt.show()

        # 计算总 skyrmion number
        area_element = dx_tilde * dy_tilde
        skyrmion_num = np.sum(nsk_xy) * area_element
        print(f"Skyrmion number (实空间积分) ≈ {skyrmion_num:.3f}")

    plt.show()
