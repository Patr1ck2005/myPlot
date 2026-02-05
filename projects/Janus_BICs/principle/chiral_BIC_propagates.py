import numpy as np
import matplotlib.pyplot as plt

from projects.Janus_BICs.principle.fields import *


# ==========================================================
# Fourier helpers (k → x)
# ==========================================================

def jones_k_to_x(Ex_k, Ey_k):
    Ex = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Ex_k)))
    Ey = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Ey_k)))
    return Ex, Ey


# ==========================================================
# Visualization helpers
# ==========================================================

def complex_to_rgb(E):
    """
    Map complex field to RGB using HSV:
    phase → hue, amplitude → value
    """
    amp = np.abs(E)
    phase = np.angle(E)

    hue = (phase + np.pi) / (2 * np.pi)
    val = amp / (amp.max() + 1e-12)
    sat = np.ones_like(val)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = plt.cm.hsv(hue)[..., :3] * val[..., None]
    return rgb


def plot_stokes_maps(S0, S1, S2, S3, extent):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for ax, S, title in zip(
        axs.ravel(),
        [S0, S1, S2, S3],
        ["S0", "S1", "S2", "S3"],
    ):
        im = ax.imshow(S, origin="lower", extent=extent)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Stokes parameters")
    plt.tight_layout()


def plot_complex_rgb(Ex, Ey, extent):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(complex_to_rgb(Ex), origin="lower", extent=extent)
    axs[0].set_title("Ex complex (RGB phase map)")
    axs[1].imshow(complex_to_rgb(Ey), origin="lower", extent=extent)
    axs[1].set_title("Ey complex (RGB phase map)")
    plt.tight_layout()


def plot_polarization_ellipses(X, Y, S0, S1, S2, S3, step=5, scale=1):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    t = np.linspace(0, 2 * np.pi, 60)

    S0n = S0 / (S0.max() + 1e-12)

    for i in range(0, X.shape[0], step):
        for j in range(0, X.shape[1], step):
            s1, s2, s3 = S1[i, j], S2[i, j], S3[i, j]

            psi = 0.5 * np.arctan2(s2, s1)
            chi = 0.5 * np.arcsin(np.clip(s3, -1, 1))

            a = scale * np.sqrt(S0n[i, j])
            b = a * np.abs(np.tan(chi))

            x = a * np.cos(t)
            y = b * np.sin(t)

            xr = x * np.cos(psi) - y * np.sin(psi)
            yr = x * np.sin(psi) + y * np.cos(psi)

            color = plt.cm.coolwarm((s3 + 1) / 2)
            ax.plot(X[i, j] + xr, Y[i, j] + yr, color=color, linewidth=0.8)

    ax.set_title("Polarization ellipses (size ~ S0, color ~ S3)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()


# ==========================================================
# Main demo
# ==========================================================

if __name__ == "__main__":

    # ---------------- grid in k-space ----------------
    N = 513
    k0 = 2 * np.pi
    kmax = 2 * k0

    k = np.linspace(-kmax, kmax, N)
    dk = k[1] - k[0]
    KX, KY = np.meshgrid(k, k)
    extent_k = [k.min(), k.max(), k.min(), k.max()]

    # ---------------- k-space Poincare beam ----------------
    # Ex_k, Ey_k = poincare_beam(KX, KY, m=1)
    # Ex_k, Ey_k = jones_C_pair(KX, KY, d=0.00)
    Ex_k, Ey_k = jones_C_pair(KX, KY, dx=0.1, dy=0.0)
    # Ex_k, Ey_k = radial_polarization(KX, KY)

    # ---------------- Gaussian envelope (0.3 k0) ----------------
    sigma_k = 0.2 * k0
    Gk = np.exp(-(KX**2 + KY**2) / (2 * sigma_k**2))

    Ex_k *= Gk
    Ey_k *= Gk

    # 乘传播核
    H = propagation_kernel(KX, KY, 0)

    Ex_k *= H
    Ey_k *= H

    S0_k, S1_k, S2_k, S3_k = stokes_from_jones(Ex_k, Ey_k)

    # ---------------- transform to real space (z = 0) ----------------
    Ex, Ey = jones_k_to_x(Ex_k, Ey_k)

    # real-space coordinates corresponding to FFT grid
    L = 2 * np.pi / dk
    x = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    extent = [x.min(), x.max(), x.min(), x.max()]

    # ---------------- Stokes ----------------
    S0, S1, S2, S3 = stokes_from_jones(Ex, Ey)

    # ---------------- plots ----------------
    plot_stokes_maps(S0_k, S1_k, S2_k, S3_k, extent_k)
    plot_complex_rgb(Ex_k, Ey_k, extent_k)
    plot_polarization_ellipses(KX, KY, S0_k, S1_k, S2_k, S3_k, scale=0.2)

    # plot_stokes_maps(S0, S1, S2, S3, extent)
    # plot_complex_rgb(Ex, Ey, extent)
    # plot_polarization_ellipses(X, Y, S0, S1, S2, S3, scale=2)

    plt.show()
