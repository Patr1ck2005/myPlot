"""
===========================================================
Poincaré Sphere Topology Demo for Polarization Fields
-----------------------------------------------------------
1. Generate multiple典型偏振场 (polarization ellipse + colormap)
2. Extract closed loop in real space
3. Map polarization evolution to Poincaré sphere
4. Compute:
   - Oriented solid angle
   - Berry phase
   - Winding numbers in different projections
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from projects.Janus_BICs.principle.fields import *


# ==========================================================
# Gaussian envelope modulation
# ==========================================================

def apply_gaussian_envelope_stokes(
    X, Y,
    S0, S1, S2, S3,
    w0=0.5,
    center=(0, 0),
    power=1.0
):
    """
    Apply Gaussian envelope to a Stokes field.

    Output remains Stokes parameters.
    """

    x0, y0 = center
    R2 = (X - x0)**2 + (Y - y0)**2

    envelope = np.exp(-power * R2 / w0**2)

    # ---- convert to Jones ----
    Ex, Ey = jones_from_stokes(S1, S2, S3, S0)

    Ex = Ex * envelope
    Ey = Ey * envelope

    # ---- back to Stokes ----
    return stokes_from_jones(Ex, Ey)

# ==========================================================
# Jones Fourier transform modules
# ==========================================================

def jones_k_to_x(Ex_k, Ey_k, shift=True):
    """
    k-space → real space
    """

    if shift:
        Ex_k = np.fft.ifftshift(Ex_k)
        Ey_k = np.fft.ifftshift(Ey_k)

    Ex = np.fft.ifft2(Ex_k)
    Ey = np.fft.ifft2(Ey_k)

    if shift:
        Ex = np.fft.fftshift(Ex)
        Ey = np.fft.fftshift(Ey)

    return Ex, Ey


def jones_x_to_k(Ex, Ey, shift=True):
    """
    real space → k-space
    """

    if shift:
        Ex = np.fft.ifftshift(Ex)
        Ey = np.fft.ifftshift(Ey)

    Ex_k = np.fft.fft2(Ex)
    Ey_k = np.fft.fft2(Ey)

    if shift:
        Ex_k = np.fft.fftshift(Ex_k)
        Ey_k = np.fft.fftshift(Ey_k)

    return Ex_k, Ey_k

# ==========================================================
# Stokes Fourier wrappers
# ==========================================================

def stokes_k_to_x(S0, S1, S2, S3):
    Ex_k, Ey_k = jones_from_stokes(S1, S2, S3, S0)
    Ex, Ey = jones_k_to_x(Ex_k, Ey_k)
    return stokes_from_jones(Ex, Ey)

def stokes_x_to_k(S0, S1, S2, S3):
    Ex, Ey = jones_from_stokes(S1, S2, S3, S0)
    Ex_k, Ey_k = jones_x_to_k(Ex, Ey)
    return stokes_from_jones(Ex_k, Ey_k)


def circular_loop(center=(0, 0), radius=0.7, n=400):
    """Closed circular path in real space."""
    t = np.linspace(0, 2 * np.pi, n)
    x = radius * np.cos(t) + center[0]
    y = radius * np.sin(t) + center[1]
    return x, y


def poincare_from_stokes(S1, S2, S3):
    return np.vstack([S1, S2, S3]).T


def solid_angle(theta, phi):
    """Oriented solid angle Ω = ∮ (1 - cosθ) dφ"""
    dphi = np.diff(phi)
    return np.sum((1 - np.cos(theta[:-1])) * dphi)


def berry_phase(theta, phi):
    return 0.5 * solid_angle(theta, phi)


def winding_number(x, y):
    angle = np.unwrap(np.arctan2(y, x))
    return (angle[-1] - angle[0]) / (2 * np.pi)


def draw_poincare_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 41)
    v = np.linspace(0, np.pi, 21)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.3, rstride=4, cstride=4)
    # 绘制灰色表面
    # ax.plot_surface(
    #     x, y, z, rstride=1, cstride=1, color='lightgray', alpha=0.15, linewidth=0, zorder=0
    # )


def plot_poincare_path(ax, S, step=10, cmap='rainbow'):
    # 在路径上绘制渐变色点
    cmap = plt.get_cmap(cmap)
    for i in range(len(S) - 1):
        ax.plot(
            S[i:i + 2, 0], S[i:i + 2, 1], S[i:i + 2, 2],
            color=cmap(i / len(S)),
            linewidth=2
        )
    # 隔step绘制渐变箭头
    for i in range(0, len(S) - step, step):
        ax.quiver(
            S[i, 0], S[i, 1], S[i, 2],
            S[i + step, 0] - S[i, 0],
            S[i + step, 1] - S[i, 1],
            S[i + step, 2] - S[i, 2],
            length=0.1, normalize=True,
            arrow_length_ratio=2,
            color=cmap(i / len(S))
        )


def plot_polarization_field(ax, X, Y, S1, S2, S3, S0, step=4, scale=0.04):
    t = np.linspace(0, 2 * np.pi, 60)
    for i in range(0, X.shape[0], step):
        for j in range(0, X.shape[1], step):
            s1, s2, s3 = S1[i, j], S2[i, j], S3[i, j]

            # Azimuth angle
            psi = 0.5 * np.arctan2(s2, s1)

            # Ellipticity angle
            chi = 0.5 * np.arcsin(np.clip(s3, -1, 1))

            # Semi-axes (normalized)
            a = scale
            b = scale * np.abs(np.tan(chi))

            # Parametric ellipse
            x = a * np.cos(t)
            y = b * np.sin(t)

            # Rotate ellipse
            xr = x * np.cos(psi) - y * np.sin(psi)
            yr = x * np.sin(psi) + y * np.cos(psi)

            ax.plot(
                X[i, j] + xr,
                Y[i, j] + yr,
                color=plt.cm.coolwarm((s3 + 1) / 2),
                linewidth=0.8
            )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def track_polarization_field(ax, xl, yl, S1, S2, S3, S0):
    # Poincaré coordinates
    S = poincare_from_stokes(S1, S2, S3)
    theta = np.arccos(S3l)
    phi = np.unwrap(np.arctan2(S2, S1))

    Omega = solid_angle(theta, phi)
    gamma = berry_phase(theta, phi)

    w12 = winding_number(S1, S2)
    w13 = winding_number(S1, S3)
    w23 = winding_number(S2, S3)
    draw_poincare_sphere(ax)
    plot_poincare_path(ax, S)
    # 去掉3D绘图的背景和背景网格线
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.view_init(elev=20, azim=30)
    S12 = S.copy()
    S12[:, 2] = -2
    plot_poincare_path(ax, S12, cmap='rainbow')
    S13 = S.copy()
    S13[:, 1] = -2
    plot_poincare_path(ax, S13, cmap='rainbow')
    S23 = S.copy()
    S23[:, 0] = -2
    plot_poincare_path(ax, S23, cmap='rainbow')
    ax.set_box_aspect([1, 1, 1])

    ax.set_title(
        f"Ω={Omega:.2f}, γ={gamma:.2f}\n"
        f"w12={w12:.1f}, w13={w13:.1f}, w23={w23:.1f}"
    )
    ax.set_box_aspect([1, 1, 1])
    return ax


if __name__ == "__main__":
    grid = np.linspace(-1, 1, 4 * 20 + 1, endpoint=True)
    X, Y = np.meshgrid(grid, grid)

    jones_field = jones_test
    # stokes_field = stokes_skyrmion

    Ex, Ey = jones_field(X, Y)
    S0, S1, S2, S3 = stokes_from_jones(Ex, Ey)

    # S0, S1, S2, S3 = stokes_field(X, Y)

    S0, S1, S2, S3 = apply_gaussian_envelope_stokes(
        X, Y,
        S0, S1, S2, S3,
        w0=0.5
    )
    #
    # # → Jones(k)
    # Ex, Ey = jones_from_stokes(S1, S2, S3, S0)
    # # → real space
    # Ex, Ey = jones_k_to_x(Ex, Ey)

    # → Stokes
    # S0, S1, S2, S3 = stokes_from_jones(Ex, Ey)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_title("Polarization Field (Jones Ellipses)")
    ax = plot_polarization_field(
        ax, X, Y, S1, S2, S3, S0,
    )
    xl, yl = circular_loop(center=(0.0, 0), radius=0.2)
    # 在ax上绘制loop, 渐变色
    cmap = plt.get_cmap("rainbow")
    for i in range(len(xl) - 1):
        ax.plot(
            xl[i:i + 2], yl[i:i + 2],
            color=cmap(i / len(xl)),
            linewidth=2
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(S2, origin="lower", extent=(-1, 1, -1, 1), cmap='coolwarm', vmin=-1, vmax=1)
    # 在ax上绘制loop, 渐变色
    cmap = plt.get_cmap("rainbow")
    for i in range(len(xl) - 1):
        ax.plot(
            xl[i:i + 2], yl[i:i + 2],
            color=cmap(i / len(xl)),
            linewidth=2
        )

    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(6, 6))
    bounds = [-1, -0.995, -0.99, -0.95, -0.5, -0.05, 0.05, 0.5, 0.95, 0.99, 0.995, 1]
    norm = mcolors.BoundaryNorm(bounds, ncolors=256)
    ax.imshow(S3, origin="lower", extent=(-1, 1, -1, 1), cmap='coolwarm', norm=norm)
    # 在ax上绘制loop, 渐变色
    cmap = plt.get_cmap("rainbow")
    for i in range(len(xl) - 1):
        ax.plot(
            xl[i:i + 2], yl[i:i + 2],
            color=cmap(i / len(xl)),
            linewidth=2
        )


    Exl, Eyl = jones_field(xl, yl)
    S0l, S1l, S2l, S3l = stokes_from_jones(Exl, Eyl)
    # S0l, S1l, S2l, S3l = stokes_C_pair(xl, yl)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    track_polarization_field(ax, xl, yl, S1l, S2l, S3l, S0l)
    plt.show()
