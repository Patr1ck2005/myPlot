import numpy as np


def radial_polarization(X, Y):
    """Radial polarization field."""
    phi = np.arctan2(Y, X)
    Ex = np.cos(phi).astype(np.complex128)
    Ey = np.sin(phi).astype(np.complex128)
    return Ex, Ey


def azimuthal_polarization(X, Y):
    """Azimuthal polarization field."""
    phi = np.arctan2(Y, X)
    Ex = -np.sin(phi).astype(np.complex128)
    Ey = np.cos(phi).astype(np.complex128)
    return Ex, Ey


def hybrid_spin_orbit_field(X, Y, charge=2):
    """Hybrid polarization with spatially varying ellipticity."""
    phi = np.arctan2(Y, X)
    Ex = np.cos(phi)
    Ey = np.exp(1j * charge * phi) * np.sin(phi)
    return Ex, Ey


def poincare_beam(X, Y, m=1):
    """Canonical Poincaré beam."""
    phi = np.arctan2(Y, X)
    Ex = np.cos(m * phi)
    Ey = 1j * np.sin(m * phi)
    return Ex, Ey


def poincare_beam_oam(X, Y, l1=1, l2=-1, theta=np.pi / 2):
    phi = np.arctan2(Y, X)

    Ex = np.cos(theta / 2) * np.exp(1j * l1 * phi)
    Ey = np.sin(theta / 2) * np.exp(1j * l2 * phi)

    return Ex, Ey


def stokes_skyrmion(X, Y, R=1.0):
    r = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)

    theta = np.pi * r / R
    theta = np.clip(theta, 0, np.pi)

    s0 = np.ones_like(X)
    s1 = np.sin(theta) * np.cos(phi)
    s2 = np.sin(theta) * np.sin(phi)
    s3 = np.cos(theta)
    return s0, s1, s2, s3

def stokes_C_pair(X, Y, d=0.1):
    r2 = X ** 2 + Y ** 2 + 1e-12

    S1 = (X ** 2 - Y ** 2 - d ** 2) / (r2 + d ** 2)
    S2 = (2 * X * Y) / (r2 + d ** 2)
    S3 = (2 * d * X) / (r2 + d ** 2)

    # S1 = X**2 - Y**2 - d**2
    # S2 = 2 * X * Y
    # S3 = 2 * d * X
    S0 = np.sqrt(S1**2 + S2**2 + S3**2) + 1e-13
    S1 /= S0
    S2 /= S0
    S3 /= S0
    S0 = np.ones_like(S0)   # 完全偏振
    return S0, S1, S2, S3

def stokes_single_C(X, Y, handedness=+1):
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    # 从极点过渡到赤道
    theta = np.pi/2 * np.tanh(r)

    s0 = np.ones_like(X)
    s1 = np.sin(theta) * np.cos(phi)
    s2 = np.sin(theta) * np.sin(phi)
    s3 = handedness * np.cos(theta)
    return s0, s1, s2, s3

# def jones_C_pair(X, Y, d=0.6):
#     S0, S1, S2, S3 = stokes_C_pair(X, Y, d=d)
#     Ex, Ey = jones_from_stokes(S1, S2, S3, S0)
#     return Ex, Ey

def jones_C_pair(X, Y, dx=0.0, dy=0.0):
    """V 点（q=1）沿动量空间分裂成两 C 点（q=1/2）的最简物理基底。"""
    # EL = (X - kxc) + 1j*(Y - kyc)/np.sqrt(X**2+Y**2+1e-12)  # q=-1
    # ER = (X + kxc) - 1j*(Y - kyc)/np.sqrt(X**2+Y**2+1e-12)  # q=-1
    # ER = ((X - dx) + 1j*(Y - dy))/np.sqrt(X**2+Y**2+1e-12)  # q=1
    # EL = ((X + dx) - 1j*(Y - dy))/np.sqrt(X**2+Y**2+1e-12)  # q=1
    ER = ((X - dx) + 1j*(Y - dy))  # q=1
    EL = ((X + dx) - 1j*(Y - dy))  # q=1
    Ex = (EL + ER)/np.sqrt(2.0)
    Ey = 1j*(EL - ER)/np.sqrt(2.0)
    return Ex, Ey

# def jones_test(X, Y):
#     epsilon = 0.1
#     # TM in grating
#     Ey = X+(0.0)*X**2-(0)*Y**2-(epsilon+epsilon*1j)
#     Ex = (-2*Y+0.0*X*Y)*1j+2*epsilon*Y
#     return Ex, Ey

def jones_test(X, Y):
    epsilon = 0.0
    # TM in grating merging
    kx0 = 0.7
    Ey = X*(X**2-kx0**2)+(1)*X*Y**2-(epsilon+epsilon*1j)
    Ex = (-2*Y+0.0*(X-kx0)*Y)*1j+0*epsilon*Y
    # # TE in grating
    # Ex = X+0.1*epsilon-100*epsilon*1j*Y**2
    # Ey = 2*Y*1j
    return Ex, Ey

# def jones_C_pair(X, Y, dx=0.0, dy=0.0):
#     phi_R = np.arctan2(Y - dy, X - dx)
#     phi_L = np.arctan2(Y - dy, X + dx)
#
#     ER = np.exp(1j * phi_R)   # 只有相位
#     EL = np.exp(-1j * phi_L)
#
#     Ex = (EL + ER)/np.sqrt(2.0)
#     Ey = 1j*(EL - ER)/np.sqrt(2.0)
#
#     return Ex, Ey



def stokes_from_jones(Ex, Ey):
    """Compute normalized Stokes parameters from Jones field."""
    S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    S1 = (np.abs(Ex) ** 2 - np.abs(Ey) ** 2) / S0
    S2 = 2 * np.real(Ex * np.conj(Ey)) / S0
    S3 = -2 * np.imag(Ex * np.conj(Ey)) / S0
    return S0, S1, S2, S3

def jones_from_stokes(S1, S2, S3, S0=None):
    """
    Convert normalized Stokes parameters to Jones field.
    Assume fully polarized.
    """
    if S0 is None:
        S0 = np.ones_like(S1)

    # 椭偏参数
    psi = 0.5 * np.arctan2(S2, S1)
    chi = 0.5 * np.arcsin(np.clip(S3, -1, 1))

    a = np.cos(chi)
    b = np.sin(chi)

    Ex = np.sqrt(S0) * (a * np.cos(psi) - 1j * b * np.sin(psi))
    Ey = np.sqrt(S0) * (a * np.sin(psi) + 1j * b * np.cos(psi))

    return Ex, Ey


def propagation_kernel(kx, ky, z):
    k = 2 * np.pi
    kz = np.sqrt(np.clip(k**2 - kx**2 - ky**2, 0, None))
    return np.exp(1j * kz * z)