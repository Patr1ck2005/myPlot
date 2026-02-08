"""
visualize_Q.py

Visualize the condition arg(d_p/d_s) = +/- pi/2  <=>  Q(Delta) = (-r_p +/- t_p)/(r_s +/- t_s) is on negative real axis.

Dependencies:
    pip install numpy scipy matplotlib

Usage:
    python visualize_Q.py
    # or run cells in a Jupyter notebook

Notes:
    - The code uses the model:
        Q(Delta) = (N_bg + C / (1j*Delta + gamma)) / S
      where N_bg = N_r + 1j*N_i, C = C_r + 1j*C_i, S = S_r + 1j*S_i.
    - The analytic quadratic for Im(Q)=0 is:
        A2 * Delta^2 + A1 * Delta + A0 = 0
      with A2 = N_i*S_r - N_r*S_i,
           A1 = -(C_r*S_r + C_i*S_i),
           A0 = gamma^2*(N_i*S_r - N_r*S_i) + gamma*(C_i*S_r - C_r*S_i).
    - The condition for arg(d_p/d_s)=+/-pi/2 (mod pi) is Q in negative real axis,
      i.e., Im(Q)=0 and Re(Q)<0.
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from math import pi

# -----------------------------
# Model functions
# -----------------------------
def Q_of_Delta(Delta, N_bg, C, S, gamma):
    """
    Compute Q(Delta) = (N_bg + C/(i Delta + gamma)) / S
    Delta: scalar or numpy array (omega - omega0)
    N_bg, C, S: complex numbers
    gamma: positive real
    """
    denom = 1j * Delta + gamma
    return (N_bg + C / denom) / S

def analytic_roots(A2, A1, A0):
    """
    Solve A2 x^2 + A1 x + A0 = 0 for real roots.
    Return real roots (list).
    """
    if abs(A2) < 1e-16:
        # linear case A1 * x + A0 = 0
        if abs(A1) < 1e-16:
            return []
        return [-A0 / A1]
    disc = A1 * A1 - 4 * A2 * A0
    if disc < 0:
        return []
    r1 = (-A1 + np.sqrt(disc)) / (2 * A2)
    r2 = (-A1 - np.sqrt(disc)) / (2 * A2)
    return [r1, r2]

def compute_A_coeffs(Nr, Ni, Cr, Ci, Sr, Si, gamma):
    A2 = Ni * Sr - Nr * Si
    A1 = -(Cr * Sr + Ci * Si)
    A0 = gamma**2 * (Ni * Sr - Nr * Si) + gamma * (Ci * Sr - Cr * Si)
    return A2, A1, A0

def find_numeric_roots(Delta_range, N_bg, C, S, gamma, npt=2001):
    """
    Find approximate Delta where Im(Q)=0 using brentq between sign changes.
    Return list of roots refined by brentq.
    """
    Dmin, Dmax = Delta_range
    Delta_grid = np.linspace(Dmin, Dmax, npt)
    ImQ = np.imag(Q_of_Delta(Delta_grid, N_bg, C, S, gamma))
    roots = []
    for i in range(len(Delta_grid)-1):
        if ImQ[i] == 0:
            roots.append(Delta_grid[i])
        elif ImQ[i] * ImQ[i+1] < 0:
            a, b = Delta_grid[i], Delta_grid[i+1]
            try:
                root = optimize.brentq(lambda d: np.imag(Q_of_Delta(d, N_bg, C, S, gamma)), a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
                roots.append(root)
            except Exception:
                pass
    return sorted(list(set([float(r) for r in roots])))

# -----------------------------
# Visualization helpers
# -----------------------------
def plot_arg_vs_delta(Delta_range, N_bg, C, S, gamma, title_suffix=''):
    Dmin, Dmax = Delta_range
    Delta = np.linspace(Dmin, Dmax, 4001)
    Qvals = Q_of_Delta(Delta, N_bg, C, S, gamma)
    argQ = np.angle(Qvals)  # -pi..pi
    # convert to principal branch
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Delta, argQ)
    ax.set_xlabel('Delta (omega - omega0)')
    ax.set_ylabel('arg(Q) [rad]')
    ax.set_title('Phase of Q vs Delta' + title_suffix)
    # analytic roots and numeric roots
    Nr, Ni = N_bg.real, N_bg.imag
    Cr, Ci = C.real, C.imag
    Sr, Si = S.real, S.imag
    A2, A1, A0 = compute_A_coeffs(Nr, Ni, Cr, Ci, Sr, Si, gamma)
    roots_analytic = analytic_roots(A2, A1, A0)
    roots_numeric = find_numeric_roots(Delta_range, N_bg, C, S, gamma)
    # mark candidates and shade where Re(Q)<0
    for r in roots_numeric:
        q = Q_of_Delta(r, N_bg, C, S, gamma)
        if np.isreal(q) and q.real < 0:
            ax.axvline(r, linestyle='--')
            ax.plot(r, np.angle(q), marker='o')
    ax.grid(True)
    plt.show()
    print("analytic roots (Delta):", roots_analytic)
    print("numeric roots (Delta):", roots_numeric)
    # print whether they satisfy Re(Q)<0
    for r in roots_numeric:
        q = Q_of_Delta(r, N_bg, C, S, gamma)
        print(f"Delta={r:.6g}: Q={q:.6g}, Im(Q)={q.imag:.3e}, Re(Q)={q.real:.6g}, Re(Q)<0? {q.real<0}")

def plot_complex_trajectory(Delta_range, N_bg, C, S, gamma, title_suffix=''):
    Dmin, Dmax = Delta_range
    Delta = np.linspace(Dmin, Dmax, 2001)
    Qvals = Q_of_Delta(Delta, N_bg, C, S, gamma)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Qvals.real, Qvals.imag)
    ax.set_xlabel('Re(Q)')
    ax.set_ylabel('Im(Q)')
    ax.set_title('Complex-plane trajectory of Q(Delta)' + title_suffix)
    ax.axhline(0, linestyle=':')  # real axis
    ax.axvline(0, linestyle=':')  # imag axis
    # mark where Im(Q)=0 and Re(Q)<0
    roots = find_numeric_roots(Delta_range, N_bg, C, S, gamma)
    for r in roots:
        q = Q_of_Delta(r, N_bg, C, S, gamma)
        ax.plot(q.real, q.imag, marker='o')
        ax.annotate(f"{r:.3g}", (q.real, q.imag))
    ax.grid(True)
    plt.show()

def plot_param_sweep_Cr_vs_Delta(Cr_vals, Delta_range, N_bg_phase, Ci, S, gamma):
    """
    Example of 2D sweep: vary C_r (real part of C) vs Delta, show arg(Q).
    N_bg_phase is complex N_bg but with Nr replaced by the provided Nr (kept).
    We'll create a pcolormesh with argument of Q (wrapped to [-pi,pi]).
    """
    Delta = np.linspace(Delta_range[0], Delta_range[1], 501)
    Cr_grid = np.array(Cr_vals)
    Z = np.zeros((len(Cr_grid), len(Delta)))
    for i, Cr in enumerate(Cr_grid):
        C = Cr + 1j * Ci
        for j, d in enumerate(Delta):
            q = Q_of_Delta(d, N_bg_phase, C, S, gamma)
            Z[i, j] = np.angle(q)
    X, Y = np.meshgrid(Delta, Cr_grid)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    pcm = ax.pcolormesh(X, Y, Z, shading='auto')
    # # contour where Z=pi
    # CS = ax.contour(X, Y, Z, levels=[pi], colors=['r'], linewidths=1)
    ax.set_xlabel('Delta')
    ax.set_ylabel('C_r')
    ax.set_title('arg(Q) over Delta and C_r')
    fig.colorbar(pcm, ax=ax, label='arg(Q) [rad]')
    # overlay Im(Q)=0 contour
    # compute Im(Q) grid
    ImZ = np.zeros_like(Z)
    for i, Cr in enumerate(Cr_grid):
        C = Cr + 1j*Ci
        for j, d in enumerate(Delta):
            ImZ[i,j] = np.imag(Q_of_Delta(d, N_bg_phase, C, S, gamma))
    # contour where ImZ=0
    CS = ax.contour(X, Y, ImZ, levels=[0], linewidths=1, colors='gray')
    ax.clabel(CS, inline=1, fontsize=8)
    plt.show()

# -----------------------------
# Example usage (defaults)
# -----------------------------
if __name__ == "__main__":
    # SIGN convention: set sign = +1 for ( -r_p + t_p ) / ( r_s + t_s )
    # or sign = -1 for ( -r_p - t_p ) / ( r_s - t_s ) etc.
    sign = +1  # change to -1 to try the other d^± channel

    # Example background parameters (you should replace these with fitted TCMT results)
    # Interpretation (example):
    #   S = r_s +/- t_s  (background in denominator)       -> choose S_r, S_i
    #   N_bg = -r_p +/- t_p (background numerator)         -> choose N_r, N_i
    #   C = coupling amplitude of the low-Q p-mode to numerator
    #   gamma = linewidth (half-width) of the low-Q mode
    #
    # These sample numbers are purely illustrative.
    Sr = 1.0   # real part of S
    Si = 0.0   # imag part of S (set small if lossless background)
    Nr = -0.2  # real part of N_bg
    Ni = 0.0   # imag part of N_bg
    Cr = -0.5  # real part of C (coupling)
    Ci = 0.1   # imag part of C
    gamma = 0.03

    # assemble complex parameters
    N_bg = complex(Nr, Ni)
    C = complex(Cr, Ci)
    S = complex(Sr, Si)

    # choose Delta scanning range
    Delta_range = (-0.5, 0.5)

    # 1) Phase vs Delta, marking candidates
    plot_arg_vs_delta(Delta_range, N_bg, C, S, gamma, title_suffix=f' (sign={sign})')

    # 2) Complex Q trajectory
    plot_complex_trajectory(Delta_range, N_bg, C, S, gamma, title_suffix=f' (sign={sign})')

    # 3) Parameter sweep example: vary Cr
    Cr_vals = np.linspace(-1.2, 0.2, 201)
    plot_param_sweep_Cr_vs_Delta(Cr_vals, Delta_range, N_bg, Ci, S, gamma)

    print("Done. Edit the top 'Example usage' block with your fitted TCMT parameters (N_bg, C, S, gamma),")
    print("or change 'sign' to explore the other ± channel corresponding to d^±.")
