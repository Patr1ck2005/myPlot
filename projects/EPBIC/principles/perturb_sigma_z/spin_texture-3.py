import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
#  Three-mode model -> projected pseudospin texture
#  Author: ChatGPT
# ============================================================

# -----------------------------
# Global fixed parameters
# -----------------------------
omega = 1.0          # average plasmonic frequency
gamma = 0.12         # average plasmonic loss
S_grid = np.linspace(0.0, 1.0, 500)

# Pauli matrices acting in the 2D plasmonic subspace
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


# -----------------------------
# Model Hamiltonian
# -----------------------------
def build_H(S, omega, gamma, delta_w, delta_g, kappa, omega3):
    """
    3x3 non-Hermitian Hamiltonian

    delta_tilde = delta_w - i delta_g
    theta = 2 pi S
    """
    theta = 2 * np.pi * S
    delta_tilde = delta_w - 1j * delta_g

    H = np.array([
        [omega - 1j * gamma + delta_tilde, 0.0,                   kappa * np.cos(theta)],
        [0.0,                             omega - 1j * gamma - delta_tilde, kappa * np.sin(theta)],
        [kappa * np.cos(theta),           kappa * np.sin(theta),  omega3]
    ], dtype=complex)
    return H


# -----------------------------
# Left/right eigensystem
# -----------------------------
def biorthogonal_eigensystem(H):
    """
    Returns eigenvalues, right eigenvectors, matched left eigenvectors,
    all sorted by Re(eigenvalue).

    Right eigenvectors: H |R_n> = lambda_n |R_n>
    Left  eigenvectors: H^\dagger |L_n> = lambda_n^* |L_n>
    """
    evals_r, VR = np.linalg.eig(H)
    evals_l, VL = np.linalg.eig(H.conj().T)

    matched_L = np.zeros_like(VR, dtype=complex)

    for i, lam in enumerate(evals_r):
        j = np.argmin(np.abs(evals_l - np.conj(lam)))
        matched_L[:, i] = VL[:, j]

    # sort by real part of eigenvalues
    idx = np.argsort(evals_r.real)
    evals_r = evals_r[idx]
    VR = VR[:, idx]
    matched_L = matched_L[:, idx]

    # normalize vectors for numerical stability
    for n in range(3):
        if np.linalg.norm(VR[:, n]) > 0:
            VR[:, n] = VR[:, n] / np.linalg.norm(VR[:, n])
        if np.linalg.norm(matched_L[:, n]) > 0:
            matched_L[:, n] = matched_L[:, n] / np.linalg.norm(matched_L[:, n])

    return evals_r, VR, matched_L


# -----------------------------
# Projected pseudospin in plasmonic subspace
# -----------------------------
def projected_pseudospin(L, R):
    """
    Project onto the first two components (plasmonic subspace), then define
    biorthogonal pseudospin expectation values:

        n_a = <L_pl | sigma_a | R_pl> / <L_pl | R_pl>

    Returns:
        nx, ny, nz      (complex, in general)
        rho_pl_R        right-state plasmonic weight
        rho_pl_L        left-state plasmonic weight
        overlap_pl      projected biorthogonal overlap
        phase_rigidity  full-state phase rigidity
    """
    Rpl = R[:2]
    Lpl = L[:2]

    overlap_pl = np.vdot(Lpl, Rpl)
    eps = 1e-12

    if abs(overlap_pl) < eps:
        nx = np.nan + 1j * np.nan
        ny = np.nan + 1j * np.nan
        nz = np.nan + 1j * np.nan
    else:
        nx = np.vdot(Lpl, sigma_x @ Rpl) / overlap_pl
        ny = np.vdot(Lpl, sigma_y @ Rpl) / overlap_pl
        nz = np.vdot(Lpl, sigma_z @ Rpl) / overlap_pl

    rho_pl_R = np.vdot(Rpl, Rpl).real
    rho_pl_L = np.vdot(Lpl, Lpl).real

    num = np.abs(np.vdot(L, R))
    den = np.sqrt(np.vdot(R, R).real * np.vdot(L, L).real)
    phase_rigidity = num / den if den > eps else np.nan

    return nx, ny, nz, rho_pl_R, rho_pl_L, overlap_pl, phase_rigidity


# -----------------------------
# Scan one branch versus S
# -----------------------------
def scan_branch(branch, delta_w, delta_g, kappa, omega3):
    """
    For a chosen eigen-branch (0,1,2 after sorting by Re lambda),
    scan over S and compute pseudospin texture.
    """
    evals = np.zeros(len(S_grid), dtype=complex)

    nx = np.zeros(len(S_grid), dtype=complex)
    ny = np.zeros(len(S_grid), dtype=complex)
    nz = np.zeros(len(S_grid), dtype=complex)

    rhoR = np.zeros(len(S_grid))
    rhoL = np.zeros(len(S_grid))
    overlap_pl = np.zeros(len(S_grid), dtype=complex)
    rigidity = np.zeros(len(S_grid))

    for i, S in enumerate(S_grid):
        H = build_H(S, omega, gamma, delta_w, delta_g, kappa, omega3)
        vals, VR, VL = biorthogonal_eigensystem(H)

        evals[i] = vals[branch]
        nxi, nyi, nzi, rr, rl, ov, rig = projected_pseudospin(VL[:, branch], VR[:, branch])

        nx[i] = nxi
        ny[i] = nyi
        nz[i] = nzi
        rhoR[i] = rr
        rhoL[i] = rl
        overlap_pl[i] = ov
        rigidity[i] = rig

    return {
        "evals": evals,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "rhoR": rhoR,
        "rhoL": rhoL,
        "overlap_pl": overlap_pl,
        "rigidity": rigidity
    }


# ============================================================
# Initial parameters
# ============================================================
delta_w0 = 0.05
delta_g0 = 0.03
kappa0 = 0.25
omega30 = 1.00
branch0 = 1
S_star0 = 0.30

data0 = scan_branch(branch0, delta_w0, delta_g0, kappa0, omega30)

# ============================================================
# Figure layout
# ============================================================
fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.2], height_ratios=[1, 1], wspace=0.28, hspace=0.28)
ax_left = fig.add_subplot(gs[:, 0])   # pseudospin trajectory
ax_rt = fig.add_subplot(gs[0, 1])     # nx,nz vs S
ax_rb = fig.add_subplot(gs[1, 1])     # weights, rigidity vs S

plt.subplots_adjust(bottom=0.28)

# ============================================================
# Left panel: pseudospin trajectory in (Re nx, Re nz)
# ============================================================
phi = np.linspace(0, 2 * np.pi, 400)
ax_left.plot(np.cos(phi), np.sin(phi), '--', lw=1, label='Bloch circle reference')

traj_line, = ax_left.plot(
    np.real(data0["nx"]), np.real(data0["nz"]),
    lw=2, label='trajectory'
)

idx0 = np.argmin(np.abs(S_grid - S_star0))
marker_left, = ax_left.plot(
    [np.real(data0["nx"][idx0])],
    [np.real(data0["nz"][idx0])],
    'o', ms=8, label=r'current $S_\star$'
)

ax_left.set_xlabel(r'$\mathrm{Re}\, n_x$')
ax_left.set_ylabel(r'$\mathrm{Re}\, n_z$')
ax_left.set_title('Projected pseudospin trajectory')
ax_left.set_aspect('equal')
ax_left.grid(True)
ax_left.legend(loc='best')

info_left = ax_left.text(
    0.02, 0.98, '', transform=ax_left.transAxes, va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
)

# ============================================================
# Right-top: Re(nx), Re(nz), optional Im parts
# ============================================================
line_nx_re, = ax_rt.plot(S_grid, np.real(data0["nx"]), lw=2, label=r'$\mathrm{Re}\,n_x$')
line_nz_re, = ax_rt.plot(S_grid, np.real(data0["nz"]), lw=2, label=r'$\mathrm{Re}\,n_z$')

line_nx_im, = ax_rt.plot(S_grid, np.imag(data0["nx"]), '--', lw=1.4, label=r'$\mathrm{Im}\,n_x$')
line_nz_im, = ax_rt.plot(S_grid, np.imag(data0["nz"]), '--', lw=1.4, label=r'$\mathrm{Im}\,n_z$')

vline_rt = ax_rt.axvline(S_star0, ls='--')
ax_rt.set_ylabel('pseudospin components')
ax_rt.set_title('Pseudospin texture vs sliding')
ax_rt.grid(True)
ax_rt.legend(loc='best', ncol=2)

# ============================================================
# Right-bottom: plasmonic weight and rigidity
# ============================================================
line_rhoR, = ax_rb.plot(S_grid, data0["rhoR"], lw=2, label=r'right plasmonic weight $\rho_{\rm pl}^{(R)}$')
line_rhoL, = ax_rb.plot(S_grid, data0["rhoL"], lw=2, label=r'left plasmonic weight $\rho_{\rm pl}^{(L)}$')
line_rig,  = ax_rb.plot(S_grid, data0["rigidity"], lw=2, label='phase rigidity')

vline_rb = ax_rb.axvline(S_star0, ls='--')

ax_rb.set_xlabel('Sliding parameter S')
ax_rb.set_ylabel('weight / rigidity')
ax_rb.set_title('Support of the pseudospin picture')
ax_rb.grid(True)
ax_rb.set_ylim(-0.02, 1.05)
ax_rb.legend(loc='best')

info_rb = ax_rb.text(
    0.02, 0.98, '', transform=ax_rb.transAxes, va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
)

# ============================================================
# Sliders
# ============================================================
ax_dw = plt.axes([0.10, 0.18, 0.32, 0.03])
ax_dg = plt.axes([0.10, 0.13, 0.32, 0.03])
ax_kp = plt.axes([0.10, 0.08, 0.32, 0.03])

ax_w3 = plt.axes([0.56, 0.18, 0.32, 0.03])
ax_br = plt.axes([0.56, 0.13, 0.18, 0.03])
ax_ss = plt.axes([0.56, 0.08, 0.32, 0.03])

s_dw = Slider(ax_dw, r'$\delta\omega$', 0.0, 0.25, valinit=delta_w0)
s_dg = Slider(ax_dg, r'$\delta\gamma$', 0.0, 0.20, valinit=delta_g0)
s_kp = Slider(ax_kp, r'$\kappa$', 0.0, 0.60, valinit=kappa0)

s_w3 = Slider(ax_w3, r'$\omega_3$', 0.70, 1.30, valinit=omega30)
s_br = Slider(ax_br, 'branch', 0, 2, valinit=branch0, valstep=1)
s_ss = Slider(ax_ss, r'$S_\star$', 0.0, 1.0, valinit=S_star0)


# ============================================================
# Update function
# ============================================================
def update(_):
    delta_w = s_dw.val
    delta_g = s_dg.val
    kappa = s_kp.val
    omega3 = s_w3.val
    branch = int(s_br.val)
    S_star = s_ss.val

    data = scan_branch(branch, delta_w, delta_g, kappa, omega3)
    idx = np.argmin(np.abs(S_grid - S_star))

    # Left trajectory
    xtraj = np.real(data["nx"])
    ztraj = np.real(data["nz"])
    traj_line.set_data(xtraj, ztraj)
    marker_left.set_data([xtraj[idx]], [ztraj[idx]])

    # dynamic limits
    allxz = np.concatenate([xtraj[np.isfinite(xtraj)], ztraj[np.isfinite(ztraj)]])
    if len(allxz) > 0:
        lim = max(1.05, 1.15 * np.nanmax(np.abs(allxz)))
    else:
        lim = 1.2
    ax_left.set_xlim(-lim, lim)
    ax_left.set_ylim(-lim, lim)

    # Right-top
    line_nx_re.set_ydata(np.real(data["nx"]))
    line_nz_re.set_ydata(np.real(data["nz"]))
    line_nx_im.set_ydata(np.imag(data["nx"]))
    line_nz_im.set_ydata(np.imag(data["nz"]))
    vline_rt.set_xdata([S_star, S_star])

    y_top = np.concatenate([
        np.real(data["nx"]), np.real(data["nz"]),
        np.imag(data["nx"]), np.imag(data["nz"])
    ])
    y_top = y_top[np.isfinite(y_top)]
    if len(y_top) > 0:
        margin = 0.08 * (np.max(y_top) - np.min(y_top) + 1e-8)
        ax_rt.set_ylim(np.min(y_top) - margin, np.max(y_top) + margin)

    # Right-bottom
    line_rhoR.set_ydata(data["rhoR"])
    line_rhoL.set_ydata(data["rhoL"])
    line_rig.set_ydata(data["rigidity"])
    vline_rb.set_xdata([S_star, S_star])

    # Text boxes
    lam = data["evals"][idx]
    nx_star = data["nx"][idx]
    ny_star = data["ny"][idx]
    nz_star = data["nz"][idx]

    info_left.set_text(
        f"branch = {branch}\n"
        f"S* = {S_star:.3f}\n"
        f"lambda = {lam.real:.4f} {lam.imag:+.4f}i\n"
        f"Re(nx,nz)=({np.real(nx_star):.3f}, {np.real(nz_star):.3f})"
    )

    info_rb.set_text(
        f"rho_pl^(R) = {data['rhoR'][idx]:.3f}\n"
        f"rho_pl^(L) = {data['rhoL'][idx]:.3f}\n"
        f"phase rigidity = {data['rigidity'][idx]:.3f}\n"
        f"Im(nx,ny,nz)=({np.imag(nx_star):.3f}, {np.imag(ny_star):.3f}, {np.imag(nz_star):.3f})"
    )

    fig.canvas.draw_idle()


for s in [s_dw, s_dg, s_kp, s_w3, s_br, s_ss]:
    s.on_changed(update)

update(None)
plt.show()
