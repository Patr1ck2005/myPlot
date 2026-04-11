import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from utils import *

# ===== helper functions assumed already defined above =====
# build_H, eig_sorted

omega = 1.0
gamma = 0.12
S_scan = np.linspace(0.0, 1.0, 500)

def compute_trajectories(kappa, omega3, delta_w, delta_g):
    traj = np.zeros((3, len(S_scan)), dtype=complex)
    for i, S in enumerate(S_scan):
        H = build_H(S, omega, gamma, delta_w, delta_g, kappa, omega3)
        vals, _ = eig_sorted(H)
        traj[:, i] = vals
    return traj

# initial
kappa0 = 0.25
omega30 = 1.00
dw0 = 0.00
dg0 = 0.00
S0 = 0.25

traj0 = compute_trajectories(kappa0, omega30, dw0, dg0)

fig, ax = plt.subplots(figsize=(7, 6))
plt.subplots_adjust(bottom=0.28)

l1, = ax.plot(traj0[0].real, traj0[0].imag, lw=2, label='mode 1')
l2, = ax.plot(traj0[1].real, traj0[1].imag, lw=2, label='mode 2')
l3, = ax.plot(traj0[2].real, traj0[2].imag, lw=2, label='mode 3')

# current S markers
def current_vals(kappa, omega3, dw, dg, S):
    H = build_H(S, omega, gamma, dw, dg, kappa, omega3)
    vals, _ = eig_sorted(H)
    return vals

vals_cur = current_vals(kappa0, omega30, dw0, dg0, S0)
m1, = ax.plot(vals_cur[0].real, vals_cur[0].imag, 'o', ms=8)
m2, = ax.plot(vals_cur[1].real, vals_cur[1].imag, 'o', ms=8)
m3, = ax.plot(vals_cur[2].real, vals_cur[2].imag, 'o', ms=8)

ax.set_xlabel(r'Re$(\lambda)$')
ax.set_ylabel(r'Im$(\lambda)$')
ax.set_title('Complex-eigenvalue trajectories under sliding')
ax.grid(True)
ax.legend()

# sliders
ax_k = plt.axes([0.18, 0.18, 0.62, 0.03])
ax_w3 = plt.axes([0.18, 0.14, 0.62, 0.03])
ax_dw = plt.axes([0.18, 0.10, 0.62, 0.03])
ax_dg = plt.axes([0.18, 0.06, 0.62, 0.03])
ax_S = plt.axes([0.18, 0.02, 0.62, 0.03])

sk = Slider(ax_k, r'$\kappa$', 0.0, 0.6, valinit=kappa0)
sw3 = Slider(ax_w3, r'$\omega_3$', 0.7, 1.3, valinit=omega30)
sdw = Slider(ax_dw, r'$\delta\omega$', 0.0, 0.25, valinit=dw0)
sdg = Slider(ax_dg, r'$\delta\gamma$', 0.0, 0.20, valinit=dg0)
sS = Slider(ax_S, 'S', 0.0, 1.0, valinit=S0)

txt = ax.text(
    0.02, 0.98, '', transform=ax.transAxes, va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

def update(val):
    kappa = sk.val
    omega3 = sw3.val
    dw = sdw.val
    dg = sdg.val
    S = sS.val

    traj = compute_trajectories(kappa, omega3, dw, dg)
    l1.set_data(traj[0].real, traj[0].imag)
    l2.set_data(traj[1].real, traj[1].imag)
    l3.set_data(traj[2].real, traj[2].imag)

    vals = current_vals(kappa, omega3, dw, dg, S)
    m1.set_data([vals[0].real], [vals[0].imag])
    m2.set_data([vals[1].real], [vals[1].imag])
    m3.set_data([vals[2].real], [vals[2].imag])

    allx = traj.real.flatten()
    ally = traj.imag.flatten()
    mx = 0.08 * (allx.max() - allx.min() + 1e-6)
    my = 0.08 * (ally.max() - ally.min() + 1e-6)
    ax.set_xlim(allx.min() - mx, allx.max() + mx)
    ax.set_ylim(ally.min() - my, ally.max() + my)

    txt.set_text(
        f"kappa = {kappa:.3f}, omega3 = {omega3:.3f}\n"
        f"delta_w = {dw:.3f}, delta_g = {dg:.3f}\n"
        f"S = {S:.3f}"
    )

    fig.canvas.draw_idle()

for s in [sk, sw3, sdw, sdg, sS]:
    s.on_changed(update)

update(None)
plt.show()
