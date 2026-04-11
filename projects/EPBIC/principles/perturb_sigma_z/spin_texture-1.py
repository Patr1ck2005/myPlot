import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from utils import *
# ===== helper functions assumed already defined above =====
# build_H, biorthogonal_data

omega = 1.0
gamma = 0.12
S_scan = np.linspace(0.0, 1.0, 500)

def state_scan(kappa, omega3, delta_w, delta_g, branch=1):
    """
    branch = 0, 1, 2 after sorting by Re(eigenvalue)
    """
    w1 = np.zeros_like(S_scan)
    w2 = np.zeros_like(S_scan)
    w3 = np.zeros_like(S_scan)
    rigidity = np.zeros_like(S_scan)
    evals = np.zeros_like(S_scan, dtype=complex)

    for i, S in enumerate(S_scan):
        H = build_H(S, omega, gamma, delta_w, delta_g, kappa, omega3)
        vals, VR, VL, rig = biorthogonal_data(H)

        v = VR[:, branch]
        v = v / np.linalg.norm(v)

        w1[i] = np.abs(v[0])**2
        w2[i] = np.abs(v[1])**2
        w3[i] = np.abs(v[2])**2
        rigidity[i] = rig[branch]
        evals[i] = vals[branch]

    return w1, w2, w3, rigidity, evals

# initial
kappa0 = 0.25
omega30 = 1.00
dw0 = 0.05
dg0 = 0.03
branch0 = 1

w10, w20, w30, rig0, ev0 = state_scan(kappa0, omega30, dw0, dg0, branch=branch0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
plt.subplots_adjust(bottom=0.28)

l1, = ax1.plot(S_scan, w10, lw=2, label=r'$|c_{p1}|^2$')
l2, = ax1.plot(S_scan, w20, lw=2, label=r'$|c_{p2}|^2$')
l3, = ax1.plot(S_scan, w30, lw=2, label=r'$|c_{3}|^2$')

ax1.set_ylabel('modal weight')
ax1.set_title('Mode composition versus sliding')
ax1.grid(True)
ax1.legend()

lr, = ax2.plot(S_scan, rig0, lw=2, label='phase rigidity')
ax2.set_xlabel('S')
ax2.set_ylabel('rigidity')
ax2.set_ylim(0, 1.05)
ax2.grid(True)
ax2.legend()

# current S vertical line
S0 = 0.3
vline1 = ax1.axvline(S0, ls='--')
vline2 = ax2.axvline(S0, ls='--')

# sliders
ax_k = plt.axes([0.18, 0.18, 0.62, 0.03])
ax_w3 = plt.axes([0.18, 0.14, 0.62, 0.03])
ax_dw = plt.axes([0.18, 0.10, 0.62, 0.03])
ax_dg = plt.axes([0.18, 0.06, 0.62, 0.03])
ax_b = plt.axes([0.18, 0.02, 0.18, 0.03])
ax_S = plt.axes([0.50, 0.02, 0.30, 0.03])

sk = Slider(ax_k, r'$\kappa$', 0.0, 0.6, valinit=kappa0)
sw3 = Slider(ax_w3, r'$\omega_3$', 0.7, 1.3, valinit=omega30)
sdw = Slider(ax_dw, r'$\delta\omega$', 0.0, 0.25, valinit=dw0)
sdg = Slider(ax_dg, r'$\delta\gamma$', 0.0, 0.20, valinit=dg0)
sb = Slider(ax_b, 'branch', 0, 2, valinit=branch0, valstep=1)
sS = Slider(ax_S, 'S', 0.0, 1.0, valinit=S0)

txt = ax1.text(
    0.02, 0.98, '', transform=ax1.transAxes, va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

def update(val):
    kappa = sk.val
    omega3 = sw3.val
    dw = sdw.val
    dg = sdg.val
    branch = int(sb.val)
    S = sS.val

    w1, w2, w3, rig, ev = state_scan(kappa, omega3, dw, dg, branch=branch)

    l1.set_ydata(w1)
    l2.set_ydata(w2)
    l3.set_ydata(w3)
    lr.set_ydata(rig)

    vline1.set_xdata([S, S])
    vline2.set_xdata([S, S])

    idx = np.argmin(np.abs(S_scan - S))
    txt.set_text(
        f"branch = {branch}\n"
        f"lambda(S) = {ev[idx].real:.4f} {ev[idx].imag:+.4f}i\n"
        f"rigidity(S) = {rig[idx]:.4f}"
    )

    fig.canvas.draw_idle()

for s in [sk, sw3, sdw, sdg, sb, sS]:
    s.on_changed(update)

update(None)
plt.show()
