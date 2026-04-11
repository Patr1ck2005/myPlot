import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------------
# Figure 4: eigenvalue spectrum vs S
# -----------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
plt.subplots_adjust(bottom=0.30)

# Initial parameters
omega0 = 1.0
gamma0 = 0.15
delta0 = 0.20
kappa0 = 0.25
omega30 = 1.05
npts = 500

S_arr = np.linspace(0, 1, npts)

def build_H(S, omega, gamma, delta, kappa, omega3):
    theta = 2 * np.pi * S
    # simple reduced model:
    # H_pl = (omega - i gamma) I + delta sigma_z
    # total 3x3 Hamiltonian
    H = np.array([
        [omega - 1j*gamma + delta, 0, kappa*np.cos(theta)],
        [0, omega - 1j*gamma - delta, kappa*np.sin(theta)],
        [kappa*np.cos(theta), kappa*np.sin(theta), omega3]
    ], dtype=complex)
    return H

def spectrum_vs_S(omega, gamma, delta, kappa, omega3):
    eigvals = np.zeros((3, len(S_arr)), dtype=complex)
    for i, S in enumerate(S_arr):
        H = build_H(S, omega, gamma, delta, kappa, omega3)
        vals = np.linalg.eigvals(H)
        vals = vals[np.argsort(vals.real)]
        eigvals[:, i] = vals
    return eigvals

eigvals = spectrum_vs_S(omega0, gamma0, delta0, kappa0, omega30)

line_r1, = ax.plot(S_arr, eigvals[0].real, lw=2, label='Re λ1')
line_r2, = ax.plot(S_arr, eigvals[1].real, lw=2, label='Re λ2')
line_r3, = ax.plot(S_arr, eigvals[2].real, lw=2, label='Re λ3')

line_i1, = ax.plot(S_arr, eigvals[0].imag, '--', lw=1.8, label='Im λ1')
line_i2, = ax.plot(S_arr, eigvals[1].imag, '--', lw=1.8, label='Im λ2')
line_i3, = ax.plot(S_arr, eigvals[2].imag, '--', lw=1.8, label='Im λ3')

ax.set_xlabel('Sliding parameter S')
ax.set_ylabel('Eigenvalues')
ax.set_title('Spectrum vs sliding')
ax.grid(True)
ax.legend(ncol=3, fontsize=9)

text_info = ax.text(
    0.02, 0.94, '', transform=ax.transAxes, fontsize=10,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# sliders
ax_omega3 = plt.axes([0.18, 0.18, 0.65, 0.03])
ax_kappa = plt.axes([0.18, 0.13, 0.65, 0.03])
ax_delta = plt.axes([0.18, 0.08, 0.65, 0.03])
ax_gamma = plt.axes([0.18, 0.03, 0.65, 0.03])

slider_omega3 = Slider(ax_omega3, r'$\omega_3$', 0.7, 1.3, valinit=omega30)
slider_kappa = Slider(ax_kappa, r'$\kappa$', 0.0, 0.6, valinit=kappa0)
slider_delta = Slider(ax_delta, r'$\delta$', 0.0, 0.5, valinit=delta0)
slider_gamma = Slider(ax_gamma, r'$\gamma$', 0.0, 0.4, valinit=gamma0)

def update(val):
    omega3 = slider_omega3.val
    kappa = slider_kappa.val
    delta = slider_delta.val
    gamma = slider_gamma.val

    eigvals = spectrum_vs_S(omega0, gamma, delta, kappa, omega3)

    line_r1.set_ydata(eigvals[0].real)
    line_r2.set_ydata(eigvals[1].real)
    line_r3.set_ydata(eigvals[2].real)

    line_i1.set_ydata(eigvals[0].imag)
    line_i2.set_ydata(eigvals[1].imag)
    line_i3.set_ydata(eigvals[2].imag)

    y_all = np.concatenate([eigvals.real.flatten(), eigvals.imag.flatten()])
    margin = 0.08 * (y_all.max() - y_all.min() + 1e-6)
    ax.set_ylim(y_all.min() - margin, y_all.max() + margin)

    if abs(delta) < 1e-8:
        extra = "delta = 0: spectrum is S-invariant (pure gauge)"
    else:
        extra = "delta > 0: sliding becomes observable"

    text_info.set_text(
        f"omega3 = {omega3:.3f}, kappa = {kappa:.3f}, gamma = {gamma:.3f}, delta = {delta:.3f}\n"
        f"{extra}"
    )

    fig.canvas.draw_idle()

slider_omega3.on_changed(update)
slider_kappa.on_changed(update)
slider_delta.on_changed(update)
slider_gamma.on_changed(update)

update(None)
plt.show()
