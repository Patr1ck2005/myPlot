import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Parameters
ky = np.linspace(-np.pi, np.pi, 500)
init_delta = 1.0
init_gamma = 0.1
init_t_same = -0.5
init_t_cross = 0.0
init_phi = 0.0

# Function to compute eigenvalues
def compute_bands(delta, gamma, t_same, t_cross, phi):
    omega1 = np.zeros_like(ky, dtype=complex)
    omega2 = np.zeros_like(ky, dtype=complex)
    for i, k in enumerate(ky):
        diag_same = 2 * t_same * np.cos(k)
        v = 2 * t_cross * np.cos(k + phi)
        H = np.array([[-delta + diag_same, v],
                      [v, delta - 1j * gamma / 2 + diag_same]])
        eigvals = np.linalg.eigvals(H)
        # Sort by imaginary part (BIC-like: less negative Im, QGM-like: more negative)
        idx = np.argsort(np.imag(eigvals))
        omega1[i] = eigvals[idx[1]]  # BIC-like
        omega2[i] = eigvals[idx[0]]  # QGM-like
    return omega1, omega2

# Initial computation
omega1, omega2 = compute_bands(init_delta, init_gamma, init_t_same, init_t_cross, init_phi)

# Plot setup
fig, (ax_re, ax_im) = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(bottom=0.35)
line_re1, = ax_re.plot(ky, np.real(omega1), label='BIC-like')
line_re2, = ax_re.plot(ky, np.real(omega2), label='QGM-like')
ax_re.set_title('Re(ω) vs ky')
ax_re.set_xlabel('ky')
ax_re.set_ylabel('Re(ω)')
ax_re.legend()
ax_re.grid()

line_im1, = ax_im.plot(ky, np.imag(omega1), label='BIC-like')
line_im2, = ax_im.plot(ky, np.imag(omega2), label='QGM-like')
ax_im.set_title('Im(ω) vs ky')
ax_im.set_xlabel('ky')
ax_im.set_ylabel('Im(ω)')
ax_im.legend()
ax_im.grid()

# Sliders
ax_delta = plt.axes([0.1, 0.25, 0.8, 0.03])
s_delta = Slider(ax_delta, 'δ', 0.1, 5.0, valinit=init_delta)
ax_gamma = plt.axes([0.1, 0.20, 0.8, 0.03])
s_gamma = Slider(ax_gamma, 'γ', 0.01, 2.0, valinit=init_gamma)
ax_t_same = plt.axes([0.1, 0.15, 0.8, 0.03])
s_t_same = Slider(ax_t_same, 't_same', -1.0, 1.0, valinit=init_t_same)
ax_t_cross = plt.axes([0.1, 0.10, 0.8, 0.03])
s_t_cross = Slider(ax_t_cross, 't_cross', 0.0, 2.0, valinit=init_t_cross)
ax_phi = plt.axes([0.1, 0.05, 0.8, 0.03])
s_phi = Slider(ax_phi, 'ϕ', 0.0, np.pi, valinit=init_phi)

# Update function
def update(val):
    delta = s_delta.val
    gamma = s_gamma.val
    t_same = s_t_same.val
    t_cross = s_t_cross.val
    phi = s_phi.val
    omega1, omega2 = compute_bands(delta, gamma, t_same, t_cross, phi)
    line_re1.set_ydata(np.real(omega1))
    line_re2.set_ydata(np.real(omega2))
    line_im1.set_ydata(np.imag(omega1))
    line_im2.set_ydata(np.imag(omega2))
    ax_re.relim()
    ax_re.autoscale_view()
    ax_im.relim()
    ax_im.autoscale_view()
    fig.canvas.draw_idle()

s_delta.on_changed(update)
s_gamma.on_changed(update)
s_t_same.on_changed(update)
s_t_cross.on_changed(update)
s_phi.on_changed(update)

plt.show()
