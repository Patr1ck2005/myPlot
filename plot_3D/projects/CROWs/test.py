import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def compute_dispersion(delta, gamma, t0, v0, deltax, G, Px=1.0, num_points=100):
    # ky range: -pi/G to pi/G (G=1 unitized, but scalable)
    ky = np.linspace(-np.pi / G, np.pi / G, num_points)

    # theta = 2 pi deltax / Px
    theta = 2 * np.pi * deltax / Px

    # Couplings
    t = t0 * np.cos(theta)  # Same-type
    v = v0 * np.sin(theta)  # Cross-type

    # Effective next-nearest neighbor for cross-dominant (with non-Hermitian)
    denom_qgm = 2 * delta + 1j * gamma / 2
    denom_bic = -2 * delta + 1j * gamma / 2  # Sign flip for BIC
    t_eff_qgm = - (v ** 2) / denom_qgm if abs(denom_qgm) > 1e-6 else 0
    t_eff_bic = - (v ** 2) / denom_bic if abs(denom_bic) > 1e-6 else 0

    # Dispersion: same-type cos(ky) + eff cross cos(2ky)
    # Add small offset if needed, but match user: no large delta
    omega_qgm = np.zeros(num_points, dtype=complex)
    omega_bic = np.zeros(num_points, dtype=complex)

    for i, k in enumerate(ky):
        cos_ky = np.cos(k * G)
        cos_2ky = np.cos(2 * k * G)

        # QGM-like: -i gamma/2 base loss, mixed
        omega_qgm[i] = delta + t * 2 * cos_ky + t_eff_qgm * 2 * cos_2ky - 1j * gamma / 2 * (
                    1 - abs(v) / (v0 + 1e-6))  # Loss adjustment

        # BIC-like: induced loss from mixing
        induced_loss = -1j * (gamma / 2) * (
                    abs(v) ** 2 / (4 * delta ** 2 + (gamma / 2) ** 2 + 1e-6))  # Perturbative sharing
        omega_bic[i] = -delta + t * 2 * cos_ky + t_eff_bic * 2 * cos_2ky + induced_loss

    return ky, omega_qgm, omega_bic


# Initial parameters (tuned to match user conclusions)
initial_delta = 0.1  # Small detuning
initial_gamma = 0.1
initial_t0 = -0.5  # For -cos(ky)
initial_v0 = 1.0  # For cos(2ky) amplitude ~1
initial_deltax = 0.1
initial_G = 1.0

# Create the figure and axes
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
plt.subplots_adjust(left=0.15, bottom=0.35)  # Space for sliders

# Initial computation and plot
ky, omega_qgm, omega_bic = compute_dispersion(initial_delta, initial_gamma, initial_t0, initial_v0, initial_deltax,
                                              initial_G)

# Plot lines (initial)
line_qgm_real, = axs[0].plot(ky, omega_qgm.real, label='QGM-like', color='blue')
line_bic_real, = axs[0].plot(ky, omega_bic.real, label='BIC-like', color='red')
axs[0].set_ylabel('Re(ω)')
axs[0].set_title('Dispersion: Re(ω) vs ky')
axs[0].legend()
axs[0].grid(True)

line_qgm_imag, = axs[1].plot(ky, omega_qgm.imag, label='QGM-like', color='blue')
line_bic_imag, = axs[1].plot(ky, omega_bic.imag, label='BIC-like', color='red')
axs[1].set_xlabel('ky (1/G)')
axs[1].set_ylabel('Im(ω) (negative for loss)')
axs[1].set_title('Radiation Loss: Im(ω) vs ky')
axs[1].legend()
axs[1].grid(True)

# Slider positions
ax_delta = plt.axes([0.15, 0.25, 0.65, 0.03])
ax_gamma = plt.axes([0.15, 0.20, 0.65, 0.03])
ax_t0 = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_v0 = plt.axes([0.15, 0.10, 0.65, 0.03])
ax_deltax = plt.axes([0.15, 0.05, 0.65, 0.03])
ax_G = plt.axes([0.15, 0.00, 0.65, 0.03])

# Create sliders
slider_delta = Slider(ax_delta, 'δ', 0.01, 1.0, valinit=initial_delta, valstep=0.01)
slider_gamma = Slider(ax_gamma, 'γ', 0.0, 1.0, valinit=initial_gamma, valstep=0.05)
slider_t0 = Slider(ax_t0, 't0', -1.0, 1.0, valinit=initial_t0, valstep=0.1)
slider_v0 = Slider(ax_v0, 'v0', 0.0, 2.0, valinit=initial_v0, valstep=0.1)
slider_deltax = Slider(ax_deltax, 'Δx', 0.0, 0.5, valinit=initial_deltax, valstep=0.01)
slider_G = Slider(ax_G, 'G', 0.5, 2.0, valinit=initial_G, valstep=0.1)


# Update function
def update(val):
    delta = slider_delta.val
    gamma = slider_gamma.val
    t0 = slider_t0.val
    v0 = slider_v0.val
    deltax = slider_deltax.val
    G = slider_G.val

    ky, omega_qgm, omega_bic = compute_dispersion(delta, gamma, t0, v0, deltax, G)

    # Update real
    line_qgm_real.set_ydata(omega_qgm.real)
    line_bic_real.set_ydata(omega_bic.real)

    # Update imag
    line_qgm_imag.set_ydata(omega_qgm.imag)
    line_bic_imag.set_ydata(omega_bic.imag)

    # Adjust x limits
    axs[0].set_xlim(-np.pi / G, np.pi / G)
    axs[1].set_xlim(-np.pi / G, np.pi / G)

    # Auto scale y
    # axs[0].relim()
    # axs[0].autoscale_view()
    # axs[1].relim()
    # axs[1].autoscale_view()

    fig.canvas.draw_idle()


# Attach updates
slider_delta.on_changed(update)
slider_gamma.on_changed(update)
slider_t0.on_changed(update)
slider_v0.on_changed(update)
slider_deltax.on_changed(update)
slider_G.on_changed(update)

plt.show()
