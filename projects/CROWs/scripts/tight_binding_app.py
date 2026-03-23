import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def dispersion(q, kappa, J, phi, omega_B=0.0, detuning=0.0):
    omega0 = omega_B + 2.0 * J * np.cos(q) * np.cos(phi)
    dz = 0.5 * detuning - 2.0 * J * np.sin(q) * np.sin(phi)
    split = np.sqrt(kappa**2 + dz**2)
    return omega0 + split, omega0 - split


def phi_from_shift(dy, ay, phase_factor=1.0):
    G = 2.0 * np.pi / ay
    return phase_factor * G * dy


def main():
    # Lattice constants
    ax = 1.0
    ay = 1.0

    # Model parameters
    kappa = 0.5
    J = -0.5
    omega_B = 0.0
    detuning = 0.0

    print(f"gap (standard) = 2|kappa| = {2.0*abs(kappa):.6f}")

    # Dimensionless BZ: q = kx*ax
    q = np.linspace(-np.pi, np.pi, 1601)

    # IMPORTANT mapping
    phase_factor = 1.0

    # Initial dy
    dy0 = 0.0

    # --- Figure & axes
    fig, axp = plt.subplots()
    plt.subplots_adjust(bottom=0.18)  # leave room for slider

    # Initial plot
    phi0 = phi_from_shift(dy=dy0, ay=ay, phase_factor=phase_factor)
    w_plus, w_minus = dispersion(q, kappa, J, phi0, omega_B, detuning)

    (line_p,) = axp.plot(q, w_plus, label="ω+(q)")
    (line_m,) = axp.plot(q, w_minus, label="ω−(q)")

    axp.set_xlabel("q = kx * ax (rad)")
    axp.set_ylabel("Eigenfrequency ω (arb. units)")
    axp.grid(True)
    axp.legend()

    title_obj = axp.set_title(
        f"Δy/ay = {dy0/ay:.3f}   |   phi = {phi0/np.pi:.3f}π   (phase_factor={phase_factor})"
    )

    # --- Slider (Δy)
    # Slider axis: [left, bottom, width, height] in figure coordinates
    slider_ax = fig.add_axes([0.12, 0.06, 0.76, 0.04])
    dy_slider = Slider(
        ax=slider_ax,
        label="Δy (in units of ay)",
        valmin=0.0,
        valmax=ay,        # slide 0 -> ay
        valinit=dy0,
        valstep=ay/400.0  # optional: snapping step; remove if you want continuous
    )

    def update(val):
        dy = dy_slider.val
        phi = phi_from_shift(dy=dy, ay=ay, phase_factor=phase_factor)
        wp, wm = dispersion(q, kappa, J, phi, omega_B, detuning)

        line_p.set_ydata(wp)
        line_m.set_ydata(wm)

        title_obj.set_text(
            f"Δy/ay = {dy/ay:.3f}   |   phi = {phi/np.pi:.3f}π   (phase_factor={phase_factor})"
        )

        # Optional: auto-rescale y each time (comment out if you want fixed y-range)
        axp.relim()
        axp.autoscale_view()

        fig.canvas.draw_idle()

    dy_slider.on_changed(update)

    plt.show()

    # Quick check prints
    for dy in [0.25 * ay, 0.5 * ay, 1.0 * ay]:
        phi = phi_from_shift(dy=dy, ay=ay, phase_factor=phase_factor)
        print(f"dy/ay={dy/ay:.2f} -> phi = {phi/np.pi:.2f}π")


if __name__ == "__main__":
    main()
