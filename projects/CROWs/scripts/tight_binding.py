"""
Tilted-lattice nonlocal metasurface: 2x2 effective model (with tunable phase factor)

Key equations (standard notation):
- ax: lattice constant along x
- ay: period of the y-modulation (bumps)
- q = kx * ax  in [-pi, pi]
- G = 2*pi/ay

We use a 2x2 effective Hamiltonian:
H(q) = [ωB + 2J cos(q) cos(phi)] I
     + [Δ/2 - 2J sin(q) sin(phi)] σz
     + κ σx

where the Peierls phase is:
phi = phase_factor * G * Δy

Important:
- If the relevant harmonics are centered around ky ≈ ±G/2 (Bragg edge), phase_factor ≈ 0.5.
- If the relevant coupling samples a full G-shift (e.g., ky ≈ 0 <-> ky ≈ ±G channel),
  then phase_factor ≈ 1.0.

Your observation ("my Δy=0.5 matches your sim 0.25") indicates phase_factor=1.0.
"""

import numpy as np
import matplotlib.pyplot as plt


def dispersion(q, kappa, J, phi, omega_B=0.0, detuning=0.0):
    """
    Eigen-dispersions ω±(q) of:
      H = (omega_B + 2J cos q cos phi) I
        + (detuning/2 - 2J sin q sin phi) σz
        + kappa σx
    """
    omega0 = omega_B + 2.0 * J * np.cos(q) * np.cos(phi)
    dz = 0.5 * detuning - 2.0 * J * np.sin(q) * np.sin(phi)
    split = np.sqrt(kappa**2 + dz**2)
    return omega0 + split, omega0 - split


def phi_from_shift(dy, ay, phase_factor=1.0):
    """
    phi = phase_factor * G * dy, with G = 2*pi/ay

    phase_factor:
      0.5 -> phi = (G/2)*dy (Bragg-edge pair ky≈±G/2)
      1.0 -> phi = G*dy     (your case, based on observed mapping)
    """
    G = 2.0 * np.pi / ay
    return phase_factor * G * dy


def main():
    # Lattice constants (use normalized units if you like)
    ax = 1.0
    ay = 1.0

    # Model parameters
    kappa = 0.1          # gap = 2|kappa| if detuning=0, J=0
    J = -0.5             # nearest-neighbor coupling along x
    omega_B = 0.0
    detuning = 0.0

    print(f"gap (standard) = 2|kappa| = {2.0*abs(kappa):.6f}")

    # Dimensionless BZ: q = kx*ax
    q = np.linspace(-np.pi, np.pi, 1601)

    # >>> IMPORTANT: use phase_factor=1.0 to match your simulation mapping
    phase_factor = 1.0

    # Compare the three offsets you mentioned (in units of ay)
    offsets = [
        ("Δy = 0", 0.0),
        ("Δy = ay/8", ay / 8.0),
        ("Δy = ay/4", ay / 4.0),
        ("Δy = ay/2", ay / 2.0),
    ]

    for title, dy in offsets:
        phi = phi_from_shift(dy=dy, ay=ay, phase_factor=phase_factor)
        w_plus, w_minus = dispersion(q, kappa, J, phi, omega_B, detuning)

        plt.figure()
        plt.plot(q, w_plus, label="ω+(q)")
        plt.plot(q, w_minus, label="ω−(q)")
        plt.xlabel("q = kx * ax (rad)")
        plt.ylabel("Eigenfrequency ω (arb. units)")
        plt.title(f"{title}   |   phi = {phi/np.pi:.2f}π   (phase_factor={phase_factor})")
        plt.grid(True)
        plt.legend()

    plt.show()

    # Quick check: show what phi becomes for your mapping intuition
    for dy in [0.25 * ay, 0.5 * ay, 1.0 * ay]:
        phi = phi_from_shift(dy=dy, ay=ay, phase_factor=phase_factor)
        print(f"dy/ay={dy/ay:.2f} -> phi = {phi/np.pi:.2f}π")


if __name__ == "__main__":
    main()
