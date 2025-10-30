import numpy as np
import matplotlib.pyplot as plt

# Parameters for the dispersion
k = np.linspace(-np.pi, np.pi, 400)
omega0 = 1.0  # bright mode center frequency
delta = 0.2   # dispersion amplitude for bright mode
omega_dark = 1.0  # dark mode frequency
g = 0.1       # coupling strength

# Bright mode dispersion
omega_bright = omega0 + delta * np.cos(k)
# Dark mode (flat band)
omega_dark_band = np.ones_like(k) * omega_dark

# Coupled modes eigenfrequencies
omega1 = 0.5 * (omega_bright + omega_dark_band) + 0.5 * np.sqrt((omega_bright - omega_dark_band)**2 + 4 * g**2)
omega2 = 0.5 * (omega_bright + omega_dark_band) - 0.5 * np.sqrt((omega_bright - omega_dark_band)**2 + 4 * g**2)

# Plot
plt.figure()
plt.plot(k, omega1, label='Upper branch')
plt.plot(k, omega2, label='Lower branch')
plt.xlabel('Wavevector k')
plt.ylabel('Frequency ω')
plt.title('Coupled-Band Dispersion with EIT Window')
plt.legend()
plt.tight_layout()
plt.show()
