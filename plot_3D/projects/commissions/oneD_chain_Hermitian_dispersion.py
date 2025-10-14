import numpy as np
import matplotlib.pyplot as plt

# Define k range for Brillouin zone: -pi to pi
k = np.linspace(-np.pi, np.pi, 1000)

# Parameters: t1 = 1 (short-range), vary t2 (long-range)
t1 = 1.0
t2_values = [0.0, 0.1, 0.25, 0.5]  # 0: no long-range, 0.1: small, 0.25: tuned for flatness

plt.figure(figsize=(6, 3))

for t2 in t2_values:
    # Dispersion relation: E(k) = 2*t1*cos(k) + 2*t2*cos(2k)
    E = 2 * t1 * np.cos(k) + 2 * t2 * np.cos(2 * k)
    label = f't2 = {t2} (r = t2/t1 = {t2/t1})'
    plt.plot(k / np.pi, E, label=label)

plt.xlabel('k / π')
plt.ylabel('E(k)')
plt.title('Dispersion Relation in 1D Tight-Binding Model with Short and Long-Range Coupling')
plt.legend()
plt.grid(True)
plt.savefig('temp.svg', transparent=True, bbox_inches='tight')
plt.show()
