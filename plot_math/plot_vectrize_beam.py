import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# --- Parameters ---
l = 3      # Phase topological charge
m = -3     # Polarization topological charge
initial_polar = 0*np.pi  # Initial polarization of the beam
w = 0.3    # Beam waist

# --- Grid ---
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)

# --- 1. LG beam amplitude & intensity ---
def LG_beam(r, phi, w=0.5, l=2):
    return (np.sqrt(2) * r / w)**abs(l) * np.exp(-r**2 / w**2) * np.exp(1j * l * phi)
Intensity = np.abs(LG_beam(r, phi, w, l))**2

# Plot LG intensity
plt.figure(figsize=(4, 4))
plt.imshow(Intensity, origin='lower', extent=(-1,1,-1,1))
plt.title('LG Beam Intensity |l=2>')
plt.axis('off')
plt.show()

# --- 2. Circular components with combined charges ---
q_L = l + m  # left-circular charge
q_R = l - m  # right-circular charge

E_L = abs(LG_beam(r, phi, w, 4)) * np.exp(1j*phi*q_L)  # Left circular component
E_R = abs(LG_beam(r, phi, w, 4)) * np.exp(1j*phi*q_R) * np.exp(1j*initial_polar)  # Right circular component

# --- 3. To linear basis (x, y) ---
# e_L = (ex + i ey)/√2, e_R = (ex - i ey)/√2
Ex = (E_L + E_R) / np.sqrt(2)
Ey = -1j*(E_L - E_R) / np.sqrt(2)

# --- 4. Strokes Parameters ---
S0 = np.abs(Ex)**2 + np.abs(Ey)**2
S1 = np.abs(Ex)**2 - np.abs(Ey)**2
S2 = 2*np.real(Ex*np.conj(Ey))
S3 = 2*np.imag(Ex*np.conj(Ey))
S4 = np.abs(E_L)**2 + np.abs(E_R)**2

# --- 4. To s, p basis via rotation matrix ---
Es = -np.sin(phi)*Ex + np.cos(phi)*Ey
Ep = np.cos(phi)*Ex + np.sin(phi)*Ey

# --- 5. Visualization function ---
def plot_complex(field, title):
    amp = np.abs(field)
    phase = np.angle(field)
    amp_norm = amp / amp.max()
    hue = (phase + np.pi) / (2*np.pi)
    hsv = np.stack((hue, np.ones_like(hue), amp_norm), axis=-1)
    rgb = hsv_to_rgb(hsv)
    plt.figure(figsize=(4,4))
    plt.imshow(rgb, origin='lower', extent=(-1,1,-1,1))
    plt.title(title)
    plt.axis('off')
    plt.show()

plt.imshow(np.angle(Es), origin='lower', extent=(-1, 1, -1, 1), cmap='rainbow')
plt.colorbar()
plt.show()
plt.imshow(np.real(Es), origin='lower', extent=(-1, 1, -1, 1), cmap='rainbow')
plt.colorbar()
plt.show()

# Plot all fields
plot_complex(E_L, r'Left Circular $E_L$')
plot_complex(E_R, r'Right Circular $E_R$')
# plot_complex(Ex, r'Linear $E_x$')
# plot_complex(Ey, r'Linear $E_y$')
# plot_complex(S1, r'S1')
# plot_complex(S2, r'S2')
# plot_complex(S3, r'S3')
plot_complex(Es, r's-polarized $E_s$')
plot_complex(np.real(Es), r's-polarized $E_s$')
plot_complex(Ep, r'p-polarized $E_p$')


