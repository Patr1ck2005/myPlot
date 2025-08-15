import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from utils.advanced_color_mapping import plot_S1S2S3_color
from utils.functions import VBG_single_resonance_converted, skyrmion_density, skyrmion_number

# --- Parameters ---
l = 1      # Phase topological charge
m = 1      # Polarization topological charge
initial_polar = 1*np.pi  # Initial polarization of the beam
w = 0.5    # Beam waist

# --- Grid ---
scale_factor = 1
x = np.linspace(-1, 1, 1024*scale_factor)*scale_factor
y = np.linspace(-1, 1, 1024*scale_factor)*scale_factor
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)

# --- 1. LG beam amplitude & intensity ---
# A0 = (np.sqrt(2) * r / w) ** abs(l) * np.exp(-r ** 2 / w ** 2)  # amplitude profile
A0 = 1  # amplitude profile
complex_purcell_factor = VBG_single_resonance_converted(X, Y, omega=1.40, a=-1, Q_ref=10, gamma_slope=1.0, gamma_nr=0)
Intensity = np.abs(A0 * complex_purcell_factor)**2


# Plot LG intensity
plt.figure(figsize=(4, 4))
plt.imshow(Intensity, origin='lower', extent=(-1,1,-1,1), vmin=0)
plt.title('LG Beam Intensity |l=2>')
plt.axis('off')
plt.show()

# --- 2. Circular components with combined charges ---
q_L = l + m  # left-circular charge
q_R = l - m  # right-circular charge

# E_L = A0 * (1*abs(complex_purcell_factor)) * np.exp(1j * q_L * phi)  # Left circular component
# E_R = A0 * (1-abs(1*complex_purcell_factor)) * np.exp(1j * q_R * phi) * np.exp(1j * initial_polar)  # Right circular component
E_L = A0 * (.25*complex_purcell_factor) * np.exp(1j * q_L * phi)  # Left circular component
E_R = A0 * (.25-.25*complex_purcell_factor) * np.exp(1j * q_R * phi) * np.exp(1j * initial_polar)  # Right circular component

# --- 3. To linear basis (x, y) ---
# e_L = (ex + i ey)/√2, e_R = (ex - i ey)/√2
Ex = (E_L + E_R) / np.sqrt(2)
Ey = -1j*(E_L - E_R) / np.sqrt(2)

# --- 4. Strokes Parameters ---
S0 = np.abs(Ex)**2 + np.abs(Ey)**2
S1 = (np.abs(Ex)**2 - np.abs(Ey)**2)/S0
S2 = 2*np.real(Ex*np.conj(Ey))/S0
S3 = 2*np.imag(Ex*np.conj(Ey))/S0
S3_mask = S3 > 0

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

def binary_momentum_space_show(reals, title, max_abs=1, show=True, cmap='RdBu'):
    fig = plt.figure(figsize=(4, 3), dpi=100)
    plt.title(title)
    plt.imshow(
        reals, origin='lower',
        vmin=-max_abs, vmax=max_abs,
        cmap=cmap, extent=(-1,1,-1,1)
    )
    # plt.axis('equal')
    plt.tight_layout()
    plt.axis('off')
    plt.colorbar()
    if show:
        plt.show()

# Plot all fields
plot_S1S2S3_color(S1, S2, -S3, s3_mode='-11', show=True)
# 计算斯格明子密度
n_sk = skyrmion_density(S1, S2, S3)
n_sk[S3_mask] = 0
n_sk[r > 0.3] = 0
# 网格间隔（假设是均匀的）
dx = dy = 1  # 这里假设每个网格的间隔为1
# 计算斯格明子数
s = skyrmion_number(n_sk, dx, dy)
plot_complex(n_sk, r'$n_sk$')
plot_complex(E_L, r'Left Circular $E_L$')
plot_complex(E_R, r'Right Circular $E_R$')
# plot_complex(Ex, r'Linear $E_x$')
# plot_complex(Ey, r'Linear $E_y$')
binary_momentum_space_show(S1, r'S1')
binary_momentum_space_show(S2, r'S2')
binary_momentum_space_show(S3, r'S3')
# plot_complex(Es, r's-polarized $E_s$')
# plot_complex(Ep, r'p-polarized $E_p$')

