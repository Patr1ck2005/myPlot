import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
# Graphene DOS under magnetic field:
# B = 0  : D(E) ~ |E|
# B > 0  : broadened Landau levels
#
# This gives a continuous visualization from B=0 to B>0.
# ============================================================

# -------------------------
# Physical constants (SI)
# -------------------------
hbar = 1.054571817e-34      # J*s
e = 1.602176634e-19         # C
vF = 1.0e6                  # m/s
g = 4                       # spin * valley degeneracy

# -------------------------
# User-tunable parameters
# -------------------------
B_max = 12.0                # Tesla
E_max_meV = 250.0           # energy window
n_max = 80                  # number of Landau levels included
Gamma_meV = 4.0             # Gaussian broadening (meV)
num_E = 900
num_B = 220

# Slider initial value
B_init = 2.0

# ------------------------------------------------
# Helpers: units and physical formulas
# ------------------------------------------------
def meV_to_J(E_meV):
    return E_meV * 1e-3 * e

def J_to_meV(E_J):
    return E_J / e * 1e3

def magnetic_length(B):
    # l_B = sqrt(hbar / (e B))
    return np.sqrt(hbar / (e * B))

def graphene_LL_energy_J(n, B):
    # E_n = sgn(n) v_F sqrt(2 e hbar B |n|)
    if n == 0:
        return 0.0
    return np.sign(n) * vF * np.sqrt(2 * e * hbar * B * abs(n))

def gaussian(x, gamma):
    return np.exp(-0.5 * (x / gamma) ** 2) / (np.sqrt(2 * np.pi) * gamma)

# ------------------------------------------------
# B=0 DOS of graphene per unit area:
# D0(E) = g |E| / [2 pi (hbar vF)^2]
# units: 1 / (J m^2)
# ------------------------------------------------
def dos_B0(E_J):
    return g * np.abs(E_J) / (2 * np.pi * (hbar * vF) ** 2)

# ------------------------------------------------
# Finite-B DOS with Gaussian-broadened Landau levels
#
# D(E,B) = [g / (2 pi l_B^2)] * sum_n G(E - E_n, Gamma)
#
# As B -> 0, LL spacing shrinks and broadened sum approaches
# the continuous Dirac DOS.
# ------------------------------------------------
def dos_B(E_J_grid, B, gamma_J):
    if B <= 1e-10:
        return dos_B0(E_J_grid)

    lB = magnetic_length(B)
    prefactor = g / (2 * np.pi * lB ** 2)   # degeneracy per unit area per LL

    D = np.zeros_like(E_J_grid)

    # n = 0
    D += prefactor * gaussian(E_J_grid, gamma_J)

    # n = ±1, ±2, ...
    for n in range(1, n_max + 1):
        En = graphene_LL_energy_J(n, B)
        D += prefactor * gaussian(E_J_grid - En, gamma_J)
        D += prefactor * gaussian(E_J_grid + En, gamma_J)

    return D

# ------------------------------------------------
# Energy and B grids
# ------------------------------------------------
E_meV = np.linspace(-E_max_meV, E_max_meV, num_E)
E_J = meV_to_J(E_meV)

B_vals = np.linspace(0.0, B_max, num_B)
gamma_J = meV_to_J(Gamma_meV)

# Precompute DOS map
DOS_map = np.zeros((num_E, num_B))
for i, B in enumerate(B_vals):
    DOS_map[:, i] = dos_B(E_J, B, gamma_J)

# Normalize for plotting clarity
DOS_map_plot = DOS_map / np.max(DOS_map)

# ------------------------------------------------
# Figure layout
# ------------------------------------------------
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.28)
ax_map = fig.add_subplot(gs[0, 0])
ax_cut = fig.add_subplot(gs[0, 1])

plt.subplots_adjust(bottom=0.16)

# ------------------------------------------------
# Left panel: heatmap D(E, B)
# ------------------------------------------------
im = ax_map.imshow(
    DOS_map_plot,
    origin="lower",
    aspect="auto",
    extent=[B_vals[0], B_vals[-1], E_meV[0], E_meV[-1]],
)

ax_map.set_title("Graphene DOS evolution: B = 0 → finite B")
ax_map.set_xlabel("Magnetic field B (T)")
ax_map.set_ylabel("Energy (meV)")

# horizontal guide at E=0
ax_map.axhline(0, color="white", lw=0.8, alpha=0.7)

# vertical line controlled by slider
vline = ax_map.axvline(B_init, color="red", lw=2)

# ------------------------------------------------
# Right panel: DOS slice at selected B
# ------------------------------------------------
current_DOS = dos_B(E_J, B_init, gamma_J)
current_DOS_plot = current_DOS / np.max(DOS_map)

line_B, = ax_cut.plot(current_DOS_plot, E_meV, lw=2, label=f"B = {B_init:.2f} T")
line_B0, = ax_cut.plot(
    dos_B0(E_J) / np.max(DOS_map),
    E_meV,
    "--",
    lw=1.8,
    label="B = 0 analytic DOS"
)

ax_cut.set_title("DOS at selected magnetic field")
ax_cut.set_xlabel("Normalized DOS")
ax_cut.set_ylabel("Energy (meV)")
ax_cut.set_ylim(-E_max_meV, E_max_meV)
ax_cut.set_xlim(0, 1.05 * np.max(DOS_map_plot))
ax_cut.axhline(0, color="black", lw=0.8)
ax_cut.grid(alpha=0.25)
ax_cut.legend(loc="upper right")

info_text = ax_cut.text(
    0.03, 0.97, "",
    transform=ax_cut.transAxes,
    ha="left", va="top",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

# ------------------------------------------------
# Slider
# ------------------------------------------------
ax_slider = fig.add_axes([0.18, 0.06, 0.64, 0.04])
slider_B = Slider(
    ax=ax_slider,
    label="B (Tesla)",
    valmin=0.0,
    valmax=B_max,
    valinit=B_init,
    valstep=B_max / 400
)

# ------------------------------------------------
# Update function
# ------------------------------------------------
def update(val):
    B = slider_B.val
    vline.set_xdata([B, B])

    D = dos_B(E_J, B, gamma_J)
    D_plot = D / np.max(DOS_map)

    line_B.set_xdata(D_plot)
    line_B.set_ydata(E_meV)
    line_B.set_label(f"B = {B:.2f} T")

    # update info
    if B < 1e-8:
        info = (
            f"B = {B:.2f} T\n"
            f"Regime: continuous Dirac DOS\n"
            f"D(E) ∝ |E|"
        )
    else:
        lB_nm = magnetic_length(B) * 1e9
        E1_meV = J_to_meV(graphene_LL_energy_J(1, B))
        info = (
            f"B = {B:.2f} T\n"
            f"l_B = {lB_nm:.2f} nm\n"
            f"E₁ = {E1_meV:.2f} meV\n"
            f"LL spacing ∝ √B"
        )

    info_text.set_text(info)

    # refresh legend
    ax_cut.legend(loc="upper right")
    fig.canvas.draw_idle()

slider_B.on_changed(update)
update(B_init)

plt.show()
