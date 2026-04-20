import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
# Graphene spectrum visualization:
#   1) DOS map D(E,B)
#   2) DOS slice at selected B
#   3) Finite-size eigenvalue sequence E_j(B)
#
# Core idea:
#   - B = 0: continuous Dirac DOS, D(E) ~ |E|
#   - B > 0: broadened Landau levels
#   - finite-size eigenvalues are obtained by inverting the
#     integrated DOS:
#           N(E;B) = j - 1/2
# ============================================================

# -------------------------
# Physical constants (SI)
# -------------------------
hbar = 1.054571817e-34      # J*s
e = 1.602176634e-19         # C
vF = 1.0e6                  # m/s
g = 4                       # spin * valley degeneracy

# -------------------------
# Model / display parameters
# -------------------------
B_max = 12.0                # Tesla
B_init = 2.0
E_max_meV = 250.0           # plotting energy window
Gamma_meV = 3.0             # LL Gaussian broadening
n_max = 120                 # maximum LL index included in DOS sum

# Finite-size effective area:
# choose so that zero-field level spacing is visible but not too large
L_nm = 220.0                # effective sample linear size in nm
A = (L_nm * 1e-9) ** 2      # area in m^2

# Number of discrete eigenvalues to show near zero energy
N_show = 220*4

# Numerical grids
num_E = 2200
num_B = 260

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def meV_to_J(E_meV):
    return E_meV * 1e-3 * e

def J_to_meV(E_J):
    return E_J / e * 1e3

def magnetic_length(B):
    if B <= 0:
        return np.inf
    return np.sqrt(hbar / (e * B))

def graphene_LL_energy_J(n, B):
    if n == 0:
        return 0.0
    return np.sign(n) * vF * np.sqrt(2 * e * hbar * B * abs(n))

def gaussian(x, gamma):
    return np.exp(-0.5 * (x / gamma) ** 2) / (np.sqrt(2 * np.pi) * gamma)

# ------------------------------------------------------------
# Zero-field graphene DOS per unit area:
#   D0(E) = g |E| / [2 pi (hbar vF)^2]
# units: 1 / (J m^2)
# ------------------------------------------------------------
def dos_B0(E_J):
    return g * np.abs(E_J) / (2 * np.pi * (hbar * vF) ** 2)

# ------------------------------------------------------------
# Finite-B DOS per unit area with broadened Landau levels:
#   D(E,B) = [g / (2 pi l_B^2)] * sum_n G(E - E_n, Gamma)
# ------------------------------------------------------------
def dos_B(E_J_grid, B, gamma_J):
    if B <= 1e-12:
        return dos_B0(E_J_grid)

    lB = magnetic_length(B)
    prefactor = g / (2 * np.pi * lB ** 2)  # degeneracy per LL per area

    D = prefactor * gaussian(E_J_grid, gamma_J)  # n = 0

    for n in range(1, n_max + 1):
        En = graphene_LL_energy_J(n, B)
        D += prefactor * gaussian(E_J_grid - En, gamma_J)
        D += prefactor * gaussian(E_J_grid + En, gamma_J)

    return D

# ------------------------------------------------------------
# Integrated DOS for finite area:
#   N(E;B) = A * \int_{-Emax}^{E} D(E',B) dE'
#
# Then define discrete eigenvalues by:
#   N(E_j;B) = j - 1/2
# ------------------------------------------------------------
def cumulative_state_count(E_J_grid, D_per_area, area):
    dE = E_J_grid[1] - E_J_grid[0]
    return area * np.cumsum(D_per_area) * dE

def eigen_sequence_from_dos(E_J_grid, D_per_area, area, N_show):
    """
    Build a finite-size eigenvalue sequence from integrated DOS.
    Return:
        indices (1..N_show), energies_meV
    """
    Ncum = cumulative_state_count(E_J_grid, D_per_area, area)

    # total states inside the energy window
    N_total_window = Ncum[-1]
    if N_total_window < N_show + 1:
        raise RuntimeError(
            "Energy window too small or area too small: "
            "not enough states to build the requested eigenvalue sequence."
        )

    # choose N_show states centered in the available spectrum window
    start = 0.5 * (N_total_window - N_show)
    targets = start + np.arange(N_show) + 0.5

    # invert cumulative function by interpolation
    E_targets_J = np.interp(targets, Ncum, E_J_grid)
    return np.arange(1, N_show + 1), J_to_meV(E_targets_J)

# ------------------------------------------------------------
# Build grids
# ------------------------------------------------------------
E_meV = np.linspace(-E_max_meV, E_max_meV, num_E)
E_J = meV_to_J(E_meV)
gamma_J = meV_to_J(Gamma_meV)

B_vals = np.linspace(0.0, B_max, num_B)

# Precompute DOS map
DOS_map = np.zeros((num_E, num_B))
for i, B in enumerate(B_vals):
    DOS_map[:, i] = dos_B(E_J, B, gamma_J)

DOS_map_norm = DOS_map / np.max(DOS_map)

# ------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------
fig = plt.figure(figsize=(15, 7.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.9, 1.1], wspace=0.30)
ax_map = fig.add_subplot(gs[0, 0])
ax_dos = fig.add_subplot(gs[0, 1])
ax_seq = fig.add_subplot(gs[0, 2])

plt.subplots_adjust(bottom=0.16)

# ------------------------------------------------------------
# Panel 1: DOS heatmap
# ------------------------------------------------------------
im = ax_map.imshow(
    DOS_map_norm,
    origin="lower",
    aspect="auto",
    extent=[B_vals[0], B_vals[-1], E_meV[0], E_meV[-1]],
)

ax_map.set_title("Graphene DOS evolution")
ax_map.set_xlabel("Magnetic field B (T)")
ax_map.set_ylabel("Energy (meV)")
ax_map.axhline(0, color="white", lw=0.8, alpha=0.7)
vline = ax_map.axvline(B_init, color="red", lw=2)

# ------------------------------------------------------------
# Panel 2: DOS slice
# ------------------------------------------------------------
D_init = dos_B(E_J, B_init, gamma_J)
line_dos, = ax_dos.plot(D_init / np.max(DOS_map), E_meV, lw=2, label=f"B = {B_init:.2f} T")
line_dos0, = ax_dos.plot(
    dos_B0(E_J) / np.max(DOS_map),
    E_meV, "--", lw=1.6, label="B = 0"
)

ax_dos.set_title("DOS at selected B")
ax_dos.set_xlabel("Normalized DOS")
ax_dos.set_ylabel("Energy (meV)")
ax_dos.set_ylim(-E_max_meV, E_max_meV)
ax_dos.axhline(0, color="black", lw=0.8)
ax_dos.grid(alpha=0.25)
ax_dos.legend(loc="upper right")

info_text = ax_dos.text(
    0.03, 0.97, "",
    transform=ax_dos.transAxes,
    ha="left", va="top",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

# ------------------------------------------------------------
# Panel 3: finite-size eigenvalue sequence
# ------------------------------------------------------------
idx_init, eig_init = eigen_sequence_from_dos(E_J, D_init, A, N_show)
sc_seq = ax_seq.scatter(idx_init, eig_init, s=2, c='k')

ax_seq.set_title("Finite-size eigenvalue sequence")
ax_seq.set_xlabel("Eigenvalue index")
ax_seq.set_ylabel("Energy (meV)")
ax_seq.set_ylim(-E_max_meV, E_max_meV)
ax_seq.grid(alpha=0.25)

seq_text = ax_seq.text(
    0.03, 0.97, "",
    transform=ax_seq.transAxes,
    ha="left", va="top",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

# ------------------------------------------------------------
# Slider
# ------------------------------------------------------------
ax_slider = fig.add_axes([0.18, 0.06, 0.64, 0.04])
slider_B = Slider(
    ax=ax_slider,
    label="B (Tesla)",
    valmin=0.0,
    valmax=B_max,
    valinit=B_init,
    valstep=B_max / 500
)

# ------------------------------------------------------------
# Update
# ------------------------------------------------------------
def update(val):
    B = slider_B.val
    vline.set_xdata([B, B])

    # DOS slice
    D = dos_B(E_J, B, gamma_J)
    D_plot = D / np.max(DOS_map)
    line_dos.set_xdata(D_plot)
    line_dos.set_ydata(E_meV)
    line_dos.set_label(f"B = {B:.2f} T")
    ax_dos.legend(loc="upper right")

    # eigenvalue sequence
    idx, eig_meV = eigen_sequence_from_dos(E_J, D, A, N_show)
    offsets = np.column_stack([idx, eig_meV])
    sc_seq.set_offsets(offsets)

    # info texts
    if B < 1e-10:
        info = (
            f"B = {B:.2f} T\n"
            f"Regime: continuous Dirac DOS\n"
            f"D(E) ∝ |E|"
        )
        seq_info = (
            f"Finite-size sample:\n"
            f"L = {L_nm:.0f} nm\n"
            f"Sequence from quantized IDOS\n"
            f"(zero-field Dirac box-like spectrum)"
        )
    else:
        lB_nm = magnetic_length(B) * 1e9
        E1_meV = J_to_meV(graphene_LL_energy_J(1, B))
        degeneracy_per_LL = g * A * e * B / (2 * np.pi * hbar)

        info = (
            f"B = {B:.2f} T\n"
            f"l_B = {lB_nm:.2f} nm\n"
            f"E₁ = {E1_meV:.2f} meV\n"
            f"LL spacing ∝ √B"
        )
        seq_info = (
            f"Finite-size sample:\n"
            f"L = {L_nm:.0f} nm\n"
            f"Approx. LL degeneracy\n"
            f"inside sample ≈ {degeneracy_per_LL:.1f}"
        )

    info_text.set_text(info)
    seq_text.set_text(seq_info)

    fig.canvas.draw_idle()

slider_B.on_changed(update)
update(B_init)

plt.show()
