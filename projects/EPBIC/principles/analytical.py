import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, RadioButtons, Button

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Arial'

# -----------------------------
# Fast vectorized computation
# -----------------------------
def compute_maps(S, omega1, omega2, gamma_a, gamma_r,
                 kmin, kmax, w3min, w3max,
                 Nx=240, Ny=240, dtype=np.complex64):
    """
    Compute dmax and dmin heatmaps over (kappa, omega3) grid.
    Uses batched eigvals for speed.
    """
    # Grid
    k = np.linspace(kmin, kmax, Nx, dtype=np.float32)
    w3 = np.linspace(w3min, w3max, Ny, dtype=np.float32)
    K, W3 = np.meshgrid(k, w3)  # shape (Ny, Nx)

    # Couplings
    s = np.float32(np.sin(2.0 * np.pi * S))
    c = np.float32(np.cos(2.0 * np.pi * S))
    g1 = K * s
    g2 = K * c

    # Diagonals
    d1 = np.complex64(omega1) - 1j * np.complex64(gamma_a + gamma_r)
    d2 = np.complex64(omega2) - 1j * np.complex64(gamma_a)
    d3 = W3.astype(np.float32).astype(np.complex64)

    # Build batched matrices H[y,x,:,:]
    H = np.zeros((Ny, Nx, 3, 3), dtype=dtype)
    H[..., 0, 0] = d1
    H[..., 1, 1] = d2
    H[..., 2, 2] = d3
    H[..., 0, 2] = g1.astype(np.float32).astype(dtype)
    H[..., 2, 0] = H[..., 0, 2]
    H[..., 1, 2] = g2.astype(np.float32).astype(dtype)
    H[..., 2, 1] = H[..., 1, 2]

    # Batched eigenvalues: shape (Ny, Nx, 3)
    lam = np.linalg.eigvals(H)

    # Pairwise distances
    d12 = np.abs(lam[..., 0] - lam[..., 1])
    d13 = np.abs(lam[..., 0] - lam[..., 2])
    d23 = np.abs(lam[..., 1] - lam[..., 2])

    dmax = np.maximum(np.maximum(d12, d13), d23).astype(np.float32)
    dmin = np.minimum(np.minimum(d12, d13), d23).astype(np.float32)

    return k, w3, dmax, dmin


# -----------------------------
# Matplotlib UI
# -----------------------------
def main():
    # ---- Initial params (match your typical values) ----
    S0 = 0.15
    omega1_0 = -0.2
    omega2_0 = 0.2
    gamma_a0 = 0.5
    gamma_r0 = 0.0

    k_range0 = (0.0, 1)
    w3_range0 = (-1, 1)

    # Grid resolution (trade speed vs detail)
    Nx = 260
    Ny = 260

    # Compute initial maps
    k, w3, dmax, dmin = compute_maps(
        S0, omega1_0, omega2_0, gamma_a0, gamma_r0,
        k_range0[0], k_range0[1], w3_range0[0], w3_range0[1],
        Nx=Nx, Ny=Ny
    )

    extent = [k.min(), k.max(), w3.min(), w3.max()]

    # # 临时分开保存图片 ===============================================================================================
    # fig, ax = plt.subplots(figsize=(1, 1))
    # im = ax.imshow(dmax, origin="lower", extent=extent, aspect="auto", vmin=0, cmap="magma", interpolation='none')
    # # ax.set_title(r"Heatmap: $\max_{i<j}|\lambda_i-\lambda_j|$")
    # ax.set_xlabel(r"$\kappa$")
    # ax.set_ylabel(r"$\omega_3$")
    # # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    # plt.savefig("dmax_initial.svg", dpi=300, bbox_inches='tight', transparent=True)
    # plt.close(fig)
    # fig, ax = plt.subplots(figsize=(1, 1))
    # im = ax.imshow(dmin, origin="lower", extent=extent, aspect="auto", vmin=0, cmap="magma", interpolation='none')
    # # ax.set_title(r"Heatmap: $\min_{i<j}|\lambda_i-\lambda_j|$")
    # ax.set_xlabel(r"$\kappa$")
    # ax.set_ylabel(r"$\omega_3$")
    # # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    # plt.savefig("dmin_initial.svg", dpi=300, bbox_inches='tight', transparent=True)
    # plt.close(fig)
    # ================================================================================================================

    # ---- Figure layout ----
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[12, 1], width_ratios=[1, 1, 0.55])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_ctrl = fig.add_subplot(gs[0, 2])
    ax_ctrl.axis("off")

    # Two images (for side-by-side mode)
    im1 = ax1.imshow(dmax, origin="lower", extent=extent, aspect="auto", vmin=0, cmap="magma", interpolation='none')
    im2 = ax2.imshow(dmin, origin="lower", extent=extent, aspect="auto", vmin=0, cmap="magma", interpolation='none')

    ax1.set_title(r"Heatmap: $\max_{i<j}|\lambda_i-\lambda_j|$")
    ax2.set_title(r"Heatmap: $\min_{i<j}|\lambda_i-\lambda_j|$")
    ax1.set_xlabel(r"$\kappa$")
    ax2.set_xlabel(r"$\kappa$")
    ax1.set_ylabel(r"$\omega_3$")
    ax2.set_ylabel(r"$\omega_3$")

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)

    # ---- Controls area (bottom row sliders) ----
    # # Slider axes
    # axS      = fig.add_subplot(gs[1, 0])
    # axO1     = fig.add_subplot(gs[1, 1])

    # Create an inset axes area for more sliders (stacked)
    # We'll place them manually in figure coords for compactness.
    # (This keeps it readable without over-complicating gridspec.)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.2, top=0.92, wspace=0.4)

    # Manual slider positions (x, y, w, h) in figure fraction
    axS_      = fig.add_axes([0.06, 0.05, 0.20, 0.03])
    axO1_     = fig.add_axes([0.06, 0.01, 0.20, 0.03])
    axO2_     = fig.add_axes([0.06, 0.09, 0.20, 0.03])
    axGa_     = fig.add_axes([0.06, 0.13, 0.20, 0.03])
    axGr_     = fig.add_axes([0.06, 0.17, 0.20, 0.03])
    axKR_     = fig.add_axes([0.70, 0.01, 0.20, 0.03])  # kappa range
    axWR_     = fig.add_axes([0.70, 0.05, 0.20, 0.03])  # omega3 range (above)

    # Sliders
    sS  = Slider(axS_,  "S",        0.0, 0.5, valinit=S0, valstep=0.01)
    sO1 = Slider(axO1_, "ω1",      -1.0, 1.0, valinit=omega1_0, valstep=0.01)
    sO2 = Slider(axO2_, "ω2",      -1.0, 1.0, valinit=omega2_0, valstep=0.01)
    sGa = Slider(axGa_, "γa",       0.0, 1.0, valinit=gamma_a0, valstep=0.01)
    sGr = Slider(axGr_, "γr",       0.0, 1.0, valinit=gamma_r0, valstep=0.01)

    rW3 = RangeSlider(axWR_, "ω3 range", -2.0, 2.0, valinit=w3_range0, valstep=0.01)
    rK  = RangeSlider(axKR_, "κ range",   0.0, 2.0, valinit=k_range0,  valstep=0.01)

    # Radio buttons for view mode
    axRadio = fig.add_axes([0.80, 0.33, 0.18, 0.22])
    radio = RadioButtons(axRadio, ("two", "max only", "min only"), active=0)

    # Reset button
    axBtn = fig.add_axes([0.82, 0.26, 0.14, 0.05])
    btn = Button(axBtn, "Reset")

    # For speed: cache current arrays; recompute only when needed.
    state = {"last": None}

    def update(_=None):
        S  = float(sS.val)
        o1 = float(sO1.val)
        o2 = float(sO2.val)
        ga = float(sGa.val)
        gr = float(sGr.val)
        kmin, kmax = map(float, rK.val)
        w3min, w3max = map(float, rW3.val)

        # Avoid degenerate ranges
        if kmax <= kmin + 1e-12:
            kmax = kmin + 1e-3
        if w3max <= w3min + 1e-12:
            w3max = w3min + 1e-3

        # Compute
        k, w3, dmax_new, dmin_new = compute_maps(
            S, o1, o2, ga, gr, kmin, kmax, w3min, w3max, Nx=Nx, Ny=Ny
        )
        ext = [k.min(), k.max(), w3.min(), w3.max()]

        # Update images
        im1.set_data(dmax_new)
        im2.set_data(dmin_new)
        im1.set_extent(ext)
        im2.set_extent(ext)

        # Update axes limits
        ax1.set_xlim(ext[0], ext[1]); ax1.set_ylim(ext[2], ext[3])
        ax2.set_xlim(ext[0], ext[1]); ax2.set_ylim(ext[2], ext[3])

        # # Keep colorbars responsive (optional: autoscale each time)
        # im1.set_clim(np.min(dmax_new), np.max(dmax_new))
        # im2.set_clim(np.min(dmin_new), np.max(dmin_new))

        # View mode
        mode = radio.value_selected
        if mode == "two":
            ax1.set_visible(True); ax2.set_visible(True)
            cbar1.ax.set_visible(True); cbar2.ax.set_visible(True)
        elif mode == "max only":
            ax1.set_visible(True); ax2.set_visible(False)
            cbar1.ax.set_visible(True); cbar2.ax.set_visible(False)
        else:  # "min only"
            ax1.set_visible(False); ax2.set_visible(True)
            cbar1.ax.set_visible(False); cbar2.ax.set_visible(True)

        fig.canvas.draw_idle()

    def on_reset(event):
        sS.reset()
        sO1.reset()
        sO2.reset()
        sGa.reset()
        sGr.reset()
        rK.set_val(k_range0)
        rW3.set_val(w3_range0)
        radio.set_active(0)
        update()

    # Hook events (fast enough for typical use; if you want even faster, update on release only)
    for w in (sS, sO1, sO2, sGa, sGr, rK, rW3):
        w.on_changed(update)
    radio.on_clicked(update)
    btn.on_clicked(on_reset)

    plt.show()


if __name__ == "__main__":
    main()
