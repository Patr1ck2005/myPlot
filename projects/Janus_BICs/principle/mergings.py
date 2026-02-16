import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

# ============================================================
# Utilities: Stokes, ellipse parameters, toy-Q
# ============================================================
def stokes_from_jones(Ex, Ey, eps=1e-12):
    """
    Normalized Stokes from Jones.
      S0 = |Ex|^2 + |Ey|^2
      S1 = (|Ex|^2 - |Ey|^2)/S0
      S2 = 2 Re(Ex Ey*)/S0
      S3 = 2 Im(Ex Ey*)/S0
    """
    S0 = (np.abs(Ex)**2 + np.abs(Ey)**2) + eps
    S1 = (np.abs(Ex)**2 - np.abs(Ey)**2) / S0
    S2 = (2*np.real(Ex*np.conj(Ey))) / S0
    S3 = (2*np.imag(Ex*np.conj(Ey))) / S0
    return S0, S1, S2, S3

def ellipse_params_from_stokes(S1, S2, S3):
    """
    alpha: ellipse orientation
    chi: ellipticity angle in [-pi/4, pi/4]
    b/a = |tan(chi)|
    """
    alpha = 0.5 * np.arctan2(S2, S1)
    chi = 0.5 * np.arcsin(np.clip(S3, -1.0, 1.0))
    return alpha, chi

def toy_Q_from_radiation(Ex, Ey, Q0=1e6, eps=1e-12, power=1.0):
    """
    Toy Q proxy: Q ~ 1/|c|^2.
    """
    rad = (np.abs(Ex)**2 + np.abs(Ey)**2)
    return Q0 / np.power(rad + eps, power)

# ============================================================
# Mirror symmetry convention (polar vector):
#   σx: (kx,ky)->(-kx,ky), (Ex,Ey)->(-Ex, Ey)
#   σy: (kx,ky)->(kx,-ky), (Ex,Ey)->( Ex,-Ey)
#
# Sufficient parity structure:
#   Ex odd in kx, even in ky
#   Ey even in kx, odd in ky
# ============================================================

# ============================================================
# Model A (process-1 analogue): S3=0 real field, σx & σy symmetric
# ============================================================
def model_A_field(KX, KY, a=0.8, b=0.6):
    Ex = (KX * (a*a - KX*KX - b*KY*KY)).astype(np.complex128)
    Ey = (KY).astype(np.complex128)
    return Ex, Ey

# ============================================================
# Model B (process-2 analogue): CONTINUOUS flip from equator to meridian
#
# Parameter λ in [0,1]:
#   phi(λ) = λ * (pi/2)
# Field:
#   Ex = kx
#   Ey = exp(i * eta * phi(λ)) * ky
#
# λ=0  -> Ey real -> S3=0 (equator)
# λ=1  -> Ey = i*eta*ky -> S2=0 meridian passing poles
# ============================================================
def model_B_field(KX, KY, lam=0.0, eta=+1, phi_max=np.pi, alpha=1.0, beta=1.0, gamma=1.0):
    phi = lam * phi_max
    Ex = (alpha * KX).astype(np.complex128)
    Ey = ((beta * np.exp(1j * eta * phi) + gamma*(KX**2+KY**2)) * KY).astype(np.complex128)
    return Ex, Ey

# ============================================================
# Visualization: S3 heatmap + TRUE ellipse glyphs
# ============================================================
def plot_S3_with_ellipses(
    title,
    KX, KY, Ex, Ey,
    extent,
    step=24,
    glyph_a=0.14,        # major axis length in k-units
    amp_cut=1e-3,        # skip ellipses where amplitude too small (near BIC)
    max_ratio=0.98       # cap b/a for visibility
):
    S0, S1, S2, S3 = stokes_from_jones(Ex, Ey)
    alpha, chi = ellipse_params_from_stokes(S1, S2, S3)

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    im = ax.imshow(
        S3,
        extent=extent,
        origin="lower",
        aspect="equal",
        vmin=-1, vmax=1
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("S3")

    xs = KX[::step, ::step]
    ys = KY[::step, ::step]
    a = alpha[::step, ::step]
    c = chi[::step, ::step]
    amp = np.sqrt(S0[::step, ::step])

    ratio = np.abs(np.tan(c))              # b/a
    ratio = np.clip(ratio, 0.0, max_ratio)

    patches = []
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            if amp[i, j] < amp_cut:
                continue
            width = glyph_a
            height = glyph_a * ratio[i, j] + 1e-6
            angle_deg = np.degrees(a[i, j])
            patches.append(Ellipse((xs[i, j], ys[i, j]),
                                   width=width, height=height,
                                   angle=angle_deg, fill=False, linewidth=0.9))

    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.set_title(title)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    plt.tight_layout()

# ============================================================
# Q heatmap plotter
# ============================================================
def plot_Q_heatmap(title, Q, extent, log=True):
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    if log:
        Z = np.log10(Q)
        im = ax.imshow(Z, extent=extent, origin="lower", aspect="equal")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log10(Q) (toy)")
    else:
        im = ax.imshow(Q, extent=extent, origin="lower", aspect="equal")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Q (toy)")

    ax.set_title(title)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    plt.tight_layout()

# ============================================================
# Poincaré-sphere trajectory for a loop around Γ
# ============================================================
def plot_poincare_trajectory(title, Ex_loop, Ey_loop):
    _, S1, S2, S3 = stokes_from_jones(Ex_loop, Ey_loop)
    r = np.sqrt(S1*S1 + S2*S2 + S3*S3) + 1e-15
    X, Y, Z = S1/r, S2/r, S3/r

    fig = plt.figure(figsize=(7.2, 6.7))
    ax = fig.add_subplot(111, projection="3d")

    # wireframe sphere
    u = np.linspace(0, 2*np.pi, 140)
    v = np.linspace(0, np.pi, 70)
    Xs = np.outer(np.cos(u), np.sin(v))
    Ys = np.outer(np.sin(u), np.sin(v))
    Zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(Xs, Ys, Zs, rstride=10, cstride=8, linewidth=0.6, alpha=0.22)

    ax.plot(X, Y, Z, linewidth=2.4)
    ax.scatter([X[0]], [Y[0]], [Z[0]], s=40)

    ax.set_title(title)
    ax.set_xlabel("S1")
    ax.set_ylabel("S2")
    ax.set_zlabel("S3")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.view_init(elev=18, azim=35)
    plt.tight_layout()

# ============================================================
# Demo (run locally)
# ============================================================
def main():
    L = 1.6
    N = 601
    k = np.linspace(-L, L, N)
    KX, KY = np.meshgrid(k, k, indexing="xy")
    extent = [-L, L, -L, L]

    # loop around Γ
    t = np.linspace(0, 2*np.pi, 2400, endpoint=False)
    r0 = 1.0
    kx_loop = r0 * np.cos(t)
    ky_loop = r0 * np.sin(t)

    # -------------------------
    # Model A: one snapshot
    # -------------------------
    a = 0.8
    b = 0.6
    ExA, EyA = model_A_field(KX, KY, a=a, b=b)

    QA = toy_Q_from_radiation(ExA, EyA, Q0=1e6, power=1.0)
    plot_Q_heatmap(f"Model A: toy Q heatmap (log10), a={a}, b={b}", QA, extent, log=True)

    plot_S3_with_ellipses(
        f"Model A: S3 heatmap + ellipses (should be ~0), a={a}, b={b}",
        KX, KY, ExA, EyA, extent,
        step=26, glyph_a=0.14, amp_cut=1e-2
    )

    ExA_loop, EyA_loop = model_A_field(kx_loop, ky_loop, a=a, b=b)
    plot_poincare_trajectory(f"Model A: Poincaré trajectory (loop r={r0})", ExA_loop, EyA_loop)

    # -------------------------
    # Model B: multiple λ snapshots (continuous flip)
    # -------------------------
    eta = +1
    lam_list = [0.0, 0.25, 0.5, 0.75, 1.0]  # choose any
    for lam in lam_list:
        ExB, EyB = model_B_field(KX, KY, lam=lam, eta=eta, phi_max=np.pi, alpha=1.0, beta=1.0)

        QB = toy_Q_from_radiation(ExB, EyB, Q0=1e6, power=1.0)
        plot_Q_heatmap(f"Model B: toy Q heatmap (log10), λ={lam:.2f}, η={eta}", QB, extent, log=True)

        plot_S3_with_ellipses(
            f"Model B: S3 heatmap + ellipses, λ={lam:.2f}, η={eta}",
            KX, KY, ExB, EyB, extent,
            step=26, glyph_a=0.14, amp_cut=1e-2
        )

        ExB_loop, EyB_loop = model_B_field(kx_loop, ky_loop, lam=lam, eta=eta, phi_max=np.pi, alpha=1.0, beta=1.0)
        plot_poincare_trajectory(f"Model B: Poincaré trajectory (λ={lam:.2f}, η={eta}, loop r={r0})", ExB_loop, EyB_loop)

    plt.show()

if __name__ == "__main__":
    main()
