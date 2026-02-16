"""
Revised, standalone Python script (English plots) for TWO topological quantities
on the TWO toy polarization-field models defined earlier.

Quantities (NO skyrmion / NO area integral):
  (I)  Pancharatnam–Berry (PB) geometric phase on a closed k-space loop C:
       gamma(C) = sum_n Arg <e_n | e_{n+1}>   (mod 2π)
       where |e_n> is the normalized Jones vector on the loop.
       -> robust Wilson-loop (Bargmann invariant) implementation via phase SUM (NOT product).

  (II) 3D degree / point-defect charge in extended parameter space (kx, ky, λ):
       Q = (1/2π^2) ∫_V ε_abcd n_a (∂x n_b)(∂y n_c)(∂λ n_d) d^3x
       with n = F/||F|| ∈ S^3, F ∈ R^4 the real embedding of the complex Jones vector:
         F = (Re d1, Im d1, Re d2, Im d2).
       This is the π3(S^3)=Z “most universal” integer charge.

IMPORTANT modeling note (practical necessity):
  In the raw toy models, the BIC zero at k=0 exists for ALL parameter values (a-axis or t-axis line of zeros),
  so the 3D degree around (kx,ky,λ0) is not well-defined (not an isolated point defect).
  To compute Q in a meaningful and stable way, we introduce a tiny "lifting" term for the 3D calculation only:
     d -> d + i * eta * (λ-λ0) * u
  which makes the zero at k=0 isolated at λ=λ0 inside the 3D volume.
  This mimics the generic situation where an additional detuning/symmetry-breaking direction isolates the defect.

You can set eta=0 to see it become ill-defined/noisy (not recommended).

Dependencies:
  numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1) Toy Jones-field models (same functional forms as before)
# ============================================================

def jones_model_A(kx, ky, a=0.6):
    """
    Model A (merging-type, S3=0):
      f = z*(conj(z)^2 - a^2),  d=(Re f, Im f), purely real components.
    """
    z = kx + 1j * ky
    zb = np.conj(z)
    f = z * (zb**2 - a**2)
    d1 = np.real(f) + 0j
    d2 = np.imag(f) + 0j
    return d1, d2


def jones_model_B(kx, ky, t=0.0):
    """
    Model B (SU(2) flip, S3 != 0 in general):
      d = r * U(t) * (cos φ, sin φ)^T,   U(t)=exp(-i (π/2) t σx).
    """
    r = np.sqrt(kx**2 + ky**2)
    phi = np.arctan2(ky, kx)

    e0x = np.cos(phi)
    e0y = np.sin(phi)

    theta = 0.5 * np.pi * t
    c = np.cos(theta)
    s = np.sin(theta)

    d1 = r * (c * e0x - 1j * s * e0y)
    d2 = r * (c * e0y - 1j * s * e0x)
    return d1, d2


# ============================================================
# 2) Common utilities
# ============================================================

def normalize_jones(d1, d2, eps=1e-12):
    """Normalize Jones vectors: e = d / ||d||."""
    norm = np.sqrt(np.abs(d1)**2 + np.abs(d2)**2)
    norm = np.maximum(norm, eps)
    return d1 / norm, d2 / norm, norm


def circle_loop(R=0.8, npts=1200, center=(0.0, 0.0)):
    """Sample a closed circle in k-space (endpoint excluded)."""
    cx, cy = center
    ang = np.linspace(0.0, 2.0 * np.pi, npts, endpoint=False)
    kx = cx + R * np.cos(ang)
    ky = cy + R * np.sin(ang)
    return kx, ky, ang


# ============================================================
# 3) (I) Pancharatnam–Berry phase on a loop: stable phase SUM
# ============================================================

def pb_phase_on_loop(d1, d2, eps_norm=1e-12, tol_overlap=1e-8):
    """
    Compute PB phase gamma(C) using:
      gamma = sum_n Arg <e_n|e_{n+1}>  (wrapped to (-π,π])

    Returns:
      gamma (float), ok (bool), min_overlap (float)
    """
    e1, e2, _ = normalize_jones(d1, d2, eps=eps_norm)

    overlap = np.conj(e1) * np.roll(e1, -1) + np.conj(e2) * np.roll(e2, -1)
    abs_ov = np.abs(overlap)
    min_overlap = float(np.min(abs_ov))

    ok = (min_overlap > tol_overlap)
    # If not ok, still return a value, but caller should mask it.
    phases = np.angle(overlap + 0j)
    gamma = float(np.sum(phases))

    # Wrap to (-pi, pi]
    gamma = (gamma + np.pi) % (2.0 * np.pi) - np.pi
    return gamma, ok, min_overlap


# ============================================================
# 4) (II) 3D degree Q in (kx, ky, λ) using S^3 map of F ∈ R^4
# ============================================================

def jones_with_lift(model, kx, ky, lam, lam0, eta=0.15, u=(1.0+0j, 1.0+0j)):
    """
    Return Jones vector d(kx,ky; lam) with an *imaginary* lift term:
      d -> d + i * eta*(lam-lam0) * u
    This isolates the defect at (kx,ky,lam)=(0,0,lam0) for Q computation.

    model: "A" uses parameter a=lam, "B" uses parameter t=lam.
    """
    if model.upper() == "A":
        d1, d2 = jones_model_A(kx, ky, a=lam)
    elif model.upper() == "B":
        d1, d2 = jones_model_B(kx, ky, t=lam)
    else:
        raise ValueError("model must be 'A' or 'B'")

    u1, u2 = u
    lift = 1j * eta * (lam - lam0)
    d1 = d1 + lift * u1
    d2 = d2 + lift * u2
    return d1, d2


def complex_jones_to_R4(d1, d2):
    """
    Embed complex 2-vector (d1,d2) into R^4:
      F = (Re d1, Im d1, Re d2, Im d2)
    """
    F0 = np.real(d1)
    F1 = np.imag(d1)
    F2 = np.real(d2)
    F3 = np.imag(d2)
    return np.stack([F0, F1, F2, F3], axis=-1)  # (...,4)


def degree_Q_3d(model="A", lam0=0.6, kbox=0.35, lbox=0.20, Nk=41, Nl=33,
                eta=0.15, eps_norm=1e-12, core_cut=5e-4):
    """
    Compute Q in a 3D box around (kx,ky,lam)=(0,0,lam0).
      kx,ky ∈ [-kbox, kbox], lam ∈ [lam0-lbox, lam0+lbox]

    Discretization:
      Q = (1/2π^2) ∫ det([n, ∂x n, ∂y n, ∂λ n]) d^3x
      where n ∈ S^3.

    core_cut excludes points where ||F|| is too small (near the defect core).
    """
    kx = np.linspace(-kbox, kbox, Nk)
    ky = np.linspace(-kbox, kbox, Nk)
    lam = np.linspace(lam0 - lbox, lam0 + lbox, Nl)

    dx = float(kx[1] - kx[0])
    dy = float(ky[1] - ky[0])
    dl = float(lam[1] - lam[0])

    KX, KY, LAM = np.meshgrid(kx, ky, lam, indexing="ij")  # (Nk,Nk,Nl)

    d1, d2 = jones_with_lift(model, KX, KY, LAM, lam0=lam0, eta=eta)
    F = complex_jones_to_R4(d1, d2)  # (Nk,Nk,Nl,4)

    Fn = np.linalg.norm(F, axis=-1)
    mask_core = Fn < core_cut
    Fn = np.maximum(Fn, eps_norm)
    n = F / Fn[..., None]  # normalized S^3 field, (Nk,Nk,Nl,4)

    # Finite differences (central-ish via np.gradient)
    # gradients return arrays with same shape
    dn_dx = np.gradient(n, dx, axis=0, edge_order=2)
    dn_dy = np.gradient(n, dy, axis=1, edge_order=2)
    dn_dl = np.gradient(n, dl, axis=2, edge_order=2)

    # determinant det([n, dn_dx, dn_dy, dn_dl]) in R^4 at each grid point
    # Build a 4x4 matrix at each point: columns = these 4 vectors.
    # shape: (Nk,Nk,Nl,4,4)
    M = np.stack([n, dn_dx, dn_dy, dn_dl], axis=-1)

    # det over last two axes:
    detM = np.linalg.det(M)  # (Nk,Nk,Nl)

    # Exclude defect core
    detM = np.where(mask_core, 0.0, detM)

    integral = float(np.sum(detM) * dx * dy * dl)
    Q = integral / (2.0 * np.pi**2)

    return Q, {
        "dx": dx, "dy": dy, "dl": dl,
        "min_Fnorm": float(np.min(Fn)),
        "core_fraction": float(np.mean(mask_core)),
        "kbox": kbox, "lbox": lbox, "Nk": Nk, "Nl": Nl,
        "eta": eta, "lam0": lam0
    }


# ============================================================
# 5) Visualization: PB phase vs parameter / heatmap; Q vs parameter
# ============================================================

def pb_vs_parameter(model="A", params=None, R=0.85, npts=1600,
                    tol_overlap=1e-8, eps_norm=1e-12, title_suffix=""):
    """
    Plot PB phase/(2π) vs the model parameter.
    Also plot min |<e_n|e_{n+1}>| diagnostic to show invalid regions.
    """
    if params is None:
        params = np.linspace(0.0, 1.2, 121) if model.upper() == "A" else np.linspace(0.0, 1.0, 101)

    gam = np.zeros_like(params, dtype=float)
    ok = np.zeros_like(params, dtype=bool)
    minov = np.zeros_like(params, dtype=float)

    kx, ky, _ = circle_loop(R=R, npts=npts)

    for i, p in enumerate(params):
        if model.upper() == "A":
            d1, d2 = jones_model_A(kx, ky, a=float(p))
        else:
            d1, d2 = jones_model_B(kx, ky, t=float(p))

        g, good, m = pb_phase_on_loop(d1, d2, eps_norm=eps_norm, tol_overlap=tol_overlap)
        gam[i] = g / (2.0 * np.pi)
        ok[i] = good
        minov[i] = m

    # Plot PB phase
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(params, gam, label="PB phase / (2π)")
    if np.any(~ok):
        ax.scatter(params[~ok], gam[~ok], marker="x", label="invalid (min overlap too small)")
    ax.set_xlabel("a" if model.upper() == "A" else "t")
    ax.set_ylabel("Value")
    ax.set_title(f"PB phase on loop (Model {model.upper()}, R={R}){title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Diagnostic plot
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.plot(params, minov)
    ax.axhline(tol_overlap, linestyle="--")
    ax.set_xlabel("a" if model.upper() == "A" else "t")
    ax.set_ylabel("min |<e_n|e_{n+1}>|")
    ax.set_title("Loop step-overlap diagnostic (smaller => PB phase unreliable)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def pb_heatmap_param_radius(model="A", params=None, radii=None, npts=1200,
                            tol_overlap=1e-8, eps_norm=1e-12):
    """
    Heatmap of PB phase/(2π) over (parameter, loop radius).
    Invalid regions (min overlap below tolerance) are masked as NaN.
    """
    if params is None:
        params = np.linspace(0.0, 1.2, 121) if model.upper() == "A" else np.linspace(0.0, 1.0, 101)
    if radii is None:
        radii = np.linspace(0.15, 1.2, 70)

    G = np.full((len(radii), len(params)), np.nan, dtype=float)

    for iR, R in enumerate(radii):
        kx, ky, _ = circle_loop(R=float(R), npts=npts)
        for ip, p in enumerate(params):
            if model.upper() == "A":
                d1, d2 = jones_model_A(kx, ky, a=float(p))
            else:
                d1, d2 = jones_model_B(kx, ky, t=float(p))
            g, good, _ = pb_phase_on_loop(d1, d2, eps_norm=eps_norm, tol_overlap=tol_overlap)
            if good:
                G[iR, ip] = g / (2.0 * np.pi)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    im = ax.imshow(
        G,
        origin="lower",
        aspect="auto",
        extent=[params[0], params[-1], radii[0], radii[-1]],
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("PB phase / (2π)")
    ax.set_xlabel("a" if model.upper() == "A" else "t")
    ax.set_ylabel("Loop radius R")
    ax.set_title(f"PB phase heatmap (masked when invalid) — Model {model.upper()}")
    plt.tight_layout()
    plt.show()


def Q_vs_parameter(model="A", params=None, kbox=0.35, lbox=0.20, Nk=41, Nl=33,
                   eta=0.15, core_cut=5e-4):
    """
    Plot 3D degree Q around (kx,ky,λ)=(0,0,λ0) for a list of λ0 centers.
    This uses the lifted field so the defect becomes isolated at each λ0.

    Interpretation:
      Q should be near an integer if:
        - the defect is isolated by the lift (eta > 0),
        - the chosen 3D box encloses only that defect,
        - discretization is sufficient.
    """
    if params is None:
        params = np.linspace(0.1, 1.1, 31) if model.upper() == "A" else np.linspace(0.05, 0.95, 31)

    Qs = []
    core_fracs = []

    for lam0 in params:
        Q, info = degree_Q_3d(
            model=model,
            lam0=float(lam0),
            kbox=kbox,
            lbox=lbox,
            Nk=Nk,
            Nl=Nl,
            eta=eta,
            core_cut=core_cut
        )
        Qs.append(Q)
        core_fracs.append(info["core_fraction"])

    Qs = np.array(Qs)
    core_fracs = np.array(core_fracs)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(params, Qs, marker="o", markersize=3, linewidth=1.2, label="Q (3D degree)")
    ax.set_xlabel("λ0 (center parameter): a0" if model.upper() == "A" else "t0")
    ax.set_ylabel("Q")
    ax.set_title(f"3D degree Q in (kx,ky,λ) — Model {model.upper()} (lifted, η={eta})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.plot(params, core_fracs)
    ax.set_xlabel("λ0 (center parameter): a0" if model.upper() == "A" else "t0")
    ax.set_ylabel("core fraction (||F|| < core_cut)")
    ax.set_title("Defect-core diagnostic (should be small but nonzero)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 6) Demo run
# ============================================================

if __name__ == "__main__":
    # ----------------------------
    # A) PB phase visualizations
    # ----------------------------
    pb_vs_parameter(
        model="A",
        params=np.linspace(0.0, 1.2, 121),
        R=0.85,
        npts=2000,
        tol_overlap=1e-8,
        title_suffix="  (phase-sum Wilson loop)"
    )
    pb_heatmap_param_radius(
        model="A",
        params=np.linspace(0.0, 1.2, 121),
        radii=np.linspace(0.15, 1.2, 70),
        npts=1400,
        tol_overlap=1e-8
    )

    pb_vs_parameter(
        model="B",
        params=np.linspace(0.0, 1.0, 101),
        R=0.90,
        npts=2000,
        tol_overlap=1e-8,
        title_suffix="  (phase-sum Wilson loop)"
    )
    pb_heatmap_param_radius(
        model="B",
        params=np.linspace(0.0, 1.0, 101),
        radii=np.linspace(0.15, 1.2, 70),
        npts=1400,
        tol_overlap=1e-8
    )

    # ----------------------------
    # B) 3D degree Q (π3(S^3))
    #    using the lifted field to isolate the defect at λ=λ0.
    # ----------------------------
    Q_vs_parameter(
        model="A",
        params=np.linspace(0.15, 1.05, 31),  # centers a0
        kbox=0.35,
        lbox=0.20,
        Nk=41,
        Nl=33,
        eta=0.15,
        core_cut=5e-4
    )

    Q_vs_parameter(
        model="B",
        params=np.linspace(0.10, 0.90, 31),  # centers t0
        kbox=0.35,
        lbox=0.20,
        Nk=41,
        Nl=33,
        eta=0.15,
        core_cut=5e-4
    )
