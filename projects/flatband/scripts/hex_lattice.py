import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Utilities: build k-path (M-Γ-K or M-Γ-X) with true path coordinate s
# -------------------------------
def build_path(k_points, n_per_segment=600):
    k_points = [np.array(p, dtype=float) for p in k_points]
    s_all, k_all = [], []
    tick_s = [0.0]
    s_cum = 0.0

    for i in range(len(k_points) - 1):
        p0, p1 = k_points[i], k_points[i + 1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))

        t = np.linspace(0.0, 1.0, n_per_segment, endpoint=(i == len(k_points) - 2))
        k_seg = p0[None, :] + t[:, None] * seg
        s_seg = s_cum + t * seg_len

        if i > 0:
            k_seg = k_seg[1:, :]
            s_seg = s_seg[1:]

        k_all.append(k_seg)
        s_all.append(s_seg)

        s_cum += seg_len
        tick_s.append(s_cum)

    kvec = np.vstack(k_all)
    s = np.concatenate(s_all)
    kmag = np.linalg.norm(kvec, axis=1)
    return s, kvec, kmag, tick_s


# -------------------------------
# Reciprocal vectors
# -------------------------------
def reciprocal_vectors_square(N=6):
    G = []
    for m in range(-N, N + 1):
        for n in range(-N, N + 1):
            if m == 0 and n == 0:
                continue
            G.append([m, n])  # b1~=(1,0), b2~=(0,1)
    return np.array(G, dtype=float)


def hex_reciprocal_basis(a_def="flat"):
    """
    Return (b1~, b2~) in normalized units k̃ = k / (2π/a_def)
    Two consistent conventions:

    a_def="prim": a = Bravais primitive translation length (nearest-neighbor spacing).
      Then |b| = 4π/(√3 a) => in units of (2π/a): |b| = 2/√3
      One convenient choice:
        b1~ = ( 2/√3, 0 )
        b2~ = ( 1/√3, 1 )

    a_def="flat": a = flat-to-flat distance of Wigner-Seitz hexagon.
      Relation: a_flat = √3 a_prim
      Therefore b1~, b2~ are scaled by √3 compared with "prim".
      A convenient choice:
        b1~ = (2, 0)
        b2~ = (1, √3)
    """
    if a_def == "prim":
        b1 = np.array([2.0 / np.sqrt(3.0), 0.0])
        b2 = np.array([1.0 / np.sqrt(3.0), 1.0])
    elif a_def == "flat":
        b1 = np.array([2.0, 0.0])
        b2 = np.array([1.0, np.sqrt(3.0)])
    else:
        raise ValueError("a_def must be 'flat' or 'prim'")
    return b1, b2


def reciprocal_vectors_hex(N=6, a_def="flat"):
    b1, b2 = hex_reciprocal_basis(a_def=a_def)
    G = []
    for h in range(-N, N + 1):
        for k in range(-N, N + 1):
            if h == 0 and k == 0:
                continue
            G.append(h * b1 + k * b2)
    return np.array(G, dtype=float)


def hex_high_symmetry_points(a_def="flat"):
    """
    Return (M, Γ, K) in the same normalized coordinates k̃ = k/(2π/a_def),
    consistent with the chosen reciprocal basis.

    In this coordinate system:
      Γ = (0,0)
      M = b1~/2
      K = (b1~ + b2~)/3
    """
    b1, b2 = hex_reciprocal_basis(a_def=a_def)
    G = np.array([0.0, 0.0])
    M = 0.5 * b1
    K = (b1 + b2) / 3.0
    return M, G, K


# -------------------------------
# Strict diffraction threshold
# -------------------------------
def strict_fdif_curve(kvec, G_list, n_clad=1.0):
    diffs = kvec[:, None, :] + G_list[None, :, :]
    norms = np.linalg.norm(diffs, axis=2)
    min_norm = np.min(norms, axis=1)
    return min_norm / n_clad


# -------------------------------
# Plot
# -------------------------------
def plot_full_map(path_labels, k_points, G_list, n_clad=1.0, title=""):
    s, kvec, kmag, tick_s = build_path(k_points)

    f_light = kmag / n_clad
    f_dif = strict_fdif_curve(kvec, G_list, n_clad=n_clad)

    y_max = max(np.max(f_dif), np.max(f_light)) * 1.35
    y_max = max(y_max, 0.5)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    lower = np.minimum(f_light, f_dif)
    upper = np.maximum(f_light, f_dif)

    ax.fill_between(s, 0, lower, alpha=0.15, label="Guided & no-diffraction")

    mask_rad_nodif = f_light < f_dif
    ax.fill_between(s, f_light, f_dif, where=mask_rad_nodif, interpolate=True,
                    alpha=0.15, label="Radiative & no-diffraction")

    mask_guid_dif = f_dif < f_light
    ax.fill_between(s, f_dif, f_light, where=mask_guid_dif, interpolate=True,
                    alpha=0.15, label="Guided & diffractive")

    ax.fill_between(s, upper, y_max, alpha=0.15, label="Radiative & diffractive")

    ax.plot(s, f_light, linewidth=2.0, label=f"Light line (n={n_clad:g})")
    ax.plot(s, f_dif, linewidth=2.0, label="Strict diffraction boundary")

    for xs in tick_s:
        ax.axvline(xs, linestyle=":", linewidth=1.2)

    ax.set_xticks(tick_s)
    ax.set_xticklabels(path_labels)

    ax.set_xlim(tick_s[0], tick_s[-1])
    ax.set_ylim(0, y_max)

    ax.set_xlabel("Path coordinate s (in normalized k̃ units)")
    ax.set_ylabel(r"Normalized frequency  $\tilde{f} = f/(c/a)$")
    ax.set_title(title)

    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # -----------------------
    # Switch here:
    #   "flat" : a = flat-to-flat distance (your current convention)
    #   "prim" : a = primitive translation length (common in literature)
    # -----------------------
    a_def_hex = "prim"   # change to "prim" to match many papers
    N_enum = 7
    n_clad = 1.0

    # ========= Hexagonal =========
    G_hex = reciprocal_vectors_hex(N=N_enum, a_def=a_def_hex)
    k_hex_M, k_hex_G, k_hex_K = hex_high_symmetry_points(a_def=a_def_hex)

    plot_full_map(
        path_labels=["M", "Γ", "K"],
        k_points=[k_hex_M, k_hex_G, k_hex_K],
        G_list=G_hex,
        n_clad=n_clad,
        title=f"Hexagonal lattice: M–Γ–K  (a_def='{a_def_hex}')\nLight line + strict diffraction regions"
    )

    # ========= Square =========
    # Square lattice has no ambiguity; keep standard a = lattice constant.
    G_sq = reciprocal_vectors_square(N=N_enum)
    k_sq_M = (0.5, 0.5)
    k_sq_G = (0.0, 0.0)
    k_sq_X = (0.5, 0.0)

    plot_full_map(
        path_labels=["M", "Γ", "X"],
        k_points=[k_sq_M, k_sq_G, k_sq_X],
        G_list=G_sq,
        n_clad=n_clad,
        title="Square lattice: M–Γ–X\nLight line + strict diffraction regions"
    )
