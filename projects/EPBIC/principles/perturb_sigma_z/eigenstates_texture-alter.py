import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import colors, cm

# ============================================================
# Global config
# ============================================================
OMEGA = 1.0
GAMMA = 0.12
S_SCAN = np.linspace(0.0, 0.5, 256)

BRANCH_CMAPS = ['hsv', 'hsv', 'hsv']

PLOT_STYLE = {
    "figsize": (2.6, 2.6),
    "view_elev": 22,
    "view_azim": -52,
    "traj_lw": 1.0,
    "arrow_count": 18,
    "arrow_length": 0.12,
    "arrow_lw": 1.0,
    "disk_alpha": 0.5,
    "disk_color": "lightgray",
    "disk_res_r": 80,
    "disk_res_t": 240,
}

# ============================================================
# Model
# ============================================================
def build_full_H(S, kappa, omega3, delta_w, delta_g, omega=OMEGA, gamma=GAMMA):
    theta = 2 * np.pi * S
    delta_tilde = delta_w - 1j * delta_g
    return np.array([
        [omega - 1j * gamma + delta_tilde, 0, kappa * np.cos(theta)],
        [0, omega - 1j * gamma - delta_tilde, kappa * np.sin(theta)],
        [kappa * np.cos(theta), kappa * np.sin(theta), omega3]
    ], dtype=complex)

def eig_track_over_S(kappa, omega3, delta_w, delta_g, s_scan=S_SCAN):
    vals0, vecs0 = np.linalg.eig(build_full_H(s_scan[0], kappa, omega3, delta_w, delta_g))
    idx0 = np.lexsort((vals0.imag, vals0.real))
    vals0 = vals0[idx0]
    vecs0 = vecs0[:, idx0]

    for j in range(3):
        vecs0[:, j] /= np.linalg.norm(vecs0[:, j])
        m = np.argmax(np.abs(vecs0[:, j]))
        vecs0[:, j] *= np.exp(-1j * np.angle(vecs0[m, j]))

    evals = np.zeros((len(s_scan), 3), dtype=complex)
    evecs = np.zeros((len(s_scan), 3, 3), dtype=complex)
    evals[0] = vals0
    evecs[0] = vecs0

    prev = vecs0.copy()
    perms = [
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0)
    ]

    for n, S in enumerate(s_scan[1:], start=1):
        vals, vecs = np.linalg.eig(build_full_H(S, kappa, omega3, delta_w, delta_g))
        for j in range(3):
            vecs[:, j] /= np.linalg.norm(vecs[:, j])

        ov = np.abs(prev.conj().T @ vecs)
        best_p = None
        best_score = -1.0
        for p in perms:
            score = ov[0, p[0]] + ov[1, p[1]] + ov[2, p[2]]
            if score > best_score:
                best_score = score
                best_p = p

        vals = vals[list(best_p)]
        vecs = vecs[:, list(best_p)]

        for j in range(3):
            phase_ref = np.vdot(prev[:, j], vecs[:, j])
            if np.abs(phase_ref) > 1e-14:
                vecs[:, j] *= np.exp(-1j * np.angle(phase_ref))
            vecs[:, j] /= np.linalg.norm(vecs[:, j])

        evals[n] = vals
        evecs[n] = vecs
        prev = vecs.copy()

    return evals, evecs

# ============================================================
# Weighted Stokes mapping
# ============================================================
def stokes_from_c1c2(c1, c2):
    s0 = np.abs(c1)**2 + np.abs(c2)**2
    if s0 < 1e-14:
        return np.array([np.nan, np.nan, np.nan]), 0.0
    s1 = (np.abs(c1)**2 - np.abs(c2)**2) / s0
    s2 = 2 * np.real(c1 * np.conj(c2)) / s0
    s3 = 2 * np.imag(c1 * np.conj(c2)) / s0
    return np.array([s1, s2, s3]), s0

def build_weighted_stokes_curves(evecs, s_scan=S_SCAN):
    curves = []
    for b in range(3):
        pts = []
        for n in range(len(s_scan)):
            c1, c2, c3 = evecs[n, :, b]
            st, s0_12 = stokes_from_c1c2(c1, c2)
            s0_123 = np.abs(c1)**2 + np.abs(c2)**2 + np.abs(c3)**2
            rho = s0_12 / s0_123 if s0_123 > 1e-14 else 0.0
            pts.append(rho * st)
        curves.append(np.array(pts))
    return curves

# ============================================================
# Plot helpers
# ============================================================
def decorate_stokes_ax(ax, style=PLOT_STYLE):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=style["view_elev"], azim=style["view_azim"])

def draw_filled_disk_z0(ax, style=PLOT_STYLE):
    r = np.linspace(0.0, 1.0, style["disk_res_r"])
    t = np.linspace(0.0, 2 * np.pi, style["disk_res_t"])
    R, T = np.meshgrid(r, t, indexing='ij')

    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z = np.zeros_like(X)

    ax.plot_surface(
        X, Y, Z,
        color=style["disk_color"],
        alpha=style["disk_alpha"],
        linewidth=0,
        antialiased=True,
        shade=False
    )

def add_colored_3d_line(ax, xyz, s_scan, cmap='viridis', lw=1.0):
    xyz = np.asarray(xyz)
    mask = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[mask]
    ss = s_scan[mask]

    if len(xyz) < 2:
        return None

    pts = xyz.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    norm = colors.Normalize(vmin=s_scan.min(), vmax=s_scan.max())
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=lw)
    lc.set_array(ss[:-1])
    ax.add_collection3d(lc)
    return lc

def add_direction_arrows_sampled(ax, xyz, s_scan, cmap='viridis', n_arrows=18, length=0.12, lw=1.0):
    xyz = np.asarray(xyz)
    mask = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[mask]
    ss = s_scan[mask]

    if len(xyz) < 5:
        return

    norm = colors.Normalize(vmin=s_scan.min(), vmax=s_scan.max())
    cmap_obj = cm.get_cmap(cmap)

    left = min(20, max(1, len(xyz) // 10))
    right = len(xyz) - left - 1
    if right <= left:
        idxs = np.linspace(1, len(xyz) - 2, min(n_arrows, max(1, len(xyz) - 2))).astype(int)
    else:
        idxs = np.linspace(left, right, n_arrows).astype(int)

    for idx in idxs:
        p = xyz[idx]
        v = xyz[idx + 1] - xyz[idx - 1]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v = v / nv * length
        color_here = cmap_obj(norm(ss[idx]))

        ax.quiver(
            p[0], p[1], p[2],
            v[0], v[1], v[2],
            color=color_here,
            linewidth=lw,
            arrow_length_ratio=1.0
        )

# ============================================================
# Main
# ============================================================
def main():
    params = {
        "kappa": 0.28,
        "omega3": 1.00,
        "delta_w": 0.06*0,
        "delta_g": 0.00,
    }

    _, evecs = eig_track_over_S(**params, s_scan=S_SCAN)
    curves = build_weighted_stokes_curves(evecs, s_scan=S_SCAN)

    fig = plt.figure(figsize=PLOT_STYLE["figsize"])
    ax = fig.add_subplot(111, projection='3d')

    decorate_stokes_ax(ax, style=PLOT_STYLE)
    draw_filled_disk_z0(ax, style=PLOT_STYLE)

    for i in range(3):
        add_colored_3d_line(
            ax, curves[i], S_SCAN,
            cmap=BRANCH_CMAPS[i],
            lw=PLOT_STYLE["traj_lw"]
        )
        add_direction_arrows_sampled(
            ax, curves[i], S_SCAN,
            cmap=BRANCH_CMAPS[i],
            n_arrows=PLOT_STYLE["arrow_count"],
            length=PLOT_STYLE["arrow_length"],
            lw=PLOT_STYLE["arrow_lw"]
        )

    # hide default 3D box, panes, grid, axis lines
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

    try:
        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))
    except Exception:
        pass

    plt.savefig('weighted_stokes_disk_z0.svg', bbox_inches='tight', transparent=True)
    plt.show()

if __name__ == '__main__':
    main()
