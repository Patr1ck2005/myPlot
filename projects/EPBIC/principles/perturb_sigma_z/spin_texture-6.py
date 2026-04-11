import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm, colors

# ============================================================
# Full 3x3 model
# ============================================================
OMEGA = 1.0
GAMMA = 0.12

def build_full_H(S, kappa, omega3, delta_w, delta_g, omega=OMEGA, gamma=GAMMA):
    theta = 2 * np.pi * S
    delta_tilde = delta_w - 1j * delta_g
    return np.array([
        [omega - 1j * gamma + delta_tilde, 0, kappa * np.cos(theta)],
        [0, omega - 1j * gamma - delta_tilde, kappa * np.sin(theta)],
        [kappa * np.cos(theta), kappa * np.sin(theta), omega3]
    ], dtype=complex)

def sorted_eigs(H):
    vals, vecs = np.linalg.eig(H)
    idx = np.argsort(vals.real)
    return vals[idx], vecs[:, idx]

def pairwise_min_gap(vals):
    d12 = np.abs(vals[0] - vals[1])
    d13 = np.abs(vals[0] - vals[2])
    d23 = np.abs(vals[1] - vals[2])
    return min(d12, d13, d23)

# ============================================================
# Branch-resolved effective fields
# h_eff = h0 I + hx sigma_x + hz sigma_z
# hy = 0 always
# ============================================================
def branch_effective_fields(S, kappa, omega3, delta_w, delta_g, branch):
    H = build_full_H(S, kappa, omega3, delta_w, delta_g)
    vals, vecs = sorted_eigs(H)
    E = vals[int(branch)]

    delta_tilde = delta_w - 1j * delta_g
    theta = 2 * np.pi * S
    phi = 2 * theta  # = 4 pi S

    Delta = E - omega3
    eps = 1e-12
    if abs(Delta) < eps:
        Delta = Delta + eps

    A = kappa**2 / (2 * Delta)

    hx = A * np.sin(phi)
    hz = delta_tilde + A * np.cos(phi)

    D = hx**2 + hz**2
    return {
        "E": E,
        "hx": hx,
        "hz": hz,
        "D": D,
        "gap": pairwise_min_gap(vals),
    }

# ============================================================
# Jones -> normalized Stokes
# J = (hz, hx)^T
# ============================================================
def jones_to_stokes(hz, hx):
    S0 = np.abs(hz)**2 + np.abs(hx)**2
    if S0 < 1e-14:
        return np.array([np.nan, np.nan, np.nan]), S0

    S1 = (np.abs(hz)**2 - np.abs(hx)**2) / S0
    S2 = 2 * np.real(hz * np.conj(hx)) / S0
    S3 = 2 * np.imag(hz * np.conj(hx)) / S0
    return np.array([S1, S2, S3]), S0

# ============================================================
# Scan over S for all branches
# ============================================================
S_SCAN = np.linspace(0.0, 1.0, 500)

def compute_branch_curves(kappa, omega3, delta_w, delta_g):
    stokes_curves = []
    D_curves = []
    gap_curves = []

    for branch in range(3):
        pts = []
        Dvals = []
        gaps = []
        for S in S_SCAN:
            dat = branch_effective_fields(S, kappa, omega3, delta_w, delta_g, branch)
            stokes, _ = jones_to_stokes(dat["hz"], dat["hx"])
            pts.append(stokes)
            Dvals.append(dat["D"])
            gaps.append(dat["gap"])
        stokes_curves.append(np.array(pts))
        D_curves.append(np.array(Dvals))
        gap_curves.append(np.array(gaps))

    return stokes_curves, D_curves, gap_curves

# ============================================================
# Utilities for colored 3D line + direction arrows
# ============================================================
def add_colored_3d_line(ax, xyz, cmap='viridis', lw=3.0):
    """
    xyz: (N,3)
    color mapped by parameter index -> here effectively by S
    """
    pts = xyz.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    norm = colors.Normalize(vmin=S_SCAN.min(), vmax=S_SCAN.max())
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=lw)
    lc.set_array(S_SCAN[:-1])
    ax.add_collection3d(lc)
    return lc, norm

def add_direction_arrows(ax, xyz, n_arrows=6, color='k', length=0.1, lw=1.5):
    """
    Place arrows along trajectory direction.
    """
    N = len(xyz)
    if N < 3:
        return []

    idxs = np.linspace(20, N - 21, n_arrows).astype(int)
    arrows = []

    for idx in idxs:
        p = xyz[idx]
        v = xyz[idx + 1] - xyz[idx - 1]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v = v / nv * length

        q = ax.quiver(
            p[0], p[1], p[2],
            v[0], v[1], v[2],
            color=color, linewidth=lw, arrow_length_ratio=1,
        )
        arrows.append(q)
    return arrows

# ============================================================
# Figure layout
# ============================================================
fig = plt.figure(figsize=(14, 6.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0])

ax_sphere = fig.add_subplot(gs[0, 0], projection='3d')
ax_D = fig.add_subplot(gs[0, 1])

plt.subplots_adjust(bottom=0.28, wspace=0.24)

branch_colors = ['tab:red', 'tab:blue', 'tab:green']
branch_labels = ['branch 0', 'branch 1', 'branch 2']
branch_cmaps = ['Reds', 'Blues', 'Greens']

# initial parameters
S0 = 0.18
kappa0 = 0.28
omega30 = 1.02
dw0 = 0.06
dg0 = 0.03

# sphere mesh
u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 40)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

# static sphere
ax_sphere.plot_surface(xs, ys, zs, color='lightgray', alpha=0.10, linewidth=0)
ax_sphere.plot_wireframe(xs, ys, zs, color='gray', alpha=0.12, rstride=8, cstride=8)

# EP poles
ax_sphere.scatter([0, 0], [0, 0], [1, -1], s=90, c=['k', 'k'], marker='*')
ax_sphere.text(0, 0, 1.10, 'EP pole (+)', ha='center')
ax_sphere.text(0, 0, -1.15, 'EP pole (-)', ha='center')

ax_sphere.set_title("Three-branch Poincaré texture of $(h_z,h_x)$")
ax_sphere.set_xlabel(r'$S_1$')
ax_sphere.set_ylabel(r'$S_2$')
ax_sphere.set_zlabel(r'$S_3$')
ax_sphere.set_xlim(-1, 1)
ax_sphere.set_ylim(-1, 1)
ax_sphere.set_zlim(-1, 1)
ax_sphere.set_box_aspect([1, 1, 1])
ax_sphere.view_init(elev=22, azim=-52)

# right panel
ax_D.set_title(r'EP proximity: $-\log_{10}|D_n|$,  $D_n=h_x^2+h_z^2$')
ax_D.set_xlabel('S')
ax_D.set_ylabel(r'$-\log_{10}|D_n|$')
ax_D.grid(True)

txt = ax_D.text(
    0.02, 0.98, "", transform=ax_D.transAxes, va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), fontsize=10
)

# artist containers
curve_lcs = [None, None, None]
current_pts = [None, None, None]
arrow_sets = [[], [], []]
D_lines = []
D_pts = []

for c, lab in zip(branch_colors, branch_labels):
    lineD, = ax_D.plot([], [], lw=2.2, color=c, label=lab)
    ptD, = ax_D.plot([], [], 'o', ms=7, color=c)
    D_lines.append(lineD)
    D_pts.append(ptD)

gap_line, = ax_D.plot([], [], 'k--', lw=1.6, label='exact min gap (all branches)')
vline = ax_D.axvline(S0, color='gray', ls='--', lw=1.2)

ax_D.legend(loc='upper right', fontsize=9)

# add colorbar for S
sm = cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap='viridis')
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_sphere, fraction=0.035, pad=0.06)
cbar.set_label('S (trajectory color)')

# sliders
ax_S = plt.axes([0.12, 0.18, 0.75, 0.03])
ax_k = plt.axes([0.12, 0.13, 0.75, 0.03])
ax_w3 = plt.axes([0.12, 0.08, 0.75, 0.03])
ax_dw = plt.axes([0.12, 0.03, 0.35, 0.03])
ax_dg = plt.axes([0.52, 0.03, 0.35, 0.03])

sS = Slider(ax_S, 'S', 0.0, 1.0, valinit=S0)
sk = Slider(ax_k, r'$\kappa$', 0.0, 0.65, valinit=kappa0)
sw3 = Slider(ax_w3, r'$\omega_3$', 0.70, 1.30, valinit=omega30)
sdw = Slider(ax_dw, r'$\delta\omega$', 0.0, 0.20, valinit=dw0)
sdg = Slider(ax_dg, r'$\delta\gamma$', 0.0, 0.20, valinit=dg0)

def update(val):
    S = sS.val
    kappa = sk.val
    omega3 = sw3.val
    delta_w = sdw.val
    delta_g = sdg.val

    stokes_curves, D_curves, gap_curves = compute_branch_curves(kappa, omega3, delta_w, delta_g)

    # clear old 3D dynamic artists
    for i in range(3):
        if curve_lcs[i] is not None:
            curve_lcs[i].remove()
            curve_lcs[i] = None
        if current_pts[i] is not None:
            current_pts[i].remove()
            current_pts[i] = None
        for ar in arrow_sets[i]:
            ar.remove()
        arrow_sets[i] = []

    # redraw sphere trajectories with S-color mapping + arrows
    for i in range(3):
        curve = stokes_curves[i]
        curve_lcs[i], _ = add_colored_3d_line(ax_sphere, curve, cmap=branch_cmaps[i], lw=3.0)

        idx = np.argmin(np.abs(S_SCAN - S))
        cur = curve[idx]
        current_pts[i] = ax_sphere.scatter(
            [cur[0]], [cur[1]], [cur[2]],
            s=90, color=branch_colors[i], edgecolors='k', linewidths=0.6
        )

        arrow_sets[i] = add_direction_arrows(
            ax_sphere, curve, n_arrows=30, color=branch_colors[i], length=0.2, lw=1.4
        )

    # update D curves
    all_D_vals = []
    exact_gap = np.min(np.array(gap_curves), axis=0)

    for i in range(3):
        y = -np.log10(np.abs(D_curves[i]) + 1e-16)
        D_lines[i].set_data(S_SCAN, y)

        idx = np.argmin(np.abs(S_SCAN - S))
        D_pts[i].set_data([S], [y[idx]])
        all_D_vals.append(y)

    gap_y = -np.log10(exact_gap + 1e-16)
    gap_line.set_data(S_SCAN, gap_y)
    vline.set_xdata([S, S])

    all_y = np.concatenate(all_D_vals + [gap_y])
    ymin = np.nanmin(all_y)
    ymax = np.nanmax(all_y)
    ax_D.set_ylim(ymin - 0.3, ymax + 0.3)

    # info text
    cur_text_lines = []
    for i in range(3):
        dat = branch_effective_fields(S, kappa, omega3, delta_w, delta_g, i)
        st, _ = jones_to_stokes(dat["hz"], dat["hx"])
        cur_text_lines.append(
            f"b{i}: E={dat['E'].real:.4f}{dat['E'].imag:+.4f}i, "
            f"|D|={abs(dat['D']):.2e}, "
            f"(S1,S2,S3)=({st[0]:.2f},{st[1]:.2f},{st[2]:.2f})"
        )

    txt.set_text(
        "Trajectory color encodes S; arrows indicate increasing S\n"
        "EP on Poincaré sphere: poles (0,0,±1)\n"
        + "\n".join(cur_text_lines)
    )

    fig.canvas.draw_idle()

for s in [sS, sk, sw3, sdw, sdg]:
    s.on_changed(update)

update(None)
plt.show()
