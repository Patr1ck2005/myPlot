import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =========================
# Effective-field geometry
# =========================

# initial parameters
S0 = 0.15
delta0 = 0.35       # anisotropy field strength
kappa0 = 0.8
Delta0 = 1.0        # effective detuning: Delta = E - omega3 (simplified real model)

def compute_fields(S, delta, kappa, Delta):
    theta = 2 * np.pi * S

    # avoid division blow-up
    if abs(Delta) < 1e-6:
        Delta = 1e-6

    A = kappa**2 / (2 * Delta)

    # Pauli-space vectors in x-z plane
    h_delta = np.array([0.0, delta])
    h_v = np.array([A * np.sin(2 * theta), A * np.cos(2 * theta)])
    h_tot = h_delta + h_v

    return theta, A, h_delta, h_v, h_tot

# prepare figure
fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
ax_vec = fig.add_subplot(gs[0, 0])
ax_mag = fig.add_subplot(gs[0, 1])
plt.subplots_adjust(bottom=0.28, wspace=0.28)

# initial fields
theta0, A0, h_delta0, h_v0, h_tot0 = compute_fields(S0, delta0, kappa0, Delta0)

# -------- Left panel: vector geometry --------
ax_vec.set_title("Effective-field geometry in Pauli space")
ax_vec.set_xlabel(r"$h_x$")
ax_vec.set_ylabel(r"$h_z$")
ax_vec.axhline(0, color="gray", lw=1)
ax_vec.axvline(0, color="gray", lw=1)
ax_vec.grid(True)
ax_vec.set_aspect("equal")

# trajectory of h_v
S_circle = np.linspace(0, 1, 400)
hx_circle = []
hz_circle = []
for s in S_circle:
    _, _, _, hv, _ = compute_fields(s, delta0, kappa0, Delta0)
    hx_circle.append(hv[0])
    hz_circle.append(hv[1])

traj_line, = ax_vec.plot(hx_circle, hz_circle, "--", lw=1.5, label=r"trajectory of $\mathbf{h}_v$")

# arrows
arrow_delta = ax_vec.quiver(
    [0], [0], [h_delta0[0]], [h_delta0[1]],
    angles="xy", scale_units="xy", scale=1, width=0.008, label=r"$\mathbf{h}_\delta$"
)
arrow_v = ax_vec.quiver(
    [0], [0], [h_v0[0]], [h_v0[1]],
    angles="xy", scale_units="xy", scale=1, width=0.008, label=r"$\mathbf{h}_v$"
)
arrow_tot = ax_vec.quiver(
    [0], [0], [h_tot0[0]], [h_tot0[1]],
    angles="xy", scale_units="xy", scale=1, width=0.010, label=r"$\mathbf{h}=\mathbf{h}_\delta+\mathbf{h}_v$"
)

# point on h_v trajectory
point_v, = ax_vec.plot([h_v0[0]], [h_v0[1]], "o", ms=7)

# text box
txt = ax_vec.text(
    0.03, 0.97, "", transform=ax_vec.transAxes, va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), fontsize=10
)

ax_vec.legend(loc="lower left")

# -------- Right panel: |h| vs S --------
ax_mag.set_title(r"Magnitude $|\mathbf{h}(S)|$")
ax_mag.set_xlabel("Sliding parameter S")
ax_mag.set_ylabel(r"$|\mathbf{h}|$")
ax_mag.grid(True)

S_scan = np.linspace(0, 1, 600)
mag_vals = []
for s in S_scan:
    _, _, _, _, htmp = compute_fields(s, delta0, kappa0, Delta0)
    mag_vals.append(np.linalg.norm(htmp))
mag_vals = np.array(mag_vals)

mag_line, = ax_mag.plot(S_scan, mag_vals, lw=2)
vline = ax_mag.axvline(S0, color="gray", ls="--")
current_mag_pt, = ax_mag.plot([S0], [np.linalg.norm(h_tot0)], "o", ms=7)

# sliders
ax_S = plt.axes([0.16, 0.18, 0.70, 0.03])
ax_delta = plt.axes([0.16, 0.13, 0.70, 0.03])
ax_kappa = plt.axes([0.16, 0.08, 0.70, 0.03])
ax_Delta = plt.axes([0.16, 0.03, 0.70, 0.03])

s_S = Slider(ax_S, "S", 0.0, 1.0, valinit=S0)
s_delta = Slider(ax_delta, r"$\delta$", 0.0, 1.2, valinit=delta0)
s_kappa = Slider(ax_kappa, r"$\kappa$", 0.0, 1.5, valinit=kappa0)
s_Delta = Slider(ax_Delta, r"$\Delta=E-\omega_3$", 0.15, 2.0, valinit=Delta0)

def update(val):
    global arrow_delta, arrow_v, arrow_tot

    S = s_S.val
    delta = s_delta.val
    kappa = s_kappa.val
    Delta = s_Delta.val

    theta, A, h_delta, h_v, h_tot = compute_fields(S, delta, kappa, Delta)

    # update trajectory of h_v
    hx_circle = []
    hz_circle = []
    mag_vals = []
    for s in S_scan:
        _, _, _, hv, hsum = compute_fields(s, delta, kappa, Delta)
        hx_circle.append(hv[0])
        hz_circle.append(hv[1])
        mag_vals.append(np.linalg.norm(hsum))

    traj_line.set_data(hx_circle, hz_circle)

    # remove and redraw arrows
    arrow_delta.remove()
    arrow_v.remove()
    arrow_tot.remove()

    arrow_delta = ax_vec.quiver(
        [0], [0], [h_delta[0]], [h_delta[1]],
        angles="xy", scale_units="xy", scale=1, width=0.008
    )
    arrow_v = ax_vec.quiver(
        [0], [0], [h_v[0]], [h_v[1]],
        angles="xy", scale_units="xy", scale=1, width=0.008
    )
    arrow_tot = ax_vec.quiver(
        [0], [0], [h_tot[0]], [h_tot[1]],
        angles="xy", scale_units="xy", scale=1, width=0.010
    )

    point_v.set_data([h_v[0]], [h_v[1]])

    # auto limits
    lim = max(0.5, 1.25 * max(
        np.max(np.abs(hx_circle)),
        np.max(np.abs(hz_circle)),
        abs(h_tot[0]),
        abs(h_tot[1]),
        abs(delta)
    ))
    ax_vec.set_xlim(-lim, lim)
    ax_vec.set_ylim(-lim, lim)

    # update magnitude panel
    mag_vals = np.array(mag_vals)
    mag_line.set_ydata(mag_vals)
    ax_mag.set_ylim(0, 1.15 * np.max(mag_vals))
    vline.set_xdata([S, S])
    current_mag_pt.set_data([S], [np.linalg.norm(h_tot)])

    # dot product / angle info
    norm_hd = np.linalg.norm(h_delta)
    norm_hv = np.linalg.norm(h_v)
    if norm_hd > 1e-12 and norm_hv > 1e-12:
        cosang = np.dot(h_delta, h_v) / (norm_hd * norm_hv)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosang))
    else:
        angle_deg = np.nan

    txt.set_text(
        f"$\\theta=2\\pi S$ = {theta:.3f}\n"
        f"$A=\\kappa^2/(2\\Delta)$ = {A:.3f}\n"
        f"$\\mathbf{{h}}_\\delta=(0,\\delta)$ = ({h_delta[0]:.3f}, {h_delta[1]:.3f})\n"
        f"$\\mathbf{{h}}_v$ = ({h_v[0]:.3f}, {h_v[1]:.3f})\n"
        f"$\\mathbf{{h}}$ = ({h_tot[0]:.3f}, {h_tot[1]:.3f})\n"
        f"angle$(\\mathbf{{h}}_\\delta,\\mathbf{{h}}_v)$ = {angle_deg:.1f} deg"
    )

    fig.canvas.draw_idle()

s_S.on_changed(update)
s_delta.on_changed(update)
s_kappa.on_changed(update)
s_Delta.on_changed(update)

update(None)
plt.show()
