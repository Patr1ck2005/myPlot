import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.widgets import Slider, CheckButtons, RadioButtons

def demo_ringbeam_minimal(
    Q=3.0, N=512,
    zeta_max=100.0, Nzeta=31,
    qr=0.7, s=0.0, c=0.0, gamma=5e-3, l=1,
    use_evanescent=True,
    slice_axis="y=0",
    normalize_mode="global",
):
    # ---------- grid from (Q, N) ----------
    def grids(Q_, N_):
        N_ = int(N_)
        dq = 2*Q_ / N_
        q = (np.arange(N_) - N_//2) * dq             # centered q-grid
        QX, QY = np.meshgrid(q, q, indexing="xy")
        RQ = np.sqrt(QX**2 + QY**2)
        PHI = np.arctan2(QY, QX)

        Lx_p = 2*np.pi / dq                          # = pi*N/Q
        dxp = Lx_p / N_                              # = pi/Q
        xp = (np.arange(N_) - N_//2) * dxp
        return dq, q, xp, QX, QY, RQ, PHI, Lx_p, dxp

    def fft_u(Ms):   # shifted -> unshifted
        return np.fft.ifftshift(Ms)

    def fft_s(Mu):   # unshifted -> shifted
        return np.fft.fftshift(Mu)

    # ---------- single-ring safe D(q) ----------
    # D(q)= s*(q-qr) + 0.5*c*(q-qr)^2
    # second root q2=qr - 2s/c  (if c!=0)
    # require q2<0 <=> s/c > qr/2, otherwise we DROP c -> linear band
    def safe_D(RQ_, qr_, s_, c_):
        # if c_ != 0.0:
        #     q2 = qr_ - 2.0*s_/c_
        #     if q2 >= 0.0:
        #         c_ = 0.0  # hard fallback to linear band (one root)
        d = RQ_ - qr_
        return s_*d + 0.5*c_*d*d, c_

    def build_A(RQ_, PHI_, qr_, s_, c_, gamma_, l_):
        D, c_eff = safe_D(RQ_, qr_, s_, c_)
        A = np.exp(1j*l_*PHI_) * (gamma_ / (D + 1j*gamma_))
        # enforce vortex axis null at q=0 for l!=0 (discrete fix)
        if l_ != 0:
            A[RQ_ == 0] = 0.0
        return A, c_eff

    def kz_over_k0(RQ_, use_evan_):
        if use_evan_:
            return np.sqrt((1.0 - RQ_**2) + 0j)
        out = np.zeros_like(RQ_, dtype=np.complex128)
        m = RQ_ <= 1.0
        out[m] = np.sqrt((1.0 - RQ_[m]**2) + 0j)
        return out

    def slice_I(A_shifted, kz_shifted, zmax_, Nz_, xp_, axis_, norm_):
        zetas = np.linspace(-zmax_, zmax_, int(Nz_))
        N_ = A_shifted.shape[0]
        cx = N_//2
        I = np.empty((len(zetas), N_), float)

        Au = fft_u(A_shifted)
        ku = fft_u(kz_shifted)
        for i, zt in enumerate(zetas):
            Hu = np.exp(1j*zt*ku)
            Uu = np.fft.ifft2(Au * Hu)
            Us = fft_s(Uu)
            line = Us[cx, :] if axis_ == "y=0" else Us[:, cx]
            I[i, :] = (line.real**2 + line.imag**2)

        if norm_ == "per_zeta":
            I /= (I.max(axis=1, keepdims=True))
        else:
            I /= (I.max())
        return zetas, I

    # ---------- initial compute ----------
    dq, q1d, xp, QX, QY, RQ, PHI, Lx_p, dxp = grids(Q, N+1)
    A, c_eff = build_A(RQ, PHI, qr, s, c, gamma, int(l))
    kz = kz_over_k0(RQ, use_evanescent)
    zetas, Iimg = slice_I(A, kz, zeta_max, Nzeta, xp, slice_axis, normalize_mode)

    # ---------- plot ----------
    plt.close("all")
    fig = plt.figure(figsize=(14.5, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35])
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])

    im1 = ax1.imshow(np.real(A), origin="lower",
                     extent=[-Q,Q,-Q,Q], aspect="equal", cmap="RdBu"
                     # norm=SymLogNorm(linthresh=1e-2, vmin=-1, vmax=1)
                     )
    ax1.set_title(r"$\Re[A(q_x,q_y)]$")
    ax1.set_xlabel(r"$q_x$"); ax1.set_ylabel(r"$q_y$")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(np.abs(A), origin="lower",
                     extent=[-Q,Q,-Q,Q], aspect="equal")
    ax2.set_title(r"$|A(q_x,q_y)|$")
    ax2.set_xlabel(r"$q_x$"); ax2.set_ylabel(r"$q_y$")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    im3 = ax3.imshow(Iimg, origin="lower",
                     extent=[xp.min(), xp.max(), zetas.min(), zetas.max()],
                     aspect="auto", vmin=0.0, vmax=1.0)
    ax3.set_title(r"Slice $I(x',\zeta)$ (normalized)")
    ax3.set_xlabel(r"$x'$" if slice_axis=="y=0" else r"$y'$")
    ax3.set_ylabel(r"$\zeta$")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    info = fig.text(0.02, 0.96,
        f"Q={Q:.3g}, N={N}, Δq={dq:.3g}, Δx'={dxp:.3g}, Lx'={Lx_p:.3g},  c_eff={c_eff:.3g}",
        fontsize=10)

    # ---------- widgets ----------
    plt.subplots_adjust(left=0.05, right=0.985, bottom=0.28, top=0.92, wspace=0.30)
    axcolor = "lightgoldenrodyellow"

    ax_Q   = fig.add_axes([0.07, 0.20, 0.25, 0.03], facecolor=axcolor)
    ax_N   = fig.add_axes([0.07, 0.16, 0.25, 0.03], facecolor=axcolor)
    ax_qr  = fig.add_axes([0.07, 0.12, 0.25, 0.03], facecolor=axcolor)
    ax_s   = fig.add_axes([0.07, 0.08, 0.25, 0.03], facecolor=axcolor)
    ax_c   = fig.add_axes([0.07, 0.04, 0.25, 0.03], facecolor=axcolor)

    ax_g   = fig.add_axes([0.37, 0.20, 0.22, 0.03], facecolor=axcolor)
    ax_l   = fig.add_axes([0.37, 0.16, 0.22, 0.03], facecolor=axcolor)
    ax_zm  = fig.add_axes([0.37, 0.12, 0.22, 0.03], facecolor=axcolor)
    ax_Nz  = fig.add_axes([0.37, 0.08, 0.22, 0.03], facecolor=axcolor)

    sQ  = Slider(ax_Q,  "Q",   0.5, 10.0, valinit=Q,  valstep=0.01)
    sN  = Slider(ax_N,  "N",   128, 2048, valinit=N,  valstep=128)
    sqr = Slider(ax_qr, "qr",  0.0, 5.0,  valinit=qr, valstep=0.001)
    ss  = Slider(ax_s,  "D'",  -1.0, 1.0,  valinit=s,  valstep=0.001)
    sc  = Slider(ax_c,  "D''",  -1.0, 1.0,  valinit=c,  valstep=0.001)

    sg  = Slider(ax_g,  "gamma", 1e-5, 1e-2, valinit=gamma, valstep=1e-5)
    sl  = Slider(ax_l,  "l",   -5, 5,  valinit=l, valstep=1)
    szm = Slider(ax_zm, "zeta_max", 1.0, 300.0, valinit=zeta_max, valstep=0.5)
    sNz = Slider(ax_Nz, "Nzeta",  11, 201, valinit=Nzeta, valstep=2)

    ax_chk = fig.add_axes([0.64, 0.16, 0.14, 0.10], facecolor=axcolor)
    chk = CheckButtons(ax_chk, ["evanescent"], [use_evanescent])

    ax_axis = fig.add_axes([0.80, 0.16, 0.14, 0.10], facecolor=axcolor)
    rb_axis = RadioButtons(ax_axis, ["y=0", "x=0"], active=0 if slice_axis=="y=0" else 1)

    ax_norm = fig.add_axes([0.64, 0.04, 0.14, 0.10], facecolor=axcolor)
    rb_norm = RadioButtons(ax_norm, ["global", "per_zeta"], active=0 if normalize_mode=="global" else 1)

    state = {"evan": use_evanescent, "axis": slice_axis, "norm": normalize_mode}

    def recompute(_=None):
        nonlocal q1d, xp, RQ, PHI, dq, dxp, Lx_p
        Qv = sQ.val
        Nv = int(round(sN.val))
        dq, q1d, xp, QX, QY, RQ, PHI, Lx_p, dxp = grids(Qv, Nv+1)

        Av, c_eff2 = build_A(RQ, PHI, sqr.val, ss.val, sc.val, sg.val, int(round(sl.val)))
        kzv = kz_over_k0(RQ, state["evan"])
        zetasv, Iimgv = slice_I(Av, kzv, szm.val, int(round(sNz.val)),
                                xp, state["axis"], state["norm"])

        # crop update
        idx2 = np.where(np.abs(q1d) <= Qv)[0]
        j0, j1 = idx2.min(), idx2.max()
        def crop2(M): return M[j0:j1+1, j0:j1+1]

        im1.set_data(np.real(crop2(Av)))
        im2.set_data(np.abs(crop2(Av)))
        im1.set_extent([-Qv,Qv,-Qv,Qv])
        im2.set_extent([-Qv,Qv,-Qv,Qv])
        ax1.set_xlim(-Qv,Qv); ax1.set_ylim(-Qv,Qv)
        ax2.set_xlim(-Qv,Qv); ax2.set_ylim(-Qv,Qv)

        im3.set_data(Iimgv)
        im3.set_extent([xp.min(), xp.max(), zetasv.min(), zetasv.max()])
        ax3.set_xlim(xp.min(), xp.max()); ax3.set_ylim(zetasv.min(), zetasv.max())
        ax3.set_xlabel(r"$x'$" if state["axis"]=="y=0" else r"$y'$")

        info.set_text(
            f"Q={Qv:.3g}, N={Nv}, Δq={dq:.3g}, Δx'={dxp:.3g}, Lx'={Lx_p:.3g},  c_eff={c_eff2:.3g}"
        )
        fig.canvas.draw_idle()

    def on_chk(_lbl):
        state["evan"] = not state["evan"]
        recompute()

    def on_axis(lbl):
        state["axis"] = lbl
        recompute()

    def on_norm(lbl):
        state["norm"] = lbl
        recompute()

    chk.on_clicked(on_chk)
    rb_axis.on_clicked(on_axis)
    rb_norm.on_clicked(on_norm)

    for sld in [sQ,sN,sqr,ss,sc,sg,sl,szm,sNz]:
        sld.on_changed(recompute)

    plt.show()

if __name__ == "__main__":
    demo_ringbeam_minimal()
