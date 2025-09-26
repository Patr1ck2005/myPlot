# fit_viz_svg.py
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from physics_core import SystemParams, NonHermitianTwoLevel

# ======= Config =======
# DATA_PATH = r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\PL\3fold_PL_QGM-rad.pkl"
DATA_PATH = r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\PL\3fold_PL_BIC-rad.pkl"
# DATA_PATH = r"D:\DELL\Documents\myPlots\plot_3D\projects\SE\rsl\manual_datas\PL\3fold_PL_BIC-tot.pkl"
MODE = "Prad"  # or "Ptot"
# MODE = "Ptot"  # or "Ptot"
fs = 12  # font size for plots
plt.rcParams.update({"font.size": fs})


# Best-fit params (your latest)
PARAMS = {
    "omega0": 0.4413,
    "delta":  0.00585,
    "gamma0": 1.2e-4,
    "d1":     0.91,
    "d2":     0.0,
    # "d1": 0,
    # "d2": 1,
    "gamma_rad": 3.5e-4,
    "dispersion_v": 0.29,
}

SAVE_DIR = os.path.abspath(".")   # where to save SVGs
SAVE_PREFIX = "fit_svg"           # filename prefix


# ======= Utils =======
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_svg(fig, name):
    ensure_dir(SAVE_DIR)
    path = os.path.join(SAVE_DIR, f"{SAVE_PREFIX}_{name}.svg")
    # remove ticklabel but keep ticks
    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    # remove legend
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Saved: {path}")

def load_data(path):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    x_vals = raw.get("x_vals", np.array([]))  # k
    y_vals = raw.get("y_vals", np.array([]))  # freq(=omega)
    subs = raw.get("subs", [])
    print(f"Pickle: x_shape={x_vals.shape}, subs_len={len(subs)}")
    z = subs[0].real  # (k, freq)
    dataZ = z.T.copy()  # -> (omega, k)
    return np.asarray(x_vals, float), np.asarray(y_vals, float), np.asarray(dataZ, float)

def compute_model_grid_fast(sp: SystemParams,
                            gamma_rad: float,
                            k_grid: np.ndarray,
                            omega_grid: np.ndarray,
                            mode: str) -> np.ndarray:
    model = NonHermitianTwoLevel(sp)
    if hasattr(model, "power_on_grid_fast"):
        Z = model.power_on_grid_fast(mode, float(gamma_rad),
                                     np.asarray(k_grid, float),
                                     np.asarray(omega_grid, float))
        return np.asarray(Z, float)
    # fallback (should be rare)
    power_fn = model.prad if mode.lower()=="prad" else model.ptot
    Z = np.zeros((omega_grid.size, k_grid.size), float)
    for i, om in enumerate(omega_grid):
        for j, kk in enumerate(k_grid):
            Z[i, j] = power_fn(om, kk, gamma_rad)
    return Z

def resize_nearest(A, new_shape):
    """Nearest-neighbor resize for (omega, k) shaped arrays."""
    si, sj = A.shape
    ni, nj = new_shape
    out = np.zeros((ni, nj), dtype=A.dtype)
    for i in range(ni):
        ii = int(round(i * (si - 1) / max(ni - 1, 1)))
        for j in range(nj):
            jj = int(round(j * (sj - 1) / max(nj - 1, 1)))
            out[i, j] = A[ii, jj]
    return out

def normalize01(A):
    m = np.max(A)
    return A / (m if m != 0 else 1.0)

def k_omega_indices(k_grid, omega_grid):
    """Pick 3 representative indices (20%, 50%, 80%) for each axis."""
    idxs_k = [int(0.2*len(k_grid)), len(k_grid)//2, int(0.8*len(k_grid))]
    idxs_w = [int(0.2*len(omega_grid)), len(omega_grid)//2, int(0.8*len(omega_grid))]
    # unique & clipped
    idxs_k = sorted(set(np.clip(idxs_k, 0, len(k_grid)-1)))
    idxs_w = sorted(set(np.clip(idxs_w, 0, len(omega_grid)-1)))
    return idxs_k, idxs_w


# ======= Plot helpers (single-figure each) =======
def plot_heatmap(Z, k_grid, omega_grid, title, fname, vmin=None, vmax=None):
    # fig = plt.figure(figsize=(2, 4))
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_subplot(111)
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   extent=[k_grid.min(), k_grid.max(), omega_grid.min(), omega_grid.max()],
                   vmin=vmin, vmax=vmax, interpolation="none", cmap='magma')
    # ax.set_title(title)
    ax.set_xlabel("$k (2\pi/P)$"); ax.set_ylabel(r"f (c/P)")
    ax.set_xlim(-0.05*0, 0.01)
    ax.set_ylim(0.4465, 0.4485)
    # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_svg(fig, fname)
    plt.close(fig)

def plot_line_xy(x, y_list, labels, title, xlabel, ylabel, fname, xlim=None, ylim=None):
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(111)
    for y, lab in zip(y_list, labels):
        ax.plot(x, y, label=lab)
    # ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    save_svg(fig, fname)
    plt.close(fig)

def plot_hist(data, bins, title, xlabel, ylabel, fname):
    fig = plt.figure(figsize=(6.6, 4.0))
    ax = fig.add_subplot(111)
    ax.hist(data, bins=bins, alpha=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    save_svg(fig, fname)
    plt.close(fig)

def plot_scatter(x, y, title, xlabel, ylabel, fname):
    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(111)
    ax.plot(x, y, '.', alpha=0.35, markersize=2)
    # y=x line
    lims = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
    ax.plot(lims, lims, '-', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    save_svg(fig, fname)
    plt.close(fig)


# ======= Main =======
def main():
    # 1) load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Please set DATA_PATH correctly.")
    k_grid, omega_grid, dataZ = load_data(DATA_PATH)

    # shape ensure
    target_shape = (omega_grid.size, k_grid.size)
    if dataZ.shape != target_shape:
        print("⚠️ dataZ shape mismatch; resizing by nearest neighbor…")
        dataZ = resize_nearest(dataZ, target_shape)

    # 2) model grid
    sp = SystemParams(
        omega0=PARAMS["omega0"], delta=PARAMS["delta"], gamma0=PARAMS["gamma0"],
        d1=PARAMS["d1"], d2=PARAMS["d2"], dispersion_v=PARAMS["dispersion_v"]
    )
    Zm = compute_model_grid_fast(sp, PARAMS["gamma_rad"], k_grid, omega_grid, MODE)

    # 3) normalize (no affine)
    dataZ_n = normalize01(dataZ)
    # QGM_sim_norm_factor = 21543414367
    # sim_norm_factor = 82088874732  # BIC, tot
    Zm_n    = normalize01(Zm)
    # QGM_anal_norm_factor = 3167
    # anal_norm_factor = 13761  # BIC, tot
    # dataZ_n = dataZ / sim_norm_factor
    # Zm_n    = Zm / anal_norm_factor
    diff_n  = Zm_n - dataZ_n

    # 4) heatmaps (each separate SVG)
    # 对于数据进行y轴的镜像对称扩展
    k_grid_mirrored = np.concatenate((-k_grid[::-1], k_grid))
    dataZ_n_mirrored = np.concatenate((dataZ_n[:, ::-1], dataZ_n), axis=1)

    Zm_n_mirrored = np.concatenate((Zm_n[:, ::-1], Zm_n), axis=1)

    # plot_heatmap(dataZ_n_mirrored, k_grid_mirrored, omega_grid,
    #              title="Data (normalized)", fname="heatmap_data_normalized",
    #              vmin=0.0, vmax=1.0)
    # plot_heatmap(Zm_n_mirrored, k_grid_mirrored, omega_grid,
    #              title=f"Model {MODE} (normalized)", fname="heatmap_model_normalized",
    #              vmin=0.0, vmax=1.0)
    plot_heatmap(dataZ_n, k_grid, omega_grid,
                 title="Data (normalized)", fname="heatmap_data_normalized",
                 vmin=0.0, vmax=1.0)
    plot_heatmap(Zm_n, k_grid, omega_grid,
                 title=f"Model {MODE} (normalized)", fname="heatmap_model_normalized",
                 vmin=0.0, vmax=1.0)
    plot_heatmap(diff_n, k_grid, omega_grid,
                 title="Difference (Model - Data, normalized)", fname="heatmap_difference_normalized")

    # 5) cuts (each curve-set → one figure, but no subplots)
    idxs_k, idxs_w = k_omega_indices(k_grid, omega_grid)

    # k-cuts at several omegas (one figure total; still单图—多条曲线是允许的)
    y_data_list = []
    y_model_list = []
    labs = []
    for oi in idxs_w:
        y_data_list.append(dataZ_n[oi, :])
        y_model_list.append(Zm_n[oi, :])
        labs.append(f"w={omega_grid[oi]:.4g} Data")
        labs.append(f"w={omega_grid[oi]:.4g} Model")
    # interleave pairs for legend clarity
    ys, ls = [], []
    for d, m, labd, labm in zip(y_data_list, y_model_list, labs[::2], labs[1::2]):
        ys += [d, m]
        ls += [labd, labm]
    plot_line_xy(k_grid, ys, ls, title="k-cuts (normalized)", xlabel="k", ylabel="Power",
                 fname="cuts_k_multi_omega")

    # omega-cuts at several ks
    y_data_list = []
    y_model_list = []
    labs = []
    for kj in idxs_k:
        y_data_list.append(dataZ_n[:, kj])
        y_model_list.append(Zm_n[:, kj])
        labs.append(f"k={k_grid[kj]:.4g} Data")
        labs.append(f"k={k_grid[kj]:.4g} Model")
    ys, ls = [], []
    for d, m, labd, labm in zip(y_data_list, y_model_list, labs[::2], labs[1::2]):
        ys += [d, m]
        ls += [labd, labm]
    plot_line_xy(omega_grid, ys, ls, title="omega-cuts (normalized)", xlabel="f (c/P)", ylabel="Power",
                 fname="cuts_omega_multi_k")

    # 6) integrated profiles (two separate figures)
    # integrate over k -> vs omega
    data_Ik = np.trapz(dataZ_n, k_grid, axis=1)
    model_Ik = np.trapz(Zm_n, k_grid, axis=1)
    plot_line_xy(omega_grid, [data_Ik, model_Ik], ["Data", "Model"],
                 title="Integrated over k (vs omega, normalized)", xlabel="f (c/P)", ylabel="Integrated power",
                 fname="integrated_over_k", xlim=(0.430, 0.450))

    # integrate over omega -> vs k
    data_Iw = np.trapz(dataZ_n, omega_grid, axis=0)
    model_Iw = np.trapz(Zm_n, omega_grid, axis=0)
    plot_line_xy(k_grid, [data_Iw, model_Iw], ["Data", "Model"],
                 title="Integrated over omega (vs k, normalized)", xlabel="k", ylabel="Integrated power",
                 fname="integrated_over_omega")

    # 7) scatter & residuals (based on normalized arrays)
    D = dataZ_n.ravel()
    M = Zm_n.ravel()
    plot_scatter(D, M, title="Scatter (normalized)", xlabel="Data", ylabel="Model",
                 fname="scatter_normalized")

    R = (M - D)
    plot_hist(R, bins=80, title="Residual histogram (normalized)", xlabel="Residual", ylabel="Count",
              fname="residual_hist_normalized")

    print("All SVG figures generated ✅")

if __name__ == "__main__":
    main()
