# polar_postprocess.py
# -*- coding: utf-8 -*-
import os, pickle, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

# =========================
# 基础与对称小工具（复用）
# =========================
def _wrap_angle_pi(phi):
    """把 φ 规约到 (-π/2, π/2]（线偏振角模 π）。"""
    return (phi + np.pi/2) % np.pi - np.pi/2

def _axis_coords(arr):
    """坐标轴镜像：不重复 0 且保持升序。"""
    arr = np.asarray(arr)
    left = -arr[1:][::-1] if np.isclose(arr[0], 0) else -arr[::-1]
    return np.concatenate([left, arr], 0)

def _phi_mirror(phi):
    """轴镜像：φ' = -φ（mod π），用 wrap 收回到 (-π/2, π/2]。"""
    return _wrap_angle_pi(-_wrap_angle_pi(phi))


def _mirror_xy_data(Z):
    """对二维字段做 x、y 轴几何镜像（不含物理变换），得到全平面。"""
    Zx = np.concatenate([np.flip(Z[1: ], 0), Z], 0)
    Zy = np.concatenate([np.flip(Zx[:, 1:], 1), Zx], 1)
    return Zy

# =========================
# 仅几何补全（标量/复数场）
# =========================
def geom_complete(coords, Z, mode='xy'):
    """
    仅几何镜像/旋转，不做物理变换。
    mode: 'xy' | 'x' | 'y' | 'C4'（先 xy，再与 R90 平均）
    返回 (coords_out, Z_full)
    """
    keys = list(coords.keys())
    xk, yk = ('kx','ky') if {'kx','ky'}.issubset(coords) else (keys[0], keys[1])
    x, y, Z = np.asarray(coords[xk]), np.asarray(coords[yk]), np.asarray(Z)

    if mode == 'xy':
        x_full, y_full, Z_full = _axis_coords(x), _axis_coords(y), _mirror_xy_data(Z)
    elif mode == 'x':
        x_full, y_full = _axis_coords(x), y
        Z_full = np.concatenate([np.flip(Z[1:], 0), Z], 0)
    elif mode == 'y':
        x_full, y_full = x, _axis_coords(y)
        Z_full = np.concatenate([np.flip(Z[:,1:], 1), Z], 1)
    elif mode == 'C4':
        x_full, y_full, Z_full = _axis_coords(x), _axis_coords(y), _mirror_xy_data(Z)
        if Z_full.shape[0] == Z_full.shape[1]:
            Z_full = 0.5*(Z_full + np.rot90(Z_full, 1))
    else:
        raise ValueError(f"未知 mode: {mode}")

    out = dict(coords)
    out[xk], out[yk] = x_full, y_full
    return out, Z_full

# =========================
# 偏振补全（φ, χ）
# =========================
def _mirror_phi_chi_x(phi, chi):
    """ky→-ky 镜像：φ 用 2φ→π-2φ；χ 取反；列翻转对齐。"""
    return _phi_mirror(np.flip(phi[:,1:], 1)), -np.flip(chi[:,1:], 1)

def _mirror_phi_chi_y(phiR, chiR):
    """kx→-kx 镜像：φ 用 2φ→π-2φ；χ 取反；行翻转对齐。"""
    return _phi_mirror(np.flip(phiR[1:], 0)), -np.flip(chiR[1:], 0)

def complete_C4_polarization(kx, ky, phi_Q1, chi_Q1):
    """
    由第一象限 (phi, chi) 补全至全平面：
    """
    kx, ky = np.asarray(kx), np.asarray(ky)
    phi, chi = _wrap_angle_pi(np.asarray(phi_Q1)), np.asarray(chi_Q1)

    assert kx.ndim==ky.ndim==1 and np.all(np.diff(kx)>0) and np.all(np.diff(ky)>0)
    assert np.isclose(kx[0],0) and np.isclose(ky[0],0)
    assert phi.shape[:2]==(kx.size,ky.size) and chi.shape[:2]==(kx.size,ky.size)

    kx_full, ky_full = _axis_coords(kx), _axis_coords(ky)

    # 右半平面
    phi_RB, chi_RB = _mirror_phi_chi_x(phi, chi)
    phi_R = np.concatenate([phi_RB, phi], 1)
    chi_R = np.concatenate([chi_RB, chi], 1)
    # 左半平面
    phi_L, chi_L = _mirror_phi_chi_y(phi_R, chi_R)
    phi_full = np.concatenate([phi_L, phi_R], 0)
    chi_full = np.concatenate([chi_L, chi_R], 0)

    return kx_full, ky_full, _wrap_angle_pi(phi_full), chi_full

# =========================
# 序列化
# =========================
def from_legacy_and_save(pkl_path, m1, m2, Z_target_complex,
                         phi_Q1, tanchi_Q1, Q_Q1,
                         do_complete=True):
    m1, m2 = np.asarray(m1), np.asarray(m2)
    Zc, phi, chi, Q = np.asarray(Z_target_complex), np.asarray(phi_Q1), np.asarray(tanchi_Q1), np.asarray(Q_Q1)

    if do_complete:
        m1f, m2f, phi_f, chi_f = complete_C4_polarization(m1, m2, phi, chi)
        (_, _), Z_f = geom_complete({'m1':m1, 'm2':m2}, Zc, mode='C4')
        Q_R  = np.concatenate([np.flip(Q[:,1:],1), Q], 1)
        Q_f  = np.concatenate([np.flip(Q_R[1:],0), Q_R], 0)  # Q 偶（几何镜像）
    else:
        m1f, m2f, phi_f, chi_f, Z_f, Q_f = m1, m2, phi, chi, Zc, Q
    # 预览phi_f结果
    fig = plt.figure(figsize=(6,5))
    plt.imshow(phi_f.T, origin='lower', extent=(m1f[0], m1f[-1], m2f[0], m2f[-1]), aspect='auto', cmap='hsv')
    plt.colorbar(label='φ (rad)'); plt.title('Completed φ field preview'); plt.xlabel('m1'); plt.ylabel('m2')
    plt.show()
    bundle = dict(m1_full=m1f, m2_full=m2f, Z_full=Z_f, phi_full=phi_f, chi_full=chi_f, Q_full=Q_f)
    with open(pkl_path, 'wb') as f: pickle.dump(bundle, f)
    print(f"[SAVE] {pkl_path}")

def load_bundle(pkl_path):
    if not os.path.exists(pkl_path): raise FileNotFoundError(pkl_path)
    with open(pkl_path, 'rb') as f: bundle = pickle.load(f)
    need = ['m1_full','m2_full','Z_full','phi_full','chi_full','Q_full']
    for k in need:
        if k not in bundle: raise KeyError(f"pkl 缺少键：{k}")
    return bundle

# =========================
# 栅格插值（含角度）
# =========================
def _cell_lefts(g):
    g = np.asarray(g); d = np.diff(g)
    lefts = np.empty_like(g); lefts[1:] = (g[:-1]+g[1:])/2; lefts[0] = g[0]-d[0]/2
    return lefts

def _bilinear(xg, yg, F, xq, yq):
    xg, yg, F, xq, yq = map(np.asarray, (xg, yg, F, xq, yq))
    xl, yl = _cell_lefts(xg), _cell_lefts(yg)
    i = np.clip(np.searchsorted(xl, xq, 'right')-1, 0, xg.size-2)
    j = np.clip(np.searchsorted(yl, yq, 'right')-1, 0, yg.size-2)
    x0,x1,y0,y1 = xg[i],xg[i+1],yg[j],yg[j+1]
    tx = (xq-x0)/(x1-x0+1e-12); ty=(yq-y0)/(y1-y0+1e-12)
    f00,f10,f01,f11 = F[i,j],F[i+1,j],F[i,j+1],F[i+1,j+1]
    return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11

def _bilinear_phi(xg, yg, PHI, xq, yq):
    PHI = _wrap_angle_pi(PHI)
    c2, s2 = np.cos(2*PHI), np.sin(2*PHI)
    c2q, s2q = _bilinear(xg, yg, c2, xq, yq), _bilinear(xg, yg, s2, xq, yq)
    return 0.5*np.arctan2(s2q, c2q)

# =========================
# 等频线提取 & 采样
# =========================
def extract_isofreq_paths(xg, yg, FREQ, level):
    xg, yg, F = map(np.asarray, (xg, yg, FREQ))
    if F.shape != (xg.size, yg.size): raise ValueError("FREQ shape mismatch")
    if np.isnan(F).any(): F = np.ma.array(F, mask=np.isnan(F))
    fmin, fmax = float(np.nanmin(F)), float(np.nanmax(F))
    if not (fmin <= level <= fmax): return []
    if np.allclose(F, level): level = float(level)+1e-12

    X, Y = np.meshgrid(xg, yg, indexing='ij')
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(X, Y, F, levels=[level]); paths=[]
        if getattr(cs, "allsegs", None):
            for seg in cs.allsegs[0]:
                seg = np.asarray(seg)
                if seg.ndim==2 and seg.shape[1]==2 and seg.shape[0]>=2: paths.append(seg.copy())
    finally:
        plt.close(fig)
    return paths

def _resample_uniform(path_xy, step=None, npts=400):
    P = np.asarray(path_xy); d = np.sqrt(np.sum(np.diff(P,0)**2,1))
    s = np.concatenate([[0], np.cumsum(d)])
    s_new = np.linspace(0, s[-1], max(2, int(math.ceil(s[-1]/step))+1) if step else npts)
    x, y = np.interp(s_new, s, P[:,0]), np.interp(s_new, s, P[:,1])
    return np.stack([x,y],1), s_new

def sample_fields_along_path(xg, yg, fields, path_xy, step=None, npts=400):
    xy, s = _resample_uniform(path_xy, step=step, npts=npts)
    xq, yq = xy[:,0], xy[:,1]
    out = {'s': s, 'x': xq, 'y': yq}
    for name, arr in fields.items():
        out[name] = _bilinear_phi(xg, yg, arr, xq, yq) if name.lower() in {'phi','phi_full','angle','polar_angle'} \
                    else _bilinear(xg, yg, arr, xq, yq)
    return out

# =========================
# 绘图
# =========================
def plot_band_surface_with_overlay(m1, m2, FREQ, color_field, cmap='twilight', title=None, zlabel='Frequency'):
    M1, M2 = np.meshgrid(m1, m2, indexing='ij')
    cf = color_field
    norm = (cf - np.nanmin(cf)) / (np.nanmax(cf) - np.nanmin(cf) + 1e-12)
    colors = plt.cm.get_cmap(cmap)(norm)
    fig = plt.figure(figsize=(3.6, 4.2)); ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(M1, M2, FREQ, facecolors=colors, rstride=1, cstride=1, alpha=0.9)
    m = plt.cm.ScalarMappable(cmap=cmap); m.set_array(cf); plt.colorbar(m, ax=ax, shrink=0.75)
    ax.set(xlabel='m1', ylabel='m2', zlabel=zlabel);
    if title: ax.set_title(title)
    ax.view_init(elev=30, azim=25); ax.set_box_aspect([1,1,1]); plt.tight_layout()
    return fig, ax

def plot_lines_along_isofreq(samples_list, labels=None):
    labels = labels or [f"path {i+1}" for i in range(len(samples_list))]
    fig, axs = plt.subplots(3, 1, figsize=(6.2, 6.5), sharex=True)
    for samp, lab in zip(samples_list, labels):
        s = samp['s']
        if 'phi' in samp: axs[0].plot(s, np.mod(samp['phi'], np.pi), lw=1.5, label=lab)
        elif 'phi_full' in samp: axs[0].plot(s, np.mod(samp['phi_full'], np.pi), lw=1.5, label=lab)
        if 'chi' in samp: axs[1].plot(s, samp['chi'], lw=1.5, label=lab)
        if 'Q' in samp:   axs[2].plot(s, samp['Q'],   lw=1.5, label=lab)
    axs[0].set_ylabel('phi (mod π)'); axs[1].set_ylabel('tanchi'); axs[2].set(ylabel='Q', xlabel='arc length s')
    for ax in axs: ax.grid(True, alpha=0.3); ax.legend(frameon=False)
    plt.tight_layout(); return fig, axs

def load_and_plot(pkl_path, isofreq, color='phi', cmap='twilight', n_contour_pts=500):
    data = load_bundle(pkl_path)
    m1, m2 = data['m1_full'], data['m2_full']
    Z, FREQ = data['Z_full'], np.real(data['Z_full'])
    phi, chi, Q = data['phi_full'], data['chi_full'], data['Q_full']

    if color == 'tanchi':   cf, cm_use, title = chi, 'RdBu', 'Band with tanchi color'
    elif color == 'Q':      cf, cm_use, title = Q, 'hot',   'Band with Q color'
    else:                   cf, cm_use, title = np.mod(phi, np.pi), (cmap or 'twilight'), 'Band with φ color'

    plot_band_surface_with_overlay(m1, m2, FREQ, cf, cmap=cm_use, title=title)
    paths = extract_isofreq_paths(m1, m2, FREQ, level=isofreq)
    if not paths:
        print(f"[WARN] isofreq={isofreq} 未找到等频线。范围 [{np.nanmin(FREQ):.4g}, {np.nanmax(FREQ):.4g}]"); return

    fields = {'phi': phi, 'chi': chi, 'Q': Q, 'freq': FREQ}
    samples = [sample_fields_along_path(m1, m2, fields, p, npts=n_contour_pts) for p in paths]
    plot_lines_along_isofreq(samples, labels=[f'Γ{i+1}' for i in range(len(samples))]); plt.show()
    return dict(paths=paths, samples=samples)

def plot_isofreq_contours2D(ax, xg, yg, FREQ, levels, colors=None, linewidths=1.8, linestyles='-', return_paths=True, **kw):
    xg, yg, F = map(np.asarray, (xg, yg, FREQ))
    if F.shape != (xg.size, yg.size): raise ValueError("FREQ shape mismatch")
    if np.isnan(F).any(): F = np.ma.array(F, mask=np.isnan(F))
    X, Y = np.meshgrid(xg, yg, indexing='ij')
    cs = ax.contour(X, Y, F, levels=np.atleast_1d(levels), colors=colors, linewidths=linewidths, linestyles=linestyles, **kw)
    if not return_paths: return cs
    paths = {}
    for k, lev in enumerate(np.atleast_1d(levels)):
        segs = [np.asarray(seg).copy() for seg in getattr(cs,'allsegs',[[]])[k] if np.asarray(seg).ndim==2]
        paths[float(lev)] = [s for s in segs if s.shape[0]>=2 and s.shape[1]==2]
    return cs, paths

def plot_paths(ax, paths, colors='C0', linewidth=2.0, alpha=1.0, labels=None, zorder=3):
    lines = []; all_paths = []
    if isinstance(paths, dict):
        for plist in paths.values(): all_paths += plist
    else:
        all_paths = list(paths)
    color_list = list(colors) if isinstance(colors,(list,tuple,np.ndarray)) else None
    for i,p in enumerate(all_paths):
        c = color_list[i] if color_list and i<len(color_list) else colors
        lab = labels[i] if (labels and i<len(labels)) else None
        ln, = ax.plot(p[:,0], p[:,1], color=c, lw=linewidth, alpha=alpha, label=lab, zorder=zorder); lines.append(ln)
    return lines

def plot_polarization_ellipses(ax, xg, yg, phi, chi, step=(2,2), scale=None, color_by='chi', cmap=None, clim=None, edgecolor='k', lw=2, alpha=0.9, zorder=2):
    xg, yg = np.asarray(xg), np.asarray(yg)
    PHI, CHI = _wrap_angle_pi(np.asarray(phi)), np.asarray(chi)
    assert PHI.shape == CHI.shape == (xg.size, yg.size)
    dx = np.min(np.diff(xg)) if xg.size>1 else 1.0
    dy = np.min(np.diff(yg)) if yg.size>1 else 1.0
    scale = scale or (0.8*min(dx,dy))
    xs, ys = range(0, xg.size, max(1,int(step[0]))), range(0, yg.size, max(1,int(step[1])))

    patches, vals = [], []
    for i in xs:
        for j in ys:
            phi_ij, chi_ij = PHI[i,j], float(np.clip(CHI[i,j], -1, 1))
            eps = 0.5*np.arcsin(chi_ij); a = scale; b = max(scale*abs(np.tan(eps)), 1e-3*scale)
            patches.append(Ellipse((xg[i], yg[j]), width=a, height=b, angle=np.degrees(phi_ij)))
            vals.append((chi_ij if color_by=='chi' else (phi_ij+np.pi/2)%np.pi) if color_by in {'chi','phi'} else 0.0)

    pc = PatchCollection(patches, facecolor='none', edgecolor=edgecolor, linewidths=lw, alpha=alpha, zorder=zorder)
    if color_by in {'chi','phi'}:
        import matplotlib.cm as cm
        cmap = cmap or ('RdBu' if color_by=='chi' else 'twilight')
        vmin,vmax = (-1,1) if color_by=='chi' else (0,np.pi)
        if clim is not None: vmin,vmax = clim
        arr = np.asarray(vals); normed = (arr - vmin) / (vmax - vmin + 1e-12)
        facecolors = cm.get_cmap(cmap)(np.clip(normed,0,1))
        pc.set_facecolor(facecolors); pc.set_edgecolor(facecolors)
    ax.add_collection(pc); ax.set_aspect('equal', adjustable='box'); return pc
