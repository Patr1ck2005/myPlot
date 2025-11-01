import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from matplotlib import cm


# =========================
# 绘图函数（均“独立成图”）
# =========================

def _wrap_angle_pi(phi):
    """把取向角规约到 (-pi/2, pi/2]，避免跨分支时跳变过大。"""
    return (phi + np.pi/2) % np.pi - np.pi/2

def plot_polarization_ellipses(ax, xgrid, ygrid, s1, s2, s3, S0=None,
                               step=(6, 6), scale=None,
                               cmap='RdBu', clim=(-1.0, 1.0),
                               lw=1.2, alpha=0.9, zorder=2):
    """
    椭圆贴片表示偏振场；颜色按 S3/S0（RdBu）。
    依然接收 ax，便于链式；但示例中每次单独成图。
    """
    phi = _wrap_angle_pi(0.5*np.arctan2(s2, s1))
    chi = 0.5*np.arcsin(np.clip(s3, -1.0, 1.0))

    xg = xgrid[:, 0]
    yg = ygrid[0, :]
    dx = np.min(np.diff(xg)) if xg.size > 1 else 1.0
    dy = np.min(np.diff(yg)) if yg.size > 1 else 1.0
    if scale is None:
        scale = 0.8 * min(dx, dy)

    sx, sy = step if isinstance(step, (tuple, list)) else (step, step)
    xs = range(0, xg.size, max(1, int(sx)))
    ys = range(0, yg.size, max(1, int(sy)))

    patches, color_vals = [], []
    for j in ys:
        for i in xs:
            a = scale
            b = max(scale * abs(np.tan(chi[i, j])), 1e-3*scale)
            e = Ellipse((xg[i], yg[j]), width=a, height=b,
                        angle=np.degrees(phi[i, j]))
            patches.append(e)
            color_vals.append(s3[i, j])

    pc = PatchCollection(patches, linewidths=lw, alpha=alpha, zorder=zorder)
    cmap_obj = cm.get_cmap(cmap)
    vmin, vmax = clim
    normed = (np.asarray(color_vals) - vmin) / (vmax - vmin + 1e-12)
    facecolors = cmap_obj(np.clip(normed, 0, 1))
    pc.set_facecolor(facecolors)
    pc.set_edgecolor(facecolors)

    ax.add_collection(pc)
    ax.set_aspect('equal', adjustable='box')
    return ax

def imshow_phi(ax, phi, extent=None, vmin=-np.pi/2, vmax=np.pi/2, **imshow_kwargs):
    im = ax.imshow(phi.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, **imshow_kwargs)
    ax.set_xlabel(r'$k_x$'); ax.set_ylabel(r'$k_y$')
    ax.set_title(r'Orientation angle $\phi$')
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax

def imshow_s3(ax, s3, S0=None, extent=None, vmin=-1.0, vmax=1.0, **imshow_kwargs):
    im = ax.imshow(s3.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, **imshow_kwargs)
    ax.set_xlabel(r'$k_x$'); ax.set_ylabel(r'$k_y$')
    ax.set_title(r'$S_3$ (helicity)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax

def plot_skyrmion_quiver(ax, x, y, s1, s2, s3, S0=None,
                         step=(4,4), normalize=True,
                         cmap='RdBu', clim=(-1,1),
                         quiver_scale=None, pivot='mid', width=0.004):
    """
    用箭头展示斯格明子纹理：U,V 为 (S1,S2) 的（归一化）平面分量；
    箭头颜色按 S3/S0（RdBu）。支持网格抽样 step=(sx,sy)。
    """

    # 网格
    xg = x[:, 0]; yg = y[0, :]
    sx, sy = step
    ii = np.arange(0, xg.size, max(1, int(sx)))
    jj = np.arange(0, yg.size, max(1, int(sy)))
    X, Y = np.meshgrid(xg[ii], yg[jj], indexing='ij')
    U, V, C = s1[jj][:, ii], s2[jj][:, ii], s3[jj][:, ii]

    if normalize:
        mag = np.hypot(U, V) + 1e-12
        U, V = U/mag, V/mag

    # quiver 支持 C 作为 colormap 输入
    q = ax.quiver(X, Y, U, V, C, cmap=cmap, clim=clim, pivot=pivot,
                  angles='xy', scale_units='xy', scale=quiver_scale, width=width)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$k_x$'); ax.set_ylabel(r'$k_y$')
    ax.set_title('Skyrmion-like texture: $(S_1,S_2)$ arrows, color by $S_3/S_0$')
    cb = plt.colorbar(q, ax=ax, shrink=0.8, label=r'$S_3/S_0$')
    return ax

import numpy as np
import matplotlib as mpl

def plot_on_poincare_sphere(ax, s1, s2, s3, S0=None,
                            step=(4,4), c_by='s3', cmap='RdBu', clim=(-1,1),
                            s=6, alpha=0.9, sphere_style='wire'):
    """
    把 (s1,s2,s3) 投到 Poincaré 球面；采样后全部扁平化为 1D 传给 scatter。
    - c_by: 's3' 用 s3 上色；'phi' 用 φ 上色
    """

    # 采样（注意 step=(sy,sx) 或 (sx,sx) 都支持）
    sy, sx = step if isinstance(step, (tuple, list)) else (step, step)
    yy = np.arange(0, s1.shape[0], max(1, int(sy)))
    xx = np.arange(0, s1.shape[1], max(1, int(sx)))
    s1s, s2s, s3s = s1[yy][:, xx], s2[yy][:, xx], s3[yy][:, xx]

    # 背景球
    u = np.linspace(0, 2*np.pi, 120)
    v = np.linspace(0, np.pi, 60)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))
    if sphere_style == 'surface':
        ax.plot_surface(X, Y, Z, rstride=4, cstride=4, color='lightgray', alpha=0.15, linewidth=0)
    else:
        ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, color='lightgray', linewidth=0.5, alpha=0.6)

    # —— 扁平化 ——
    Xp = s1s.ravel()
    Yp = s2s.ravel()
    Zp = s3s.ravel()

    # 颜色：传标量 + cmap（推荐做法，避免 RGBA 维度错位）
    if c_by.lower() == 'phi':
        PHI = (0.5*np.arctan2(s2s, s1s)) % np.pi
        Cp = PHI.ravel()
        sc = ax.scatter(Xp, Yp, Zp, s=s, c=Cp, cmap='twilight',
                        vmin=0.0, vmax=np.pi, alpha=alpha, depthshade=False)
        # cb = mpl.pyplot.colorbar(sc, ax=ax, shrink=0.8, label=r'$\phi$')
    else:
        Cp = np.clip(s3s, -1, 1).ravel()
        sc = ax.scatter(Xp, Yp, Zp, s=s, c=Cp, cmap=cmap,
                        vmin=clim[0], vmax=clim[1], alpha=alpha, depthshade=False)
        # cb = mpl.pyplot.colorbar(sc, ax=ax, shrink=0.8, label=r'$S_3/S_0$')

    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('$S_1/S_0$'); ax.set_ylabel('$S_2/S_0$'); ax.set_zlabel('$S_3/S_0$')
    return ax

