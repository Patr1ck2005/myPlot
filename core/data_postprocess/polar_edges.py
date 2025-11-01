# polar_edges_split.py
# -*- coding: utf-8 -*-
import numpy as np

# ------------------ 基础 ------------------
def _wrap_0_pi(phi):
    return np.mod(np.asarray(phi), np.pi)

def _cell_lefts(g):
    g = np.asarray(g)
    lefts = np.empty_like(g)
    lefts[1:] = (g[:-1] + g[1:]) / 2
    lefts[0]  = g[0] - (g[1] - g[0]) / 2
    return lefts

def _bilinear(xg, yg, F, xq, yq):
    xg, yg, F = map(np.asarray, (xg, yg, F))
    xq, yq    = np.asarray(xq), np.asarray(yq)
    nx, ny = len(xg), len(yg)
    assert F.shape == (nx, ny), f"F.shape={F.shape} != {(nx,ny)}"
    xl, yl = _cell_lefts(xg), _cell_lefts(yg)
    i = np.clip(np.searchsorted(xl, xq, 'right')-1, 0, nx-2)
    j = np.clip(np.searchsorted(yl, yq, 'right')-1, 0, ny-2)
    x0,x1,y0,y1 = xg[i],xg[i+1],yg[j],yg[j+1]
    tx = (xq-x0)/(x1-x0+1e-12); ty=(yq-y0)/(y1-y0+1e-12)
    f00,f10,f01,f11 = F[i,j],F[i+1,j],F[i,j+1],F[i+1,j+1]
    return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11

def _gradient_central(xg, yg, F):
    F = np.asarray(F, float)
    Gx = np.empty_like(F); Gy = np.empty_like(F)
    Gx[1:-1,:] = (F[2:,:]-F[:-2,:])/(xg[2:]-xg[:-2])[:,None]
    Gx[0,:]    = (F[1,:]-F[0,:])/(xg[1]-xg[0]+1e-12)
    Gx[-1,:]   = (F[-1,:]-F[-2,:])/(xg[-1]-xg[-2]+1e-12)
    Gy[:,1:-1] = (F[:,2:]-F[:,:-2])/(yg[2:]-yg[:-2])[None,:]
    Gy[:,0]    = (F[:,1]-F[:,0])/(yg[1]-yg[0]+1e-12)
    Gy[:,-1]   = (F[:,-1]-F[:,-2])/(yg[-1]-yg[-2]+1e-12)
    return Gx, Gy

# ------------------ marching squares: F=0 ------------------
def _ms_zero_isolines(xg, yg, F):
    xg, yg, F = np.asarray(xg), np.asarray(yg), np.asarray(F, float)
    nx, ny = F.shape
    assert (nx, ny) == (len(xg), len(yg)), "F shape 与网格不符"

    def interp(p0, p1, v0, v1):
        den = (v1 - v0)
        t = 0.5 if den == 0 else (0.0 - v0) / den
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return p0 + t*(p1 - p0)

    table = {
        0:[],1:[(3,0)],2:[(0,1)],3:[(3,1)],
        4:[(1,2)],5:[(3,2),(0,1)],6:[(0,2)],7:[(3,2)],
        8:[(2,3)],9:[(0,2)],10:[(1,3),(0,2)],11:[(1,2)],
        12:[(1,3)],13:[(0,1)],14:[(3,0)],15:[]
    }

    segs = []
    for i in range(nx-1):
        x0,x1 = xg[i], xg[i+1]
        for j in range(ny-1):
            y0,y1 = yg[j], yg[j+1]
            f00,f10,f11,f01 = F[i,j],F[i+1,j],F[i+1,j+1],F[i,j+1]
            c0 = 1 if f00 >= 0 else 0
            c1 = 1 if f10 >= 0 else 0
            c2 = 1 if f11 >= 0 else 0
            c3 = 1 if f01 >= 0 else 0
            code = (c0 | (c1<<1) | (c2<<2) | (c3<<3))
            if code in (5,10):
                fcen = 0.25*(f00+f10+f11+f01)
                if (code==5 and fcen<0) or (code==10 and fcen>=0):
                    code = 5 if code==10 else 10
            if not table[code]:
                continue
            E = {
                0: ((np.array([x0,y0]), np.array([x1,y0])), (f00,f10)),
                1: ((np.array([x1,y0]), np.array([x1,y1])), (f10,f11)),
                2: ((np.array([x0,y1]), np.array([x1,y1])), (f01,f11)),
                3: ((np.array([x0,y0]), np.array([x0,y1])), (f00,f01)),
            }
            for e0,e1 in table[code]:
                (p0a,p0b),(v0a,v0b) = E[e0]
                (p1a,p1b),(v1a,v1b) = E[e1]
                P0 = interp(p0a,p0b,v0a,v0b)
                P1 = interp(p1a,p1b,v1a,v1b)
                segs.append((tuple(P0), tuple(P1)))

    if not segs: return []

    # 串联
    tol = 1e-12
    def key(pt):
        if pt[0] != np.nan and pt[1] != np.nan:
            return pt
        return round(pt[0]/tol)*tol, round(pt[1]/tol)*tol
    start_map, end_map = {}, {}
    for a,b in segs:
        start_map.setdefault(key(a), []).append((a,b))
        end_map.setdefault(key(b), []).append((a,b))

    used=set(); lines=[]
    for seg in segs:
        if seg in used: continue
        a,b = seg; used.add(seg); line=[a,b]
        cur=b
        while True:
            lst = start_map.get(key(cur), []); nxt=None
            for cand in lst:
                if cand in used: continue
                if np.allclose(cand[0], cur, atol=1e-9, rtol=0): nxt=cand; break
            if nxt is None: break
            used.add(nxt); line.append(nxt[1]); cur=nxt[1]
        cur=a
        while True:
            lst = end_map.get(key(cur), []); prv=None
            for cand in lst:
                if cand in used: continue
                if np.allclose(cand[1], cur, atol=1e-9, rtol=0): prv=cand; break
            if prv is None: break
            used.add(prv); line.insert(0, prv[0]); cur=prv[0]
        lines.append(np.asarray(line))
    return lines

# ------------------ 主逻辑：切分+分类 ------------------
def _split_and_classify_on_skeleton(xg, yg, PHI, skeleton_lines,
                                    h_factor=0.25, tol_c2=1e-12, min_pts=2):
    """
    在 sin(2φ)=0 骨架上，用法向两侧的 cos(2φ) 决定每个顶点的类别，
    然后按类别连续性切分为子段并归类。
    返回 {'phi0':[poly...], 'phi90':[poly...], 'uncertain':[poly...]}。
    """
    xg, yg = np.asarray(xg), np.asarray(yg)
    PHI = np.asarray(PHI)
    s2  = np.sin(2*PHI)
    c2  = np.cos(2*PHI)
    Gx, Gy = _gradient_central(xg, yg, s2)

    dx = np.diff(xg).min() if len(xg)>1 else 1.0
    dy = np.diff(yg).min() if len(yg)>1 else 1.0
    h  = h_factor * min(dx, dy)

    out = {'phi0':[], 'phi90':[], 'uncertain':[]}

    for P in skeleton_lines:
        P = np.asarray(P);
        if P.shape[0] < min_pts:
            continue

        # 法向（取 ∇s2 方向）
        gx = _bilinear(xg, yg, Gx, P[:,0], P[:,1])
        gy = _bilinear(xg, yg, Gy, P[:,0], P[:,1])
        nrm = np.hypot(gx, gy) + 1e-15
        nx, ny = gx/nrm, gy/nrm

        # 两侧采样 cos(2φ)
        c0 = _bilinear(xg, yg, c2, P[:,0], P[:,1])
        cp = _bilinear(xg, yg, c2, P[:,0] + h*nx, P[:,1] + h*ny)
        cm = _bilinear(xg, yg, c2, P[:,0] - h*nx, P[:,1] - h*ny)

        # 选更“远离0”的一侧作为该点类别符号
        abs_p, abs_m = np.abs(cp), np.abs(cm)
        pick_p = abs_p >= abs_m
        sign_local = np.sign(np.where(pick_p, cp, cm))

        # 数值不可靠位置（两侧都接近0）
        unreliable = (abs_p < tol_c2) & (abs_m < tol_c2)
        sign_local[unreliable] = 0.0  # 记为不确定

        # ---- 按 sign_local 分段切割 ----
        def flush(segment, label):
            if len(segment) >= min_pts:
                seg = np.vstack(segment)
                if label > 0:   out['phi0'     ].append(seg)  # cos(2φ)>0 → φ≈0/π
                elif label < 0: out['phi90'    ].append(seg)  # cos(2φ)<0 → φ≈π/2
                else:           out['uncertain'].append(seg)

        cur_seg = [P[0]]
        cur_lab = sign_local[0]
        for k in range(1, P.shape[0]):
            lab = sign_local[k]
            if lab == cur_lab or (lab==0 and cur_lab!=0):
                # 同类延续；或者当前点不确定但段落有确定标签 → 继续
                cur_seg.append(P[k])
            else:
                # 标签变化：结束上一段
                flush(cur_seg, cur_lab)
                cur_seg = [P[k-1], P[k]]  # 从上一个点开始新的段，避免缝隙
                cur_lab = lab
        flush(cur_seg, cur_lab)

    return out

def extract_phi0_phi90_split(xgrid, ygrid, phi):
    """
    从 φ∈[0,π) 出发：
      1) 用 sin(2φ)=0 得到骨架折线；
      2) 在骨架上按 cos(2φ) 的法向两侧“远离0”的符号逐点分类；
      3) 按连续性切分成多条子曲线并归入 φ≈0/π 或 φ≈π/2。
    """
    xg, yg = np.asarray(xgrid), np.asarray(ygrid)
    PHI    = _wrap_0_pi(phi)
    s2     = np.sin(2*PHI)
    axes   = _ms_zero_isolines(xg, yg, s2)
    return _split_and_classify_on_skeleton(xg, yg, PHI, axes)


import numpy as np
import matplotlib.pyplot as plt

def plot_phi_families_split(ax, xgrid, ygrid, phi, *,
                            overlay=None, overlay_alpha=0.85,
                            color_phi0='limegreen', color_phi90='black', color_uncertain='orange',
                            lw=2.0):
    """
    在给定 Axes 上绘制两族结线（基于 extract_phi0_phi90_split 的逐点切分分类）。
    参数：
      overlay: None | 'phi' | 's2' | 'c2'
               'phi' → 以 φ∈[0,π) 上色（twilight）
               's2'  → 以 sin(2φ) 上色（RdBu）
               'c2'  → 以 cos(2φ) 上色（RdBu）
    """
    xg, yg = np.asarray(xgrid), np.asarray(ygrid)
    PHI    = np.mod(np.asarray(phi), np.pi)

    # 叠加底图（可选）
    if overlay == 'phi':
        img, cm, vmin, vmax = PHI, 'twilight', 0.0, np.pi
    elif overlay == 's2':
        img, cm, vmin, vmax = np.sin(2*PHI), 'RdBu', -1.0, 1.0
    elif overlay == 'c2':
        img, cm, vmin, vmax = np.cos(2*PHI), 'RdBu', -1.0, 1.0
    else:
        img = None

    if img is not None:
        ax.imshow(img.T, origin='lower',
                  extent=(xg[0], xg[-1], yg[0], yg[-1]),
                  aspect='auto', cmap=cm, vmin=vmin, vmax=vmax, alpha=overlay_alpha)

    # 提取并绘制
    res = extract_phi0_phi90_split(xg, yg, PHI)
    for P in res['phi0']:
        ax.plot(P[:,0], P[:,1], color=color_phi0, lw=lw, label=r'$\phi\approx 0,\pi$')
    for P in res['phi90']:
        ax.plot(P[:,0], P[:,1], color=color_phi90, lw=lw, label=r'$\phi\approx \pi/2$')
    for P in res['uncertain']:
        ax.plot(P[:,0], P[:,1], color=color_uncertain, lw=lw, ls='--', label='uncertain')

    ax.set_aspect('equal', adjustable='box')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    return ax
