import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------- model ----------
def H_full(S,kappa,omega3,delta_w,delta_g):
    theta = 2*np.pi*S
    dt = delta_w - 1j*delta_g
    return np.array([
        [1-1j*0.12+dt, 0, kappa*np.cos(theta)],
        [0, 1-1j*0.12-dt, kappa*np.sin(theta)],
        [kappa*np.cos(theta), kappa*np.sin(theta), omega3]
    ],dtype=complex)

def eig_sorted(H):
    vals, vecs = np.linalg.eig(H)
    idx = np.argsort(vals.real)
    return vals[idx], vecs[:,idx]

# ---------- Bloch mapping ----------
def bloch(vec):
    v = vec[:2] / np.linalg.norm(vec[:2])
    a,b = v
    S1 = 2*np.real(a*np.conj(b))
    S2 = 2*np.imag(a*np.conj(b))
    S3 = np.abs(a)**2 - np.abs(b)**2
    return np.array([S1,S2,S3])

# ---------- figure ----------
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

S_vals = np.linspace(0,1,200)

colors = ['r','b','g']

def compute(branch,kappa,omega3,dw,dg):
    traj=[]
    gap=[]
    for S in S_vals:
        H = H_full(S,kappa,omega3,dw,dg)
        vals,vecs = eig_sorted(H)
        v = vecs[:,branch]
        traj.append(bloch(v))
        gap.append(min(abs(vals[0]-vals[1]),
                       abs(vals[0]-vals[2]),
                       abs(vals[1]-vals[2])))
    return np.array(traj),np.array(gap)

# sliders
ax_k = plt.axes([0.2,0.15,0.6,0.03])
ax_dw = plt.axes([0.2,0.10,0.6,0.03])
ax_dg = plt.axes([0.2,0.05,0.6,0.03])

sk = Slider(ax_k,'kappa',0.01,0.6,valinit=0.25)
sdw = Slider(ax_dw,'dw',0,0.2,valinit=0.05)
sdg = Slider(ax_dg,'dg',0,0.2,valinit=0.03)

def update(val):
    ax.cla()
    kappa = sk.val
    dw = sdw.val
    dg = sdg.val
    omega3=1.02

    for i,c in enumerate(colors):
        traj,gap = compute(i,kappa,omega3,dw,dg)

        # EP proximity → brightness
        w = 1/(gap+1e-3)
        w = (w-w.min())/(w.max()-w.min()+1e-6)

        ax.plot(traj[:,0],traj[:,1],traj[:,2],
                color=c,alpha=0.8)

        ax.scatter(traj[:,0],traj[:,1],traj[:,2],
                   c=w,cmap='plasma',s=5)

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_title("Three-branch pseudospin texture")
    plt.draw()

sk.on_changed(update)
sdw.on_changed(update)
sdg.on_changed(update)

update(None)
plt.show()
