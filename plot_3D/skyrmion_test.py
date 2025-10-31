# Compute and visualize the skyrmion number for a standard 2D skyrmion
import numpy as np
import matplotlib.pyplot as plt

from utils.advanced_color_mapping import map_s1s2s3_color

# ----------------------
# 1) Build a "standard" skyrmion texture on a square grid
#    S = (cos α(φ) sin β(r), sin α(φ) sin β(r), cos β(r))
#    with β(0)=π, β(R)=0  (south pole at center -> north pole at boundary) gives s ≈ +1
# ----------------------
N = 251              # grid size (odd is convenient so (0,0) is a grid point)
L = 10.0             # half-size of the box in "length units"
R = 6.0              # skyrmion radius (core ~ center to where β~π/2)
q = 1                # vorticity / winding of α
gamma = 0.0          # helicity offset

# grid and spacing
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
dx = x[1]-x[0]
dy = y[1]-y[0]
X, Y = np.meshgrid(x, y, indexing='xy')

# polar coordinates
r = np.hypot(X, Y)
phi = np.arctan2(Y, X)

# smooth monotone profile β(r): β(0)=π, β(R)=0, constant 0 beyond R
t = np.clip(r/R, 0.0, 1.0)
# C^1 smoothstep (3t^2 - 2t^3); could use other monotone profiles as well
smooth_t = 3*t**2 - 2*t**3
beta = np.pi * (1.0 - smooth_t)     # β: from π (center) down to 0 (boundary/outside)
alpha = q*phi + gamma

S1 = np.cos(alpha) * np.sin(beta)
S2 = np.sin(alpha) * np.sin(beta)
S3 = np.cos(beta)

# stack into S(x,y) field
S = np.stack([S1, S2, S3], axis=0)

# ----------------------
# 2) Compute skyrmion density n_sk = S · (∂x S × ∂y S)
#    Use centered finite differences via np.gradient with physical spacings
# ----------------------
# gradients of each component w.r.t x and y
dS1dy, dS1dx = np.gradient(S1, dy, dx, edge_order=2)
dS2dy, dS2dx = np.gradient(S2, dy, dx, edge_order=2)
dS3dy, dS3dx = np.gradient(S3, dy, dx, edge_order=2)

# cross product ∂x S × ∂y S (component-wise formula)
cx = dS2dx * dS3dy - dS3dx * dS2dy
cy = dS3dx * dS1dy - dS1dx * dS3dy
cz = dS1dx * dS2dy - dS2dx * dS1dy

nsk = S1*cx + S2*cy + S3*cz   # skyrmion density

# ----------------------
# 3) Integrate to get skyrmion number s = (1/4π) ∬ n_sk dx dy
# ----------------------
s = (nsk.sum() * dx * dy) / (4.0*np.pi)

print(f"Computed skyrmion number s ≈ {s:.6f}")
print(f"Grid: {N}x{N}, box size: [-{L}, {L}]^2, dx=dy={dx:.4f}, radius R={R}")

# ----------------------
# 4) Visualizations (separate figures; no style or color specified)
# ----------------------

# (a) Map of S3 (out-of-plane component)
plt.figure()
plt.imshow(S3, extent=[-L, L, -L, L], origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='S3')
plt.title('S3 (out-of-plane)')
plt.show()

# (b) Quiver of in-plane components (subsampled for clarity)
step = 6
plt.figure()
plt.quiver(X[::step, ::step], Y[::step, ::step],
           S1[::step, ::step], S2[::step, ::step], angles='xy', scale_units='xy', scale=1)
plt.title('In-plane field (S1, S2)')
plt.xlim(-L, L); plt.ylim(-L, L)
plt.show()

rgb = map_s1s2s3_color(S1, S2, S3, s3_mode='-11', show=False, extent=[-L, L, -L, L])
fig = plt.figure(figsize=(3, 3), dpi=100)
plt.imshow(rgb, origin='lower', extent=[-L, L, -L, L])
plt.xlabel(r'$k_x (2\pi/a)$')
plt.ylabel(r'$k_y (2\pi/a)$')
plt.tight_layout()
plt.show()

# (c) Skyrmion density n_sk
plt.figure()
plt.imshow(nsk, extent=[-L, L, -L, L], origin='lower')
plt.colorbar(label='n_sk')
plt.title('Skyrmion density n_sk')
plt.show()

# (d) Cumulative radial integral check (optional sanity visualization)
#     Integrate n_sk inside a circle of radius ρ; should approach 4π*s as ρ->∞
#     Numerically we show partial skyrmion number s(ρ).
rho_vals = np.linspace(0.0, L, 200)
s_partial = np.zeros_like(rho_vals)
# precompute mask per radius
for i, rho in enumerate(rho_vals):
    mask = (r <= rho)
    area = mask.sum() * dx * dy
    s_partial[i] = (nsk[mask].sum() * dx * dy) / (4.0*np.pi)

plt.figure()
plt.plot(rho_vals, s_partial)
plt.axhline(1.0, linestyle='--')
plt.xlabel('radius ρ')
plt.ylabel('partial skyrmion number s(ρ)')
plt.title('Convergence of skyrmion number vs radius')
plt.show()

