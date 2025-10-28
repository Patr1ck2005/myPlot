import matplotlib.pyplot as plt

from polar_postprocess import *


fs = 9
plt.rcParams.update({"font.size": fs})

# 假设已加载数据：
data = load_bundle('rsl/eigensolution/polar_fields.pkl')
m1, m2 = data['m1_full'], data['m2_full']
FREQ = np.real(data['Z_full'])
phi = data['phi_full']
chi = data['chi_full']

fig, ax = plt.subplots(figsize=(3/2, 3/2))

# 1) 先画偏振椭圆背景
plot_polarization_ellipses(
    ax, m1, m2, phi, chi,
    step=(5, 5),  # 适当抽样，防止太密
    scale=1e-2,  # 自动用 0.8*min(dx,dy)
    color_by='phi', cmap='hsv',
    alpha=1, lw=1,
)

# 2) 叠加三条等频线
levels = [0.65, 0.655, 0.66]  # 你想观察的 isofreq
cs, paths_dict = plot_isofreq_contours2D(ax, m1, m2, FREQ, levels=levels,
                                         colors=['k', 'k', 'k'],
                                         linewidths=1.0)

# ax.set_xlabel('m1')
# ax.set_ylabel('m2')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.savefig('./temp_polar_ellipses_with_isofreq.svg', bbox_inches='tight', transparent=True)
plt.show()

pkl_path = 'rsl/eigensolution/polar_fields.pkl'
data = load_bundle(pkl_path)
m1 = data['m1_full'];
m2 = data['m2_full']
Z = data['Z_full'];
FREQ = np.real(Z)
phi = data['phi_full'];
chi = data['chi_full'];
Q = data['Q_full']


fig, ax = plt.subplots(figsize=(5,4))
plt.imshow(phi-np.pi/2, extent=(m1.min(), m1.max(), m2.min(), m2.max()), origin='lower', cmap='RdBu', vmin=-1e-1, vmax=1e-1)
# plt.imshow(np.sin(2*phi), extent=(m1.min(), m1.max(), m2.min(), m2.max()), origin='lower', cmap='RdBu', vmin=-1e-6, vmax=1e-6)
plt.show()


# m1, m2 = data['m1_full'], data['m2_full']
# phi    = data['phi_full']  # φ∈[0, π)

fig, ax = plt.subplots(figsize=(3,3))
plot_phi0_phi90(
    ax, m1, m2, phi,
    lw=2.0,
    color_phi0='limegreen',   # φ=0/π
    color_phi90='black',      # φ=π/2
    overlay='s2',             # 看看 sin(2φ) 的正负分区，直观
)
plt.show()




# # 提取等频线
# paths = extract_isofreq_paths(m1, m2, FREQ, level=isofreq)
# # 沿每条等频线插值采样
# fields = {'phi': phi, 'chi': chi, 'Q': Q, 'freq': FREQ}
# samples_list = []
# for p in paths:
#     samp = sample_fields_along_path(m1, m2, fields, p, npts=n_contour_pts)
#     samples_list.append(samp)
#
# # 画曲线
# fig2, axs2 = plot_lines_along_isofreq(samples_list,
#                                       labels=[f'Γ{i + 1}' for i in range(len(samples_list))])
#
# plt.show()