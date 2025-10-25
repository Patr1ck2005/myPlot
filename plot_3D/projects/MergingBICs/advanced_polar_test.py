from polar_postprocess import *

# 假设已加载数据：
data = load_bundle('rsl/eigensolution/polar_fields.pkl')
m1, m2 = data['m1_full'], data['m2_full']
FREQ = np.real(data['Z_full'])
phi = data['phi_full']
chi = data['chi_full']

fig, ax = plt.subplots(figsize=(6, 5))

# 1) 先画偏振椭圆背景
plot_polarization_ellipses(
    ax, m1, m2, phi, chi,
    step=(5, 5),  # 适当抽样，防止太密
    scale=1e-2,  # 自动用 0.8*min(dx,dy)
    color_by='phi', cmap='hsv',
    alpha=0.7, lw=1,
)

# 2) 叠加三条等频线
levels = [0.65, 0.655, 0.66]  # 你想观察的 isofreq
cs, paths_dict = plot_isofreq_contours2D(ax, m1, m2, FREQ, levels=levels,
                                         colors=['gold', 'lime', 'cyan'],
                                         linewidths=2.0)

ax.set_xlabel('m1');
ax.set_ylabel('m2')
ax.set_title('Polarization field (ellipses) + iso-freq contours')
ax.grid(True, alpha=0.2)
plt.show()


# 设定 isofreq（与保存的 Z_full.real 在同一归一化单位）
res = load_and_plot('rsl/eigensolution/polar_fields.pkl',
                    isofreq=0.65,  # 举例
                    color='phi',  # φ / tanchi / Q
                    cmap='twilight',
                    n_contour_pts=800)  # 等频曲线重采样点数