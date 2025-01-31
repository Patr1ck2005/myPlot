import numpy as np

from plot_1D.plot_dual_y_axis import plot_two_scales

As_data = np.load('./data/RCWA-isolate-ThE-k_f_As-A48-raw_plotting_data.npy')
Ap_data = np.load('./data/RCWA-isolate-ThE-k_f_Ap-A48-raw_plotting_data.npy')
mesh_y = As_data[1]
wavelength = 299792458/mesh_y[0, :]*1e-6
As = As_data[2, 1, :]
Ap = Ap_data[2, 1, :]

# 调用通用函数
plot_two_scales(
    wavelength, As, Ap,
    figsize=(8, 6),
    x_label='Wavelength (um)',
    y1_label='As',
    y1_lim=[0, 1],
    y2_label='Ap',
    y2_lim=[0, 1],
    y1_color='brown',
    y2_color='navy',
    title='',
    y1_marker='o',
    y2_marker='s',
    save_name='iso_ThE-polar_dependence',
    markersize=3
)