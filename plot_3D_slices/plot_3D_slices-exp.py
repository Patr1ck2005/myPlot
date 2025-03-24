from plot_3D_slices.core.core import plot_3d_slices, plot_2d_slices

plot_3d_slices('./exp/optical_intensity_results-exp_focal.csv', xlim=(1500, 1600), slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1], save_dir='./rsl/')
plot_2d_slices('./exp/optical_intensity_results-exp_focal.csv', xlim=(1500, 1600), slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1], save_dir='./rsl/')


plot_2d_slices(
    './exp/optical_intensity_results-old-s2s.csv',
    xlim=(1500, 1600),
    slice_positions=[0.42, 0.24, 0.06],
    save_dir='./rsl/',
    # color_list=['black'],
)
plot_2d_slices(
    './exp/optical_intensity_results-exp_focal.csv',
    xlim=(1500, 1600),
    slice_positions=[0.42],
    save_dir='./rsl/',
    color_list=['black'],
)
