from plot_3D_slices.core.core import plot_3d_slices, plot_2d_slices

plot_3d_slices('./sim/optical_intensity_results-even.csv', xlim=(1500, 1600), slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1], save_dir='./rsl/', zticks=[0, 0.5, 1])
plot_3d_slices('./sim/optical_intensity_results-gaussian.csv', xlim=(1500, 1600), slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1], save_dir='./rsl/', zticks=[0, 0.5, 1])
plot_3d_slices('./sim/optical_intensity_results-annular.csv', xlim=(1500, 1600), slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1], save_dir='./rsl/', zticks=[0, 0.5, 1])

plot_3d_slices('./sim/optical_intensity_results-compared-even.csv', xlim=(1400, 1500), slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1], save_dir='./rsl/', zticks=[0, 0.5, 1])
plot_2d_slices('./sim/optical_intensity_results-compared-even.csv', xlim=(1400, 1500), slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1], save_dir='./rsl/', yticks=[0, 0.5, 1])
plot_3d_slices('./sim/optical_intensity_results-compared-gaussian.csv', xlim=(1400, 1500), slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1], save_dir='./rsl/', zticks=[0, 0.5, 1])
plot_3d_slices('./sim/optical_intensity_results-compared-annular.csv', xlim=(1400, 1500), slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1], save_dir='./rsl/', zticks=[0, 0.5, 1])
plot_2d_slices('./sim/optical_intensity_results-compared-annular.csv', xlim=(1400, 1500), slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1], save_dir='./rsl/', yticks=[0, 0.5, 1])
