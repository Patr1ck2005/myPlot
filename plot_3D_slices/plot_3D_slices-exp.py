from plot_3D_slices.core.core import plot_3d_slices, plot_2d_slices

plot_3d_slices(
    './exp/optical_intensity_results-old-s2s.csv',
    xlim=(1480, 1580),
    xticks=[1480, 1500, 1510, 1520, 1530, 1560, 1580],
    zticks=[0, 0.5, 0.75, 1],
    slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1],
    save_dir='./rsl/',
    box_aspect=[2, 2, 1],
)
plot_2d_slices(
    './exp/optical_intensity_results-exp_focal.csv',
    xlim=(1480, 1580),
    xticks=[1480, 1500, 1510, 1520, 1530, 1560, 1580],
    yticks=[0, 0.5, 0.75, 0.9, 1],
    slice_positions=[0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1],
    save_dir='./rsl/',
    box_aspect=1.0,
)


plot_2d_slices(
    './exp/optical_intensity_results-old-s2s.csv',
    xlim=(1480, 1580),
    xticks=[1480, 1500, 1510, 1520, 1530, 1560, 1580],
    yticks=[0, 0.5, 0.75, 0.9, 1],
    slice_positions=[0.42, 0.24, 0.06],
    save_dir='./rsl/',
    # color_list=['black'],
    box_aspect=1.0,
)
plot_2d_slices(
    './exp/optical_intensity_results-exp_focal.csv',
    xlim=(1480, 1580),
    xticks=[1480, 1500, 1510, 1520, 1530, 1560, 1580],
    yticks=[0, 0.5, 0.75, 0.9, 1],
    slice_positions=[0.42],
    save_dir='./rsl/',
    color_list=['black'],
    box_aspect=1.0,
)
plot_2d_slices(
    './exp/optical_intensity_results-exp_bessel.csv',
    xlim=(1480, 1640),
    slice_positions=[0.42],
    save_dir='./rsl/',
    color_list=['black'],
    box_aspect=0.75,
)

