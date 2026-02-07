####################################################################################################################
if __name__ == '__main__':
    from core.plot_cls import MomentumSpaceEigenVisualizer
    from core.plot_workflow import PlotConfig

    ####################################################################################################################
    # data_path = './manual/SOS-450T520-0.2k-upward.pkl'
    # data_path = './manual/SOS-450T520-0.2k-downward.pkl'
    # data_path = './manual/SOS-520T-0.2k-upward.pkl'
    # data_path = './manual/SOS-520T-0.2k-downward.pkl'
    # data_path = './manual/Vacuum-507T520-0.2k-upward.pkl'
    # data_path = './manual/Vacuum-505T520-0.2k-downward.pkl'
    # data_path = './manual/Vacuum-520T520-0.2k-upward.pkl'
    data_path = './manual/Vacuum-520T520-0.2k-upward.pkl'
    # data_path = './manual/Asym_1.67vs1-520T-0.2k-upward.pkl'
    # data_path = './manual/Asym_1.67vs1-520T-0.2k-downward.pkl'
    # data_path = './manual/Vacuum-502T-0.2k-upward.pkl'
    # data_path = './manual/Vacuum-505T520-0.2k0.05k-upward.pkl'
    # data_path = './manual/Vacuum-520T_Parity_0.05-0.2k-upward.pkl'
    # data_path = './manual/Vacuum-520T_Parity_0.1-0.2k-downward.pkl'
    # data_path = './manual/Vacuum-520T_Arrow_0.15-0.2k-upward.pkl'
    BAND_INDEX = 1
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.5, 1.5), tick_direction='in')

    plotter = MomentumSpaceEigenVisualizer(config=config, data_path=data_path)
    plotter.load_data()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s1', cmap='coolwarm', vmin=-1, vmax=1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s2', cmap='coolwarm', vmin=-1, vmax=1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='s3', cmap='coolwarm', vmin=-1, vmax=1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_field(index=BAND_INDEX, field_key='qlog', cmap='nipy_spectral', vmin=2, vmax=7)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_3d_fig(figsize=(2, 2))
    plotter.plot_on_poincare_sphere_along_around_path(
        index=BAND_INDEX, center=(0, 0), radius=0.05, cmap='rainbow',
        sphere_style='wire', arrow_length_ratio=1
    )
    # 去掉3D绘图的背景和背景网格线
    plotter.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plotter.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plotter.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plotter.ax.grid(False)
    plotter.ax.view_init(elev=20, azim=30)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    print("mesh step", plotter.coordinates['m1'][0]-plotter.coordinates['m1'][1])
    plotter.plot_field_regimes(index=BAND_INDEX, z_key='s3', colors=("lightcoral", "lightblue"))
    plotter.plot_field_splits(index=BAND_INDEX, s1_key='s3', s2_key='s1')
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    print("mesh step", plotter.coordinates['m1'][0]-plotter.coordinates['m1'][1])
    plotter.plot_field_regimes(index=BAND_INDEX, z_key='s2')
    plotter.plot_field_splits(index=BAND_INDEX, s1_key='s2', s2_key='s1')
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(1, 1), scale=1e-2, cmap='coolwarm', s1_key='s1', s2_key='s3', s3_key='s2')
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(1, 1), scale=1e-2, cmap='coolwarm', s1_key='s1', s2_key='s2', s3_key='s3')
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1, 1))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(4, 4), scale=2e-2, cmap='coolwarm', s1_key='s1', s2_key='s3', s3_key='s2')
    plotter.ax.set_xlim(-0.1, 0.1)
    plotter.ax.set_ylim(-0.1, 0.1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1, 1))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(4, 4), scale=2e-2, cmap='coolwarm', s1_key='s1', s2_key='s2', s3_key='s3')
    plotter.ax.set_xlim(-0.1, 0.1)
    plotter.ax.set_ylim(-0.1, 0.1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.25, 1.25))
    import matplotlib.colors as mcolors
    bounds = [-1, -0.995, -0.99, -0.95, -0.5, -0.05, 0.05, 0.5, 0.95, 0.99, 0.995, 1]
    norm = mcolors.BoundaryNorm(bounds, ncolors=256)
    plotter.imshow_field(index=BAND_INDEX, field_key='s2', cmap='coolwarm', norm=norm)
    # plotter.plot_skyrmion_quiver(index=BAND_INDEX, step=(1, 1), cmap='coolwarm', s1_key='s1', s2_key='s3', s3_key='s2')
    # plotter.plot_polar_quiver(index=BAND_INDEX, step=(1, 1), color='gray', s1_key='s1', s2_key='s2', s3_key='s3')
    plotter.plot_polar_quiver(index=BAND_INDEX, step=(1, 1), color='gray', s1_key='s1', s2_key='s3', s3_key='s2')
    # plotter.add_annotations()
    plotter.save_and_show()

    import matplotlib.colors as mcolors
    plotter.new_2d_fig(figsize=(1.25, 1.25))
    bounds = [-1, -0.995, -0.99, -0.95, -0.5, -0.05, 0.05, 0.5, 0.95, 0.99, 0.995, 1]
    norm = mcolors.BoundaryNorm(bounds, ncolors=256)
    plotter.imshow_field(index=BAND_INDEX, field_key='s3', cmap='coolwarm', norm=norm)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_3d_fig(figsize=(3, 3))
    # plotter.plot_3d_surfaces(
    #     indexs=(0, 1), z1_key='eigenfreq_real', z2_key='qlog', cmap='rainbow', elev=45, vmin=2, vmax=7, shade=False
    # )
    # rbga = plotter.get_advanced_color_mapping(index=BAND_INDEX)
    # plotter.plot_3d_surface(index=BAND_INDEX, z1_key='eigenfreq_real', rgba=rbga, cmap='rainbow', elev=45, shade=False)
    plotter.plot_3d_surface(
        index=BAND_INDEX, z1_key='eigenfreq_real', z2_key='qlog', cmap='hot', elev=45, vmin=2, vmax=7, shade=False
    )
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(4, 4), scale=2e-2, cmap='coolwarm')
    # plotter.plot_iso_contours2D(index=BAND_INDEX, levels=(0.36, 0.37, 0.38), z_key='eigenfreq_real', colors=('r', 'k', 'b'))
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1, 1))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(4, 4), scale=2e-2, cmap='coolwarm')
    plotter.ax.set_xlim(-0.1, 0.1)
    plotter.ax.set_ylim(-0.1, 0.1)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(3.8, 1))
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.03, color='r', field_key='phi', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.05, color='k', field_key='phi', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.07, color='b', field_key='phi', alpha=0.5)
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(3.8, 1))
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.03, color='r', field_key='s3', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.05, color='k', field_key='s3', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.07, color='b', field_key='s3', alpha=0.5)
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_compare_datas(
        index_A=0, index_B=1,
        field_key='eigenfreq_real',
        cmap='nipy_spectral',
        vmin=0,
    )
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_compare_datas(
        index_A=0, index_B=1,
        field_key='eigenfreq_imag',
        cmap='nipy_spectral',
        vmin=0,
    )
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    plotter.imshow_compare_datas(
        index_A=0, index_B=1,
        field_key='eigenfreq',
        cmap='nipy_spectral',
        vmin=0,
    )
    plotter.save_and_show()
    ####################################################################################################################
