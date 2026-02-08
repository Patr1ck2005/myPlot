if __name__ == '__main__':
    from core.plot_cls import MomentumSpaceEigenVisualizer
    from core.plot_workflow import PlotConfig
####################################################################################################################
    # data_path = './manual/Asym_1.2vs1-520T-aroundFWBIC-upward.pkl'
    # data_path = './manual/Asym_1.2vs1-520T-aroundFWBIC-downward.pkl'
    # data_path = './manual/Asym_0.9vs1-520T-aroundFWBIC-upward.pkl'
    # data_path = './manual/Asym_0.9vs1-520T-aroundFWBIC-downward.pkl'
    # data_path = './manual/Asym_1vs1-520T-aroundFWBIC-upward.pkl'
    # data_path = './manual/Vacuum-520T-0.015k-upward.pkl'
    data_path = './manual/StrB-around_X_BIC.pkl'
    BAND_INDEX = 0
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.25, 1.25), tick_direction='in')

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
    plotter.imshow_field(index=BAND_INDEX, field_key='qlog', cmap='nipy_spectral', vmin=2, vmax=8)
    plotter.add_annotations()
    plotter.save_and_show()

    BIC_KX = 0.1156
    # BIC_KX = 0.1165
    BIC_KY = 0.0
    REGIME_RADIUS = 0.015

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    print("mesh step", plotter.coordinates['m1'][0] - plotter.coordinates['m1'][1])
    # plotter.plot_field_regimes(index=BAND_INDEX, z_key='s2')
    # plotter.plot_field_splits(index=BAND_INDEX, s1_key='s2', s2_key='s1')
    # plotter.plot_field_regimes(index=BAND_INDEX, z_key='s1')
    # plotter.plot_field_splits(index=BAND_INDEX, s1_key='s1', s2_key='s2')
    plotter.plot_field_regimes(index=BAND_INDEX, z_key='s3', colors=("lightcoral", "lightblue"))
    plotter.plot_field_splits(index=BAND_INDEX, s1_key='s3', s2_key='s1')
    # 在图中标出 FW-BIC 位置
    plotter.ax.scatter(BIC_KX, BIC_KY, color='black', marker='o', s=10, label='FW-BIC')
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.25, 1.25))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(4, 4), scale=2e-3, cmap='coolwarm', s1_key='s1',
                                       s2_key='s2', s3_key='s3')
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.25, 1.25))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(4, 4), scale=2e-3, cmap='coolwarm', s1_key='s1',
                                       s2_key='s3', s3_key='s2')
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_3d_fig(figsize=(2, 2))
    plotter.plot_on_poincare_sphere_along_around_path(
        # index=BAND_INDEX, center=(BIC_KX, 0), radius=0.015, cmap='rainbow',
        index=BAND_INDEX, center=(0, 0), radius=0.005, cmap='rainbow',
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

    import matplotlib.colors as mcolors
    plotter.new_2d_fig(figsize=(1.25, 1.25))
    bounds = [-1, -0.995, -0.99, -0.95, -0.5, -0.05, 0.05, 0.5, 0.95, 0.99, 0.995, 1]
    norm = mcolors.BoundaryNorm(bounds, ncolors=256)
    im = plotter.imshow_field(index=BAND_INDEX, field_key='s3', cmap='coolwarm', norm=norm)
    plotter.ax.set_xlim(BIC_KX - REGIME_RADIUS, BIC_KX + REGIME_RADIUS)
    plotter.ax.set_ylim(BIC_KY - REGIME_RADIUS, BIC_KY + REGIME_RADIUS)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.25, 1.25))
    # plotter.add_annotations()
    import matplotlib.colors as mcolors
    bounds = [-1, -0.995, -0.99, -0.95, -0.5, -0.05, 0.05, 0.5, 0.95, 0.99, 0.995, 1]
    norm = mcolors.BoundaryNorm(bounds, ncolors=256)
    plotter.imshow_field(index=BAND_INDEX, field_key='s2', cmap='coolwarm', norm=norm)
    # plotter.plot_skyrmion_quiver(index=BAND_INDEX, step=(1, 1), color='gray', s1_key='s1', s2_key='s3', s3_key='s2')
    plotter.plot_polar_quiver(index=BAND_INDEX, step=(1, 1), color='gray', s1_key='s1', s2_key='s3', s3_key='s2')
    plotter.ax.set_xlim(BIC_KX - REGIME_RADIUS, BIC_KX + REGIME_RADIUS)
    plotter.ax.set_ylim(BIC_KY - REGIME_RADIUS, BIC_KY + REGIME_RADIUS)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.5, 1.5))
    # plotter.plot_field_regimes(index=BAND_INDEX, z_key='s2')
    # plotter.plot_field_splits(index=BAND_INDEX, s1_key='s2', s2_key='s1')
    # plotter.plot_field_regimes(index=BAND_INDEX, z_key='s1')
    # plotter.plot_field_splits(index=BAND_INDEX, s1_key='s1', s2_key='s2')
    plotter.plot_field_regimes(index=BAND_INDEX, z_key='s3')
    plotter.plot_field_splits(index=BAND_INDEX, s1_key='s3', s2_key='s1')
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.25, 1.25))
    plotter.plot_polarization_ellipses(index=BAND_INDEX, step=(3, 3), scale=2e-3, cmap='coolwarm')
    plotter.ax.set_xlim(BIC_KX - REGIME_RADIUS, BIC_KX + REGIME_RADIUS)
    plotter.ax.set_ylim(BIC_KY - REGIME_RADIUS, BIC_KY + REGIME_RADIUS)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(3.8, 1))
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(BIC_KX, BIC_KY), radius=0.001, color='purple', field_key='phi', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(BIC_KX, BIC_KY), radius=0.003, color='r', field_key='phi', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(BIC_KX, BIC_KY), radius=0.005, color='k', field_key='phi', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(BIC_KX, BIC_KY), radius=0.01, color='b', field_key='phi', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(BIC_KX, BIC_KY), radius=0.015, color='g', field_key='phi', alpha=0.5)
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(3.8, 1))
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.001, color='r', field_key='s3', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.003, color='k', field_key='s3', alpha=0.5)
    plotter.plot_field_along_round_path(index=BAND_INDEX, center=(0, 0), radius=0.005, color='b', field_key='s3', alpha=0.5)
    plotter.save_and_show()

    plotter.new_3d_fig(figsize=(2, 2))
    # 通过限制 x_key_lim 和 y_key_lim 来放大查看 FW-BIC 附近的极化态分布
    # 通过波矢离FW-BIC的距离计算得到散点的rgba颜色
    plotter.plot_on_poincare_sphere(
        index=BAND_INDEX,
        x_key_lim=(BIC_KX - REGIME_RADIUS, BIC_KX + REGIME_RADIUS),
        y_key_lim=(BIC_KY - REGIME_RADIUS, BIC_KY + REGIME_RADIUS),
        cmap='Reds',
        s=3,
        sphere_style='surface'
    )
    # 去掉3D绘图的背景和背景网格线
    plotter.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plotter.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plotter.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plotter.ax.grid(False)
    plotter.ax.view_init(elev=20, azim=30)
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.25, 1.25))
    plotter.imshow_field(index=BAND_INDEX, field_key='qlog', cmap='hot', vmin=4, vmax=8)
    plotter.ax.set_xlim(BIC_KX - REGIME_RADIUS, BIC_KX + REGIME_RADIUS)
    plotter.ax.set_ylim(BIC_KY - REGIME_RADIUS, BIC_KY + REGIME_RADIUS)
    # plotter.add_annotations()
    plotter.save_and_show()

    plotter.new_2d_fig(figsize=(1.25, 1.25))
    plotter.imshow_skyrmion_density(index=BAND_INDEX, cmap='bwr')
    # plotter.add_annotations()
    plotter.save_and_show()
    ####################################################################################################################
