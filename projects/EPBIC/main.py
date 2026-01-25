from core.plot_cls import TwoDimFieldVisualizer
from core.plot_workflow import PlotConfig

if __name__ == '__main__':
    # =====================================================================================================================
    data_path = r"D:\DELL\Documents\myPlots\projects\EPBIC\mannual\EP_BIC\0.00P-3eigens.pkl"
    # data_path = r"D:\DELL\Documents\myPlots\projects\EPBIC\mannual\EP_BIC\0.25P-3eigens.pkl"
    KEY_X = 'spacer (nm)'
    KEY_Y = 'h_die_grating (nm)'
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    BOX_ASPECT = (1, 1, 2)
    config.update(figsize=(1.25, 1.25), tick_direction='in')
    plotter = TwoDimFieldVisualizer(config=config, data_path=data_path)
    plotter.load_data()
    plotter.new_3d_fig(figsize=(1.25, 1.5))
    # plotter.new_3d_fig(figsize=(1.75, 2))
    # plotter.new_3d_fig(figsize=(3, 2))
    plotter.plot_3d_surface(
        index=0, x_key=KEY_X, y_key=KEY_Y, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        elev=20, azim=120, alpha=1, vmin=0.0, vmax=0.0015,
        cmap='coolwarm', box_aspect=BOX_ASPECT, shade=False
    )
    plotter.plot_3d_surface(
        index=1, x_key=KEY_X, y_key=KEY_Y, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        elev=20, azim=120, alpha=1, vmin=0.0, vmax=0.0015,
        cmap='coolwarm', box_aspect=BOX_ASPECT, shade=False
    )
    plotter.plot_3d_surface(
        index=2, x_key=KEY_X, y_key=KEY_Y, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        elev=20, azim=120, alpha=1, vmin=0.0, vmax=0.0015,
        cmap='coolwarm', box_aspect=BOX_ASPECT, shade=False
    )
    # plotter.plot_3d_surface(
    #     index=0, x_key=X_KEY, y_key=Y_KEY, z1_key='eigenfreq_imag', z2_key='eigenfreq_real',
    #     elev=20, azim=120, alpha=1, vmin=0.573, vmax=0.5755,
    #     cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False
    # )
    # plotter.plot_3d_surface(
    #     index=1, x_key=X_KEY, y_key=Y_KEY, z1_key='eigenfreq_imag', z2_key='eigenfreq_real',
    #     elev=20, azim=120, alpha=1, vmin=0.573, vmax=0.5755,
    #     cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False
    # )
    # plotter.plot_3d_surface(
    #     index=2, x_key=X_KEY, y_key=Y_KEY, z1_key='eigenfreq_imag', z2_key='eigenfreq_real',
    #     elev=20, azim=120, alpha=1, vmin=0.573, vmax=0.5755,
    #     cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False
    # )
    plotter.add_annotations()
    plotter.save_and_show()


    data_path = r"D:\DELL\Documents\myPlots\projects\EPBIC\mannual\EP_QBIC\0.15Pslide-3eigens.pkl"
    KEY_X = 'spacer (nm)'
    KEY_Y = 'h_die_grating (nm)'
    config = PlotConfig(
        plot_params={},
        annotations={},
    )
    config.update(figsize=(1.25, 1.25), tick_direction='in')
    plotter = TwoDimFieldVisualizer(config=config, data_path=data_path)
    plotter.load_data()
    plotter.new_3d_fig(figsize=(1.25, 1.5))
    # plotter.new_3d_fig(figsize=(1.75, 2))
    # plotter.new_3d_fig(figsize=(3, 2))
    plotter.plot_3d_surface(
        index=0, x_key=KEY_X, y_key=KEY_Y, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        elev=20, azim=120, alpha=1, vmin=0.0, vmax=0.0015,
        cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False, linewidth=0, edgecolor='none', antialiased=False
    )
    plotter.plot_3d_surface(
        index=1, x_key=KEY_X, y_key=KEY_Y, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        elev=20, azim=120, alpha=1, vmin=0.0, vmax=0.0015,
        cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False, linewidth=0, edgecolor='none', antialiased=False
    )
    plotter.plot_3d_surface(
        index=2, x_key=KEY_X, y_key=KEY_Y, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        elev=20, azim=120, alpha=1, vmin=0.0, vmax=0.0015,
        cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False, linewidth=0, edgecolor='none', antialiased=False
    )
    # plotter.plot_3d_surface(
    #     index=0, x_key=X_KEY, y_key=Y_KEY, z1_key='eigenfreq_imag', z2_key='eigenfreq_real',
    #     elev=20, azim=120, alpha=1, vmin=0.573, vmax=0.5755,
    #     cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False, linewidth=0, edgecolor='none', antialiased=False
    # )
    # plotter.plot_3d_surface(
    #     index=1, x_key=X_KEY, y_key=Y_KEY, z1_key='eigenfreq_imag', z2_key='eigenfreq_real',
    #     elev=20, azim=120, alpha=1, vmin=0.573, vmax=0.5755,
    #     cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False, linewidth=0, edgecolor='none', antialiased=False
    # )
    # plotter.plot_3d_surface(
    #     index=2, x_key=X_KEY, y_key=Y_KEY, z1_key='eigenfreq_imag', z2_key='eigenfreq_real',
    #     elev=20, azim=120, alpha=1, vmin=0.573, vmax=0.5755,
    #     cmap='coolwarm', box_aspect=(1, 1, 0.75), shade=False, linewidth=0, edgecolor='none', antialiased=False
    # )
    plotter.add_annotations()
    plotter.save_and_show()