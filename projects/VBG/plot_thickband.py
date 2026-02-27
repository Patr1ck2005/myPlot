from core.plot_cls import BandPlotterOneDim
from core.plot_workflow import PlotConfig, LinePlotter


class MyScriptPlotter(BandPlotterOneDim):
    pass


# plot_params={
#     'zlabel': "f", 'enable_line_fill': True, 'alpha': 0.3, 'legend': False,
#     'line_colors': ['blue', 'red', 'red', 'blue', 'red']
# },
# plot_params={
#     'figsize': (3, 3),
#     'zlabel': "freq (c/P)", 'xlabel': r"k ($2\pi/P$)",
#     'enable_fill': True, 'enable_line_fill': True, 'cmap': 'magma',
#     'add_colorbar': False,
#     "global_color_vmin": 0, "global_color_vmax": 0.02, "default_color": 'gray', 'legend': False,
#     'alpha_fill': 0.5,
#     'edge_color': 'none', 'title': False,
# },
# plot_params = {
#     'figsize': (3, 3),
#     'zlabel': "freq (c/P)", 'xlabel': r"k ($2\pi/P$)",
#     'enable_fill': True, 'gradient_fill': True, 'gradient_direction': 'z3', 'cmap': 'magma', 'add_colorbar': False,
#     "global_color_vmin": 0, "global_color_vmax": 0.02, "default_color": 'gray', 'legend': False, 'alpha_fill': 0.5,
#     'edge_color': 'none', 'title': False,
# },

def main(data_path):
    config = PlotConfig(
        plot_params={'scale': 0.5},
        annotations={
            'xlabel': 'k (2$\pi$/P)', 'ylabel': 'f (c/P)', 'show_axis_labels': True, 'show_tick_labels': True,
            # 'ylim': (0.51, 0.57),
            'ylim': (0.41, 0.57),
            'y_log_scale': False,
        },
    )
    config.figsize = (1.25, 2)
    config.update(tick_direction='in')
    plotter = MyScriptPlotter(config=config, data_path=data_path)
    plotter.load_data()
    plotter.prepare_data()
    plotter.new_2d_fig()

    # plotter.plot_thick_bg()
    plotter.plot_colored_bg(vmin=0, vmax=0.02, alpha=0.5)
    # plotter.plot_colored_line(vmin=2, vmax=7, cmap='magma')
    # plotter.plot_ordered_line()

    # plotter.plot_ordered_qfactor()

    plotter.adjust_view_2dim()
    plotter.add_annotations()
    plotter.save_and_show()

if __name__ == '__main__':
    data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\2Deigen-Hole_P450_T200_L250_R0.3.pkl"
    main(data_path)
