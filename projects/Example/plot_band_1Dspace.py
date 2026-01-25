from core.plot_cls import OneDimFieldVisualizer
from core.plot_workflow import PlotConfig


class MyScriptPlotter(OneDimFieldVisualizer):
    pass


def main(data_path):
    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={
            'xlabel': r'', 'ylabel': '',
            # 'xlabel': r'$t_{die}$', 'ylabel': 'f (c/P)',
            'show_axis_labels': True, 'show_tick_labels': True,
            # 'ylim': (0.51, 0.57),
        },
    )
    config.update(figsize=(1.25, 0.75), tick_direction='in')
    plotter = MyScriptPlotter(config=config, data_path=data_path)
    plotter.load_data()
    # plotter.prepare_data()
    plotter.new_2d_fig()
    plotter.plot(index=0, x_key='h_die_grating (nm)', z1_key='eigenfreq', default_color='black',)
    # 绘制一条y=0.54的水平线
    plotter.ax.axhline(y=0.575, color='red', linestyle='--', linewidth=1)
    plotter.add_annotations()
    plotter.save_and_show()

if __name__ == '__main__':
    data_path = r"D:\DELL\Documents\myPlots\projects\MergingBICs\manual\2Deigen-Hole_P450_T200_L250_R0.3.pkl"
    main(data_path)
