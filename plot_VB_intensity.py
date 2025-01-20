from pathlib import Path

from direct_plot_2D import direct_plot_2D


if __name__ == '__main__':
    plot_paras = {
        # 'colormap': 'hot',
        'colormap': 'magma',
        # 'colormap': 'twilight',
        'crop': 0.15,
    }

    # data_file =Path('fourier_applet-farfield_intensity.npy')
    data_file =Path('fourier_applet-original_intensity.npy')
    data_filename = data_file.stem
    data_type = data_file.suffix
    direct_plot_2D(data_filename=data_filename,
                   data_type=data_type,
                   plot_paras=plot_paras,
                   show=True)

