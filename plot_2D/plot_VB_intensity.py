from pathlib import Path

from plot_2D.direct_plot_2D import direct_plot_2D
from utils.utils import load_2D_data

if __name__ == '__main__':
    plot_paras = {
        # 'colormap': 'hot',
        # 'colormap': 'magma',
        # 'colormap': 'rainbow',
        'colormap': 'hsv',
        # 'colormap': 'twilight',
    }

    data_files = []
    data_files.append(Path('fourier_applet-farfield_intensity.npy'))
    # data_files.append(Path('fourier_applet-original_intensity.npy'))

    crop_rate = 0.05
    for data_file in data_files:
        data_filename = data_file.stem
        data_type = data_file.suffix
        twoD_data = load_2D_data(
            data_filename=data_filename,
            data_type=data_type,
            crop_rate=crop_rate
        )
        direct_plot_2D(
            twoD_data=twoD_data, save_name=f'{data_filename}-crop_rate={crop_rate}',
            plot_paras=plot_paras,
            show=False,
        )
