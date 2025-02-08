from pathlib import Path

import numpy as np

from plot_2D.direct_plot_2D import direct_plot_2D
from utils.utils import load_img, compute_circular_average

if __name__ == '__main__':
    plot_paras = {
        # 'colormap': 'hot',
        'colormap': 'magma',
        # 'colormap': 'twilight',
    }
    # work_dir = 'data/img_data'
    # work_dir = '../data/low_loss'
    # work_dir = '../data/low_loss/full'
    work_dir = './data/opt'
    data_files = Path(work_dir).glob('*conversion-efficiency*.npy')
    # data_files = Path(work_dir).glob('*phase*.npy')
    # data_files = Path(work_dir).glob('*.png')
    # data_files = Path(work_dir).glob('*')
    for data_file in data_files:
        for crop in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
            print(crop)
            data_filename = data_file.stem
            data_type = data_file.suffix
            if data_type == '.npy':
                data_2D = np.load(f'{work_dir}/{data_filename}' + data_type)
            else:
                data_2D = load_img(f'{work_dir}/{data_filename}' + data_type)[:, :, 0]
            if 1 > crop > 0:
                margin = int((1 - crop) / 2 * data_2D.shape[0])
                data_2D = data_2D[margin:-margin, margin:-margin]
            print(f'average value {compute_circular_average(data_2D)}')
            print(f'max value {np.max(data_2D)}')
            # plot_paras['crop'] = crop
            direct_plot_2D(data_2D,
                           save_name=f'{data_filename}-crop{crop}',
                           plot_paras=plot_paras,
                           show=False)
