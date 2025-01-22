import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from plot_2D.direct_plot_2D import direct_plot_2D
from utils.utils import compute_circular_average, load_img

if __name__ == '__main__':
    plot_paras = {
        # 'colormap': 'hot',
        # 'colormap': 'magma',
        'colormap': 'gray',
        'crop': 1.0,
    }
    # work_dir = 'data/img_data'
    work_dir = '../data'
    data_files = Path(work_dir).glob('*LG*')
    for data_file in data_files:
        data_filename = data_file.stem
        data_type = data_file.suffix
        if data_type == '.npy':
            data_2D = np.load(f'{work_dir}/{data_filename}' + data_type)
        else:
            data_2D = load_img(f'{work_dir}/{data_filename}' + data_type)[:, :, 0]
        # data_2D = np.log(data_2D)
        data_2D *= -1
        direct_plot_2D(
            data_2D,
            save_name=f'{data_filename}',
            plot_paras=plot_paras,
            show=False
        )
