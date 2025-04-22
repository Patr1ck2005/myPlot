from pathlib import Path

import numpy as np

from plot_2D.core.direct_plot_2D import direct_plot_2D
from plot_2D.core.discrete_plot import direct_plot_3D, direct_plot_aggregated_bar

from utils.utils import load_img

if __name__ == '__main__':
    plot_paras = {
        'colormap': 'magma',
    }
    # work_dir = 'data/img_data'
    # work_dir = '../data'
    work_dir = './data/LG_compo'
    data_files = Path(work_dir).glob('*LG*')
    for data_file in data_files:
        data_filename = data_file.stem
        data_type = data_file.suffix
        if data_type == '.npy':
            data_2D = np.load(f'{work_dir}/{data_filename}' + data_type)
        else:
            data_2D = load_img(f'{work_dir}/{data_filename}' + data_type)[:, :, 0]
        # data_2D = np.log(data_2D)
        direct_plot_2D(
            data_2D,
            save_name=f'{data_filename}',
            plot_paras=plot_paras,
            show=False,
            stretch=100,
        )
        plot_paras_2 = {
            'font_size': 18,
            'colormap': 'magma',
            # 'xlabel': r'$p$',
            # 'ylabel': r'$l$',
            # 'zlabel': 'a.u',
            'yticks': [0, 3, 5, 7, 10],
            # 'ytickslabels': [-5, -2, 0, 2, 5],
            'ytickslabels': [],
            'ztickslabels': [],
            'xtickslabels': [],
            'z-a.u': True,
            # 'log_scale': 'log'
            'log_color': True,
            'view_angle': [30, -60-90],
            'raw_yindex': [5, 16],
            'box_aspect': [1, 1, 1],
            # 'vmin': 0,
            # 'vmax': 3,
            'grid': True,
        }
        data_2D = data_2D[5:16, :]
        direct_plot_3D(
            data_2D,
            save_name=f'{data_filename}',
            plot_paras=plot_paras_2,
            show=False,
        )
        plot_paras_3 = {
            'font_size': 20,
            'title': None,
            'colormap': 'magma',
            'bar_color': 'black',
            'bulk_bar': False,
            'xlabel': None,
            'ylabel': None,
            'zlabel': None,
            'yticks': [],
            'ytickslabels': [],
            'xtickslabels': [],
            'z-a.u': True,
            # 'log_scale': 'log'
            'view_angle': [30, -60-90],
            'box_aspect': 1,
            'vmin': 0,
            'vmax': 1,
            'zlim': [0, 1],
            'mark_value': False,
            'grid': True,
        }
        direct_plot_aggregated_bar(
            data_2D,
            save_name=f'{data_filename}',
            aggregate_axis='y',
            plot_paras=plot_paras_3,
            show=False,
        )
