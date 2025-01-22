import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.utils import compute_circular_average, load_img


def direct_plot_2D(work_dir: str = './data',
                   data_filename: str = None,
                   data_type : str = None,
                   plot_paras: dict = None,
                   show: bool = False):
    if data_type == '.npy':
        data_2D = np.load(f'{work_dir}/{data_filename}'+data_type)
    else:
        data_2D = load_img(f'{work_dir}/{data_filename}'+data_type)[:,:,0]
    # Load your numpy array from the .npy file
    image_array = data_2D

    if 'crop' in plot_paras:
        crop = plot_paras['crop']
        if 1 > crop > 0:
            margin = int((1-crop)/2 * image_array.shape[0])
            image_array = image_array[margin:-margin, margin:-margin]
        print(f'average value {compute_circular_average(image_array)}')

    # Apply a colormap using matplotlib's imshow and get the image data
    # This will create a colormapped image based on your numpy array
    plt.imshow(image_array, cmap=plot_paras['colormap'])
    plt.axis('off')  # Optional: turns off the axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove any padding

    # Save the colormapped image to a buffer
    # Convert dictionary to a string format (JSON is a good option)
    dict_str = json.dumps(plot_paras, separators=(',', ':'))  # Remove extra spaces for a cleaner filename
    # Safely replace characters that may not be allowed in filenames (e.g., colon, commas)
    dict_str = dict_str.replace('{', '').replace('}', '').replace('"', '').replace(' ', '_').replace(':', '_')
    plt.savefig(f'./rsl/{data_filename}+{dict_str}.png', bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()

    # Close the plot to free memory
    plt.close()


if __name__ == '__main__':
    plot_paras = {
        # 'colormap': 'hot',
        # 'colormap': 'magma',
        'colormap': 'twilight',
        'crop': 0.8,
    }
    # work_dir = 'data/img_data'
    work_dir = 'data/low_loss'
    # data_files = Path(work_dir).glob('*conversion-efficiency*.npy')
    data_files = Path(work_dir).glob('*phase*.npy')
    # data_files = Path(work_dir).glob('*.png')
    # data_files = Path(work_dir).glob('*')
    for data_file in data_files:
        for crop in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        # for crop in [1.0]:
            print(crop)
            plot_paras['crop'] = crop
            data_filename = data_file.stem
            data_type = data_file.suffix
            direct_plot_2D(work_dir, data_filename, data_type, plot_paras, show=False)
