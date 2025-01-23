import json
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import compute_circular_average, load_img


def direct_plot_2D(
        twoD_data: np.ndarray,
        save_name: str,
        plot_paras: dict = None,
        show: bool = False,
        **kwargs
):
    # Load your numpy array from the .npy file
    image_array = twoD_data
    shape_ratio = image_array.shape[0] / image_array.shape[1]
    # Create a new figure with the specified size
    plt.figure(figsize=(10, 10 * shape_ratio))  # Width, Height in inches
    # Apply a colormap using matplotlib's imshow and get the image data
    # This will create a colormapped image based on your numpy array
    plt.imshow(image_array, cmap=plot_paras['colormap'], aspect='auto', origin='lower', **kwargs)
    plt.axis('off')  # Optional: turns off the axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove any padding

    # Save the colormapped image to a buffer
    # Convert dictionary to a string format (JSON is a good option)
    dict_str = json.dumps(plot_paras, separators=(',', ':'))  # Remove extra spaces for a cleaner filename
    # Safely replace characters that may not be allowed in filenames (e.g., colon, commas)
    dict_str = dict_str.replace('{', '').replace('}', '').replace('"', '').replace(' ', '_').replace(':', '_')
    plt.savefig(f'./rsl/{save_name}+{dict_str}.png', bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()

    # Close the plot to free memory
    plt.close()
