import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

plt.rcParams['font.size'] = 12


def plot_3d_slices(filename, xlim=(1500, 1600), slice_positions=None, save_dir='./rsl/'):

    df = pd.read_csv(f'./data/{filename}')
    dataset = {}

    for slice_value in slice_positions:
        wavelength_array = df['wavelength_nm'].array
        intensity_array = df[f'avg_intensity_NA{slice_value}'].array
        dataset[slice_value] = {
            'x': wavelength_array[wavelength_array < xlim[1]],
            'z': intensity_array[wavelength_array < xlim[1]],
            'max_point': (wavelength_array[intensity_array.argmax()], intensity_array.max())
        }

    def polygon_under_graph(x, y):
        return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

    verts = []
    for i, slice in enumerate(slice_positions):
        x = dataset[slice]['x']
        y = dataset[slice]['z']
        verts.append(polygon_under_graph(x, y))

    fig1 = plt.figure(figsize=(8, 8))
    ax = fig1.add_subplot(111, projection='3d')

    facecolors = plt.colormaps['inferno_r'](np.linspace(0, 1, len(verts)))[::-1]
    poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
    ax.add_collection3d(poly, zs=slice_positions, zdir='y')

    for i, slice in enumerate(slice_positions):
        ax.plot([xlim[0], xlim[1]], [slice, slice], [0, 0], color=facecolors[i], linewidth=2)
        ax.plot([xlim[0]], [slice], [dataset[slice]['max_point'][1]], color=facecolors[i], marker='o', markersize=5,
                markeredgecolor='black', markeredgewidth=1, zorder=99)
        ax.plot([dataset[slice]['max_point'][0], xlim[0]], [slice, slice],
                [dataset[slice]['max_point'][1], dataset[slice]['max_point'][1]], color=facecolors[i], linewidth=1,
                linestyle='--')

    ax.grid(True)
    ax.set_xlim(xlim)
    ax.set_ylim(0, 0.42)
    ax.set_zlim(0, 1)
    ax.set_xticks([1500, 1520, 1540, 1550, 1560, 1580, 1600])
    ax.set_yticks(slice_positions[::2])
    ax.set_zticks([0, 0.5, 0.75, 0.9, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.tick_params(axis='x', pad=1)
    ax.tick_params(axis='y', pad=1)
    ax.tick_params(axis='z', pad=1)
    ax.view_init(elev=30, azim=60)
    ax.set_box_aspect([2, 2, 1])

    plt.tight_layout()
    plt.savefig(f'{save_dir}3D_slices_fig.png', dpi=300, bbox_inches='tight', pad_inches=0.3, transparent=True)
    plt.savefig(f'{save_dir}3D_slices_fig.svg', bbox_inches='tight', pad_inches=0.3, transparent=True)
    plt.show()


def plot_2d_slices(filename, xlim=(1500, 1600), slice_positions=None, save_dir='./rsl/', color_list=None):
    if slice_positions is None:
        slice_positions = [0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.06][::-1]

    df = pd.read_csv(f'./data/{filename}')
    dataset = {}

    for slice_value in slice_positions:
        wavelength_array = df['wavelength_nm'].array
        intensity_array = df[f'avg_intensity_NA{slice_value}'].array
        dataset[slice_value] = {
            'x': wavelength_array[wavelength_array < xlim[1]],
            'z': intensity_array[wavelength_array < xlim[1]],
            'max_point': (wavelength_array[intensity_array.argmax()], intensity_array.max())
        }

    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(111)

    if color_list is None:
        facecolors = plt.colormaps['inferno_r'](np.linspace(0, 1, len(slice_positions)))[::-1]
    else:
        facecolors = color_list
    for i, slice in enumerate(slice_positions):
        ax2.plot(dataset[slice]['x'], dataset[slice]['z'], color=facecolors[i], linewidth=3)
        ax2.plot([dataset[slice]['max_point'][0]], [dataset[slice]['max_point'][1]], color=facecolors[i], marker='o',
                 markersize=5)
        ax2.plot([dataset[slice]['max_point'][0], xlim[0]],
                 [dataset[slice]['max_point'][1], dataset[slice]['max_point'][1]], color=facecolors[i], linewidth=3,
                 linestyle='--')

    ax2.grid(True)
    ax2.set_xlim(xlim)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([1500, 1520, 1540, 1550, 1560, 1580, 1600])
    ax2.set_yticks([0, 0.5, 0.75, 1])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(f'{save_dir}2D_slices_fig.png', dpi=300, bbox_inches='tight', pad_inches=0.3, transparent=True)
    plt.savefig(f'{save_dir}2D_slices_fig.svg', bbox_inches='tight', pad_inches=0.3, transparent=True)
    plt.show()
