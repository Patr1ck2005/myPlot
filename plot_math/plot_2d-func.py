from utils.functions import *
from plot_math.core.plt_func_2D import plot_2d_function

if __name__ == '__main__':
    plot_2d_function(
        lambda x, y: np.abs(VBG_single_resonance_converted(x, y))**2,
        x_range=(-1, 1), y_range=(-1, 1),
        filename='./rsl/VBG_single_resonance_efficiency.png',
        vmin=0, vmax=1, cmap_name='magma', transparent=False,
    )

    plot_2d_function(
        lambda x, y: gaussian_profile(x, y, w=0.6)**2,
        x_range=(-1, 1), y_range=(-1, 1),
        filename='./rsl/gaussian_profile.png',
        vmin=0,
        cmap_name='magma',
    )

    # plot_2d_function(
    #     lambda x, y: gaussian_profile(x, y, w=0.6, l=2)**2,
    #     x_range=(-1, 1), y_range=(-1, 1),
    #     filename='./rsl/vortex_gaussian_profile.png',
    #     vmin=0, vmax=1, cmap_name='magma',
    # )

    plot_2d_function(
        lambda x, y: gaussian_profile(x, y, w=0.8) ** 2 * np.abs(VBG_single_resonance_converted(x, y))**2,
        x_range=(-1, 1), y_range=(-1, 1), vmin=0,
        filename='./rsl/multiplication.png', cmap_name='magma',
    )

    plot_2d_function(
        lambda x, y: np.real(np.exp(2j*np.atan(y/x)) * VBG_single_resonance_converted(x, y)),
        x_range=(-1, 1), y_range=(-1, 1),
        filename='./rsl/multiplication.png', cmap_name='RdBu',
    )

    file_path = r"D:\DELL\Documents\myPlots\plot_2D\data\opt\NIR-mystructure-conversion-efficiency-Opt-Si-193.54THz-1549.0nm-25deg-E0.8896-kx=-1.35-1.35-101_ky=-1.35-1.35-101.npy"
    sim_best_efficiency = np.load(file_path)
    plot_2d_function(
        lambda x, y: gaussian_profile(x, y)**2 * interpolate_2d(z=sim_best_efficiency, x_new=x, y_new=y),
        x_range=(-1, 1), y_range=(-1, 1),
        filename='./rsl/best_multiplication.png',
        vmin=0, vmax=1, cmap_name='magma',
    )
