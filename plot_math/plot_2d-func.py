from plot_math.functions import *
from plot_math.plt_func_2D import plot_2d_function

if __name__ == '__main__':
    plot_2d_function(
        VBG_single_resonance_efficiency,
        x_range=(-1, 1), y_range=(-1, 1),
        filename='./rsl/VBG_single_resonance_efficiency.png',
        vmin=0, vmax=1, cmap='magma',
    )

    plot_2d_function(
        gaussian_profile,
        x_range=(-1, 1), y_range=(-1, 1),
        filename='./rsl/gaussian_profile.png',
        vmin=0, vmax=1, cmap='gray',
    )

    plot_2d_function(
        lambda x, y: gaussian_profile(x, y) * VBG_single_resonance_efficiency(x, y),
        x_range=(-1, 1), y_range=(-1, 1),
        filename='./rsl/multiplication.png',
        vmin=0, vmax=1, cmap='gray',
    )
