import numpy as np

from core.plot_from_comsol import main


if __name__ == '__main__':
    freq = np.loadtxt('./data/3EP_noslab/Γ-buffer_space/freq.txt', comments='%')[:, 1]
    individual = 3
    each_length = int(len(freq)/individual)
    main(
        freq,
        val=None,
        x=np.linspace(500, 650, each_length),
        # ylim=[7.0e13-7.12e13, 7.22e13-7.12e13],
        # xlim=[1e-5, 1e-3],
        # plot_ylim=[1e10, 1e12],
        selection=[1, 2, 3],
        save_name='3EP_noslab-Γ-buffer_space-freq'
    )
    ifreq = np.loadtxt('./data/3EP_noslab/Γ-buffer_space/ifreq.txt', comments='%')[:, 1]
    main(
        ifreq,
        val=None,
        x=np.linspace(500, 650, each_length),
        # xlim=[1e-5, 1e-3],
        # ylim=[-2*1e13+1.2e13, 0+1.2e13],
        # plot_ylim=[1e10, 1e12],
        selection=[1, 2, 3],
        save_name='3EP_noslab-Γ-buffer_space-ifreq'
    )

