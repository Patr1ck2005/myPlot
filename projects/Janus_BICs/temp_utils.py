

def plot_phase_diff(raw_datasets, new_coords, BAND_INDEX):
    from matplotlib import pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(2, 1.5))
    # phase_diff = (np.angle(up_cx2) - np.angle(up_cy2) + np.pi)%(2*np.pi) - np.pi
    # 通过kx, ky, 变换 (x, y) -> (s, p)
    kx, ky = np.meshgrid(new_coords['m1'], new_coords['m2'], indexing='ij')  # 或 new_coords['m1'], new_coords['m2']
    k_par = np.sqrt(kx ** 2 + ky ** 2)
    up_cs = (-ky * raw_datasets[BAND_INDEX]['up_cx (V/m)'] + kx * raw_datasets[BAND_INDEX]['up_cy (V/m)']) / k_par
    up_cp = (kx * raw_datasets[BAND_INDEX]['up_cx (V/m)'] + ky * raw_datasets[BAND_INDEX]['up_cy (V/m)']) / k_par
    phase_diff = (np.angle(up_cs) - np.angle(up_cp) + np.pi) % (2 * np.pi) - np.pi
    c = ax.imshow(phase_diff.T, origin='lower', extent=(
        new_coords['m1'][0], new_coords['m1'][-1],
        new_coords['m2'][0], new_coords['m2'][-1],
    ), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    # cs = ax.contour(kx, ky, phase_diff, levels=[-np.pi / 2, np.pi / 2], colors=['r', 'b'], linewidths=0.5)
    cs = ax.contour(kx, ky, np.real(up_cs * np.conj(up_cp)), levels=[0], colors=['r'], linewidths=0.5)
    cs = ax.contour(kx, ky, np.imag(up_cs * np.conj(up_cp)), levels=[0], colors=['b'], linewidths=0.5)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    plt.savefig('./c.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()