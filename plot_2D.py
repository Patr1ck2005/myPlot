import matplotlib.pyplot as plt
import numpy as np

data_filename = 'NIR-mystructure-conversion-efficiency-Si-193.41THz-1550.0nm-25deg-E0.8151-kx=-1.32-1.32-51_ky=-1.32-1.32-51'+'.npy'
data_2D = np.load(f'./plot_dataset/{data_filename}')

xstart = -10
xend = 10
ystart = -10
yend = 10

xlabel = 'x'
ylabel = 'y'

colormap = 'twilight'
colorbar = True
colorbar_label = 'colorbar_label'
vmin = 0
vmax = 1

fig, axs = plt.subplots(1, 1, figsize=(12, 8))
im = axs.imshow(data_2D, extent=[xstart, xend, ystart, yend], origin='lower', cmap=colormap, vmin=vmin, vmax=vmax)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
if colorbar:
    fig.colorbar(im, ax=axs, label=colorbar_label)
plt.savefig(f'./rsl/{data_filename}.png', dpi=300)
plt.show()


