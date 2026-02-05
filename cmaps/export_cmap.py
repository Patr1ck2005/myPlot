import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Parameters
# cmap_name = "coolwarm"
# cmap_name = "nipy_spectral"
cmap_name = "magma"
# width = 1024
# height = 64

width = 128
height = 8

# Create gradient
gradient = np.linspace(0, 1, width)
gradient = np.tile(gradient, (height, 1))

# Plot without any borders, axes, or padding
fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
ax = plt.Axes(fig, [0, 0, 1, 1])  # full canvas
fig.add_axes(ax)
ax.imshow(gradient, aspect="auto", cmap=cmap_name)
ax.set_axis_off()

# Save pure PNG
out_path = Path(f"./{cmap_name}.png")
fig.savefig(out_path, dpi=100, transparent=False)
plt.close(fig)

out_path.as_posix()
