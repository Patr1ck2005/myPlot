# Set n=2 for 3/2 approximant
import numpy as np
from quasicrystal import *

n = 2
s = 1 + np.sqrt(2)
lattice_constant = 1 / s**n
xs = find_AB_points(n)
hops = hopping_PBC(xs, lattice_constant)
z = nearest_neighbor(xs[:,0:2], hops)
x_ws, bond_ws = make_window(n)
make_plot_window(x_ws, bond_ws, xs, z)
hops_uc = get_bonds(xs[:,0:2], lattice_constant)
make_plot_uc(xs[:,0:2], hops_uc)
make_plot_real(xs, hops_uc, z)
make_plot_hop(xs[:,0:2], hops, lattice_constant)
make_plot_z(z)

# Add DXF export for supercell tiling
with open('supercell_3_2.dxf', 'w') as f:
    f.write(' 0\nSECTION\n 2\nHEADER\n 0\nENDSEC\n 0\nSECTION\n 2\nENTITIES\n')
    # For each hopping bond, add line (simplified; for full tiles, group bonds into polygons)
    for hop in hops_uc:
        i, j = hop
        pt1 = xs[i, 0:2]
        pt2 = xs[j, 0:2]
        f.write(' 0\nLINE\n 8\nLayer0\n')
        f.write(' 10\n{:.6f}\n 20\n{:.6f}\n 30\n0.0\n'.format(pt1[0], pt1[1]))
        f.write(' 11\n{:.6f}\n 21\n{:.6f}\n 31\n0.0\n'.format(pt2[0], pt2[1]))
    f.write(' 0\nENDSEC\n 0\nEOF\n')

print("Generated 3/2 approximant images and supercell_3_2.dxf (lines for tiling edges).")