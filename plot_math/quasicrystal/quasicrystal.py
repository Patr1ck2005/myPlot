import numpy as np
import matplotlib.pyplot as plt

def Octonacci(n):
    if n == 1:
        return 1
    elif n == 2:
        return 2
    else:
        return 2 * Octonacci(n - 1) + Octonacci(n - 2)

def make_params(n):
    alpha = (-1)**(n + 1) * Octonacci(n)
    delta = (-1)**(n) * Octonacci(n + 1)
    beta = alpha + delta
    return alpha, delta, beta

def find_points_in_window(n):
    alpha, _, beta = make_params(n)
    if n % 2 == 1:
        alpha, beta = -alpha, -beta
    result = []
    # region I
    S = np.array([[1, 1], [1, -1]])
    T = np.array([[2 * alpha], [0]])
    P = abs(alpha)
    Q = abs(alpha)
    for p in range(P):
        for q in range(Q):
            v = np.array([[p], [q]])
            A = S @ v + T
            result += [A]
    # region I'
    S = np.array([[1, 1], [1, -1]])
    T = np.array([[2 * alpha + 1], [0]])
    P = abs(alpha)
    Q = abs(alpha)
    for p in range(P):
        for q in range(Q):
            v = np.array([[p], [q]])
            A = S @ v + T
            result += [A]
    # region II
    S = np.array([[1, -1], [0, 1]])
    T = np.array([[0], [0]])
    P = abs(beta)
    Q = abs(alpha)
    for p in range(P):
        for q in range(Q):
            v = np.array([[p], [q]])
            A = S @ v + T
            result += [A]
    # region III
    S = np.array([[1, 1], [0, 1]])
    T = np.array([[alpha], [alpha]])
    P = abs(beta)
    Q = abs(alpha)
    for p in range(P):
        for q in range(Q):
            v = np.array([[p], [q]])
            A = S @ v + T
            result += [A]
    # region IV
    S = np.array([[0, 1], [1, -1]])
    T = np.array([[alpha + beta], [-alpha]])
    P = abs(beta)
    Q = abs(alpha)
    for p in range(P):
        for q in range(Q):
            v = np.array([[p], [q]])
            A = S @ v + T
            result += [A]
    # region V
    S = np.array([[0, 1], [-1, 1]])
    T = np.array([[2 * alpha], [beta]])
    P = abs(beta)
    Q = abs(alpha)
    for p in range(P):
        for q in range(Q):
            v = np.array([[p], [q]])
            A = S @ v + T
            result += [A]
    # region VI
    S = np.array([[1, 0], [0, 1]])
    T = np.array([[alpha], [-alpha]])
    P = abs(beta)
    Q = abs(beta)
    for p in range(P):
        for q in range(Q):
            v = np.array([[p], [q]])
            A = S @ v + T
            result += [A]
    return result

def find_AB_points(n):
    n_perps = find_points_in_window(n)
    J = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    xs = []
    for i in range(len(n_perps)):
        B = n_perps[i]
        A = J @ B
        xs += [[A[0, 0], A[1, 0], B[0, 0], B[1, 0]]]
    xs = np.array(xs)
    xmin = np.min(xs[:, 0])
    ymin = np.min(xs[:, 1])
    for x in xs:
        x[0] -= xmin
        x[0] %= 1
        x[1] -= ymin
        x[1] %= 1
    return np.array(xs)

def get_index_list(array_list):
    hopping_indices = []
    ijs = []
    for array in array_list:
        i = array[0]
        j = array[1]
        xij = array[2]
        yij = array[3]
        if j < i:
            i, j, xij, yij = j, i, -xij, -yij
        if [i, j] not in ijs:
            ijs += [[i, j]]
            hopping_indices += [[i, j, xij, yij]]
    return sorted(hopping_indices)

def get_hopping_indices(xs, lattice_constant):
    hopping_indices = []
    coordinates = xs[:, 0:2]
    for i in range(len(coordinates)):
        for j in range(i, len(coordinates)):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            if distance < lattice_constant + 10**(-4) and distance > lattice_constant - 10**(-4):
                xij = (coordinates[i, 0] - coordinates[j, 0]) / lattice_constant
                yij = (coordinates[i, 1] - coordinates[j, 1]) / lattice_constant
                hopping_indices.append([i, j, xij, yij])
    hopping_indices = get_index_list(hopping_indices)
    return hopping_indices

def hopping_PBC(xs, lattice_constant):
    coordinates = xs[:, 0:2]
    N = len(coordinates)
    hopping_indices = []
    if n % 2 == 0:
        a = lattice_constant * (1 - 1 / np.sqrt(2))
    else:
        a = lattice_constant / np.sqrt(2)
    uc_0 = coordinates
    uc_1 = coordinates + np.array([np.max(coordinates[:, 0]) + a, 0])
    uc_2 = coordinates + np.array([np.max(coordinates[:, 0]) + a, -np.max(coordinates[:, 1]) - a])
    uc_3 = coordinates + np.array([0, -np.max(coordinates[:, 1]) - a])
    for i in range(N):
        for j in range(N):
            distance = np.linalg.norm(uc_0[i] - uc_0[j])
            if distance < lattice_constant + 10**(-4) and distance > lattice_constant - 10**(-4):
                xij = (uc_0[i, 0] - uc_0[j, 0]) / lattice_constant
                yij = (uc_0[i, 1] - uc_0[j, 1]) / lattice_constant
                hopping_indices.append([i, j, xij, yij])
            distance = np.linalg.norm(uc_0[i] - uc_1[j])
            if distance < lattice_constant + 10**(-4) and distance > lattice_constant - 10**(-4):
                xij = (uc_0[i, 0] - uc_1[j, 0]) / lattice_constant
                yij = (uc_0[i, 1] - uc_1[j, 1]) / lattice_constant
                hopping_indices.append([i, j, xij, yij])
            distance = np.linalg.norm(uc_0[i] - uc_2[j])
            if distance < lattice_constant + 10**(-4) and distance > lattice_constant - 10**(-4):
                xij = (uc_0[i, 0] - uc_2[j, 0]) / lattice_constant
                yij = (uc_0[i, 1] - uc_2[j, 1]) / lattice_constant
                hopping_indices.append([i, j, xij, yij])
            distance = np.linalg.norm(uc_0[i] - uc_3[j])
            if distance < lattice_constant + 10**(-4) and distance > lattice_constant - 10**(-4):
                xij = (uc_0[i, 0] - uc_3[j, 0]) / lattice_constant
                yij = (uc_0[i, 1] - uc_3[j, 1]) / lattice_constant
                hopping_indices.append([i, j, xij, yij])
    hopping_indices = get_index_list(hopping_indices)
    return hopping_indices

def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

def nearest_neighbor(coordinates, hopping_indices):
    inds = [[int(i), int(j)] for [i, j, x, y] in hopping_indices]
    inds = get_unique_list(inds)
    temp = []
    for i in range(len(inds)):
        [j, k] = inds[i]
        if not [k, j] in temp:
            temp += [[j, k]]
    inds = temp
    z = np.zeros(len(coordinates))
    for i in range(len(inds)):
        z[inds[i][0]] += 1
        z[inds[i][1]] += 1
    return z

def make_4D_cube():
    result = [np.array([0, 0, 0, 0]),
              np.array([1, 0, 0, 0]),
              np.array([0, 1, 0, 0]),
              np.array([0, 0, 1, 0]),
              np.array([0, 0, 0, 1]),
              np.array([1, 1, 0, 0]),
              np.array([1, 0, 1, 0]),
              np.array([1, 0, 0, 1]),
              np.array([0, 1, 1, 0]),
              np.array([0, 1, 0, 1]),
              np.array([0, 0, 1, 1]),
              np.array([1, 1, 1, 0]),
              np.array([1, 1, 0, 1]),
              np.array([1, 0, 1, 1]),
              np.array([0, 1, 1, 1]),
              np.array([1, 1, 1, 1])]
    return result

def make_M_matrix(n):
    d = Octonacci(n) + Octonacci(n - 1)
    nd = Octonacci(n)
    M = np.array([[d, 0, -nd, nd],
                  [0, d, -nd, -nd],
                  [-nd, -nd, d, 0],
                  [nd, -nd, 0, d]])
    return M

def make_window(n):
    cube_sites = make_4D_cube()
    M = make_M_matrix(n)
    sites = []
    for cube_site in cube_sites:
        A = M @ cube_site.reshape(4, 1)
        sites += [A[2:, :].reshape(2)]
    bonds = []
    a_w1 = np.linalg.norm(sites[0] - sites[3])
    a_w2 = np.linalg.norm(sites[0] - sites[1])
    for i in range(16):
        for j in range(i, 16):
            distance = np.linalg.norm(sites[i] - sites[j])
            if distance < a_w1 + 10**(-4) and distance > a_w1 - 10**(-4):
                bonds.append([i, j])
            elif distance < a_w2 + 10**(-4) and distance > a_w2 - 10**(-4):
                bonds.append([i, j])
    return sites, bonds

def get_bonds(coordinates, lattice_constant):
    hopping_indices = []
    for i in range(len(coordinates)):
        for j in range(i, len(coordinates)):
            distance = np.linalg.norm(coordinates[i, 0:2] - coordinates[j, 0:2])
            if distance < lattice_constant + 10**(-4) and distance > lattice_constant - 10**(-4):
                hopping_indices.append([i, j])
    return hopping_indices

def make_plot_uc(coordinates, hopping_indices):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    for hopping_index in hopping_indices:
        ax.plot([coordinates[hopping_index[0]][0], coordinates[hopping_index[1]][0]],
                [coordinates[hopping_index[0]][1], coordinates[hopping_index[1]][1]], c="darkblue")
    plt.savefig('unit_cell.png')
    plt.close()

def make_plot_hop(coordinates, hopping_indices, lattice_constant):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    for X in coordinates:
        ax.scatter(X[0], X[1], c="darkblue")
    for hopping_index in hopping_indices:
        xi = coordinates[hopping_index[0]][0]
        yi = coordinates[hopping_index[0]][1]
        xitoj = -hopping_index[2] * lattice_constant
        yitoj = -hopping_index[3] * lattice_constant
        ax.plot([xi, xi + xitoj], [yi, yi + yitoj], c = "darkblue")
    plt.savefig('hopping.png')
    plt.close()

def make_plot_real(xs, hopping_indices, z):
    fig, ax = plt.subplots(figsize=(13,10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    mappable = ax.scatter(xs[:, 0], xs[:, 1], c=z, cmap="plasma", zorder=3, s=10)
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label("Coordination number $z$", fontsize=32)
    for hopping_index in hopping_indices:
        ax.plot([xs[hopping_index[0], 0], xs[hopping_index[1], 0]],
                [xs[hopping_index[0], 1], xs[hopping_index[1], 1]], c = "gray")
    plt.savefig('real_space.png')
    plt.close()

def make_plot_window(x_ws, bond_ws, xs, z):
    fig, ax = plt.subplots(figsize=(13,10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    mappable = ax.scatter(xs[:, 2], xs[:, 3], c=z, cmap="plasma", s=10)
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label("Coordination number $z$", fontsize=32)
    for bond_w in bond_ws:
        ax.plot([x_ws[bond_w[0]][0], x_ws[bond_w[1]][0]],
                [x_ws[bond_w[0]][1], x_ws[bond_w[1]][1]], c = "black")
    plt.savefig('window_space.png')
    plt.close()

def make_plot_z(z):
    data = {3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
    for i in range(len(z)):
        data[int(z[i])] += 1
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel(r"Coordination number $z$", fontsize=32)
    ax.set_ylabel(r"Number of sites", fontsize=32)
    ax.set_xticks([3, 4, 5, 6, 7, 8])
    ax.set_xticklabels(['3', '4', '5', '6', '7', '8'])
    ax.minorticks_on()
    ax.set_xlim(2.5, 8.5)
    colors = ['navy', 'purple', 'palevioletred', 'darkorange', 'gold', 'yellow']
    for i in range(3, 9):
        ax.bar(i, data[i], capsize=10, color=colors[i - 3])
    plt.savefig('z_histogram.png')
    plt.close()

# Implementation
n = 3  # 调整n以控制大小（n=3生成小图案，n=5更大但计算密集）
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

print("Generated images: unit_cell.png, real_space.png, hopping.png, window_space.png, z_histogram.png")
