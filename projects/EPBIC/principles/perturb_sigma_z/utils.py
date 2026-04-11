import numpy as np

def build_H(S, omega, gamma, delta_w, delta_g, kappa, omega3):
    """
    3x3 non-Hermitian Hamiltonian

    delta_tilde = delta_w - i delta_g
    H_pl = (omega - i gamma) I + delta_tilde sigma_z
    coupling vector = kappa (cos theta, sin theta), theta = 2 pi S
    """
    theta = 2 * np.pi * S
    delta_tilde = delta_w - 1j * delta_g

    H = np.array([
        [omega - 1j*gamma + delta_tilde, 0, kappa*np.cos(theta)],
        [0, omega - 1j*gamma - delta_tilde, kappa*np.sin(theta)],
        [kappa*np.cos(theta), kappa*np.sin(theta), omega3]
    ], dtype=complex)
    return H


def eig_sorted(H):
    vals, vecs = np.linalg.eig(H)
    idx = np.argsort(vals.real)
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs


def left_eigvectors(H):
    # left eigenvectors of H: eigenvectors of H^\dagger
    vals_l, vecs_l = np.linalg.eig(H.conj().T)
    return vals_l, vecs_l


def biorthogonal_data(H):
    """
    Returns eigenvalues, right eigenvectors, matched left eigenvectors,
    and phase rigidity r_n = |<L_n|R_n>| / sqrt(<R_n|R_n><L_n|L_n>)
    """
    vals_r, VR = np.linalg.eig(H)
    vals_l, VL = np.linalg.eig(H.conj().T)

    # match left eigenvectors by conjugate eigenvalue
    matched_L = np.zeros_like(VR, dtype=complex)
    rigidity = np.zeros(len(vals_r))

    for i, lam in enumerate(vals_r):
        j = np.argmin(np.abs(vals_l - np.conj(lam)))
        L = VL[:, j]
        R = VR[:, i]

        # normalize for numerical stability
        if np.linalg.norm(R) > 0:
            R = R / np.linalg.norm(R)
        if np.linalg.norm(L) > 0:
            L = L / np.linalg.norm(L)

        matched_L[:, i] = L
        num = np.abs(np.vdot(L, R))
        den = np.sqrt(np.vdot(R, R).real * np.vdot(L, L).real)
        rigidity[i] = num / den if den > 0 else 0.0

    idx = np.argsort(vals_r.real)
    return vals_r[idx], VR[:, idx], matched_L[:, idx], rigidity[idx]


def pairwise_min_gap(vals):
    d12 = np.abs(vals[0] - vals[1])
    d13 = np.abs(vals[0] - vals[2])
    d23 = np.abs(vals[1] - vals[2])
    return min(d12, d13, d23)
