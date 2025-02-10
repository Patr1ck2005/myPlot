import numpy as np


def lorenz_func(delta_omega, gamma=1, gamma_nr=0.01):
    """
    Calculate the Lorenz function value.

    Returns:
        float: The calculated Lorenz function value.
    """
    return (gamma ** 2) / ((delta_omega ** 2) + ((gamma+gamma_nr) ** 2))

def VBG_single_resonance_efficiency(kx, ky, omega=1.2, omega_Gamma=1.5, a=-1., gamma=0.1):
    """
    Calculate the efficiency of the VBG single resonance.

    Returns:
        float: The calculated efficiency.
    """
    kr = np.sqrt(kx ** 2 + ky ** 2)
    omega_0 = omega_Gamma+a*kr**2
    delta_omega = omega - omega_0
    gamma = 0.1*kr**1
    return lorenz_func(delta_omega=delta_omega, gamma=gamma)


def gaussian_profile(x, y, w=0.5):
    """
    Calculate the Gaussian profile.

    Returns:
        float: The calculated Gaussian profile value.
    """
    return np.exp(-((x ** 2 + y ** 2) / (w ** 2)))

