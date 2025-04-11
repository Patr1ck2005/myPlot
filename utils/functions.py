import numpy as np
from scipy.interpolate import griddata


def lorenz_func(delta_omega, gamma=1, gamma_nr=0.001):
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
    gamma = 0.1*kr**2
    return lorenz_func(delta_omega=delta_omega, gamma=gamma)


def gaussian_profile(x, y, w=0.5, l=0):
    """
    Calculate the Gaussian profile.

    Returns:
        float: The calculated Gaussian profile value.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    return np.exp(-(r ** 2 / (w ** 2))) * r ** (l)


def interpolate_2d(z, x_new, y_new):
    """
    对已知z数组进行二维插值计算，返回在(x_new, y_new)处的插值结果。

    参数:
        z: 二维数组，数据必须是一个规则的网格（例如形状为(m, n)）。
        x_new: 查询点的x坐标，可以是单个值或者数组。
        y_new: 查询点的y坐标，可以是单个值或者数组。

    返回:
        z_new: 在(x_new, y_new)处插值得到的z值。
    """
    # 生成与z对应的网格坐标
    # 这里假设x轴和y轴都在[-1,1]区间内均匀取样
    x = np.linspace(-2, 2, z.shape[0])
    y = np.linspace(-2, 2, z.shape[1])

    # 构建二维网格（注意这里用 indexing='ij' 保证第一个维度对应 x）
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # 将网格拉平得到点集
    points = np.column_stack((xx.ravel(), yy.ravel()))

    # 将z也拉平成一维数组，与points中的每个点对应
    values = z.ravel()

    # 使用 griddata 进行插值
    # method 可以换成 'linear' 或 'nearest'，如果 cubic 存在精度问题可以尝试其他方法
    z_new = griddata(points, values, (x_new, y_new), method='cubic')

    return z_new

