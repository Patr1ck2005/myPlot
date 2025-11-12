import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


def lorenz_func(delta_omega, gamma=1, gamma_nr=0.001):
    """
    Calculate the Lorenz function value.

    Returns:
        float: The calculated Lorenz function value.
    """
    # return (gamma ** 2) / ((delta_omega ** 2) + ((gamma+gamma_nr) ** 2))
    return -(gamma) / ((gamma + gamma_nr) - 1j * delta_omega)


# def VBG_single_resonance_converted(kx, ky, omega=1.499, omega_Gamma=1.5, a=-.3, Q_ref=0.1, gamma_slope=0.1 * 10,
#                                    gamma_nr=0.00):
def VBG_single_resonance_converted(kx, ky, omega=1.4, omega_Gamma=1.5, a=-1, Q_ref=1, gamma_slope=0.1,
                                   gamma_nr=0.00):
    """
    Calculate the converted beam of the VBG single resonance.

    Returns:
        complex: The calculated converted amplitude.
    """
    kr = np.sqrt(kx ** 2 + ky ** 2)
    omega_0 = omega_Gamma + a * kr ** 2
    delta_omega = omega - omega_0
    gamma = gamma_slope * kr ** 2 / (gamma_slope * Q_ref * kr ** 2 + 1)
    return lorenz_func(delta_omega=delta_omega, gamma=gamma, gamma_nr=gamma_nr)


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


def skyrmion_density(s1, s2, s3):
    # ----------------------
    #    Compute skyrmion density n_sk = S · (∂x S × ∂y S)
    #    Use centered finite differences via np.gradient with physical spacings
    # ----------------------
    # gradients of each component w.r.t x and y
    dS1dy, dS1dx = np.gradient(s1, 1, 1, edge_order=2)
    dS2dy, dS2dx = np.gradient(s2, 1, 1, edge_order=2)
    dS3dy, dS3dx = np.gradient(s3, 1, 1, edge_order=2)

    # cross product ∂x S × ∂y S (component-wise formula)
    cx = dS2dx * dS3dy - dS3dx * dS2dy
    cy = dS3dx * dS1dy - dS1dx * dS3dy
    cz = dS1dx * dS2dy - dS2dx * dS1dy

    nsk = s1 * cx + s2 * cy + s3 * cz  # skyrmion density
    return nsk


def skyrmion_number(nsk, dx=1, dy=1, mask=None, show=False):
    # ----------------------
    #    Compute skyrmion number s = (∑_i nsk_i * dx * dy) / (4π)
    # ----------------------
    if mask is None:
        mask = np.ones_like(nsk, dtype=bool)
    s = (nsk[mask].sum() * dx * dy) / (4.0 * np.pi)

    print(f"Computed skyrmion number s ≈ {s:.6f}")
    if show:
        fig, ax = plt.subplots(figsize=(2, 2))
        # 绘制遮罩后数据
        im = ax.imshow(nsk.T * mask.T, origin='lower', extent=[-1, 1, -1, 1],
                       cmap='viridis')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Skyrmion density')
        ax.set_title(f'Skyrmion density, s ≈ {s:.6f}')
        plt.show()
        plt.close()
    # print(f"Grid: {N}x{N}, box size: [-{L}, {L}]^2, dx=dy={dx:.4f}, radius R={R}")
    return s


def ellipse2stokes(phi, tanchi):
    """
    Convert ellipse parameters to Stokes parameters.
    """
    s1 = np.cos(2 * phi) * (1 - tanchi ** 2) / (1 + tanchi ** 2)
    s2 = np.sin(2 * phi) * (1 - tanchi ** 2) / (1 + tanchi ** 2)
    s3 = 2 * tanchi / (1 + tanchi ** 2)
    return s1, s2, s3


def stokes2ellipse(s1, s2, s3):
    pass


from scipy import ndimage


def divide_regions_by_zero(z, X=None, Y=None, visualize=False):
    """
    以值为 0 作为分界线划分二维数组的区域，生成 mask，并可选可视化。

    参数:
    z (np.ndarray): 二维数组 (height, width)。
    X (np.ndarray, optional): 网格 x 坐标，与 z 同形状。如果 None，则使用 np.arange(z.shape[1])。
    Y (np.ndarray, optional): 网格 y 坐标，与 z 同形状。如果 None，则使用 np.arange(z.shape[0])。
    visualize (bool, optional): 是否绘制 =0 曲线和区域。默认 False。

    返回:
    dict: {
        'n': 总区域数 (int),
        'num_pos': 正区域数 (int),
        'num_neg': 负区域数 (int),
        'masks': n 个 mask 的列表 (list of bool np.ndarray),
        'middle_mask': 中间区域的 mask (bool np.ndarray 或 None),
    }
    """
    if X is None:
        X = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))[0]
    if Y is None:
        Y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))[1]

    # 符号数组
    sign_z = np.sign(z)

    # 标签正区域 (z > 0)
    positive = (sign_z > 0).astype(int)
    pos_labels, num_pos = ndimage.label(positive)

    # 标签负区域 (z < 0)，偏移标签
    negative = (sign_z < 0).astype(int)
    neg_labels, num_neg = ndimage.label(negative)
    neg_labels[neg_labels > 0] += num_pos

    # 总区域数 n
    n = num_pos + num_neg

    # 生成所有 mask
    masks = []
    for i in range(1, n + 1):
        if i <= num_pos:
            mask = (pos_labels == i)
        else:
            mask = (neg_labels == i)
        masks.append(mask)

    # 选择中间区域
    center_y, center_x = z.shape[0] // 2, z.shape[1] // 2
    center_sign = sign_z[center_y, center_x]
    middle_label = None
    if center_sign > 0:
        middle_label = pos_labels[center_y, center_x]
    elif center_sign < 0:
        middle_label = neg_labels[center_y, center_x]
    else:
        print("警告: 中心点在边界上，无法直接选择中间区域。")

    middle_mask = masks[middle_label - 1] if middle_label is not None else None

    # 可视化
    if visualize:
        plt.figure(figsize=(4, 3))
        plt.contour(X, Y, z, levels=[0], colors='red', linewidths=2)  # =0 曲线
        plt.contourf(X, Y, pos_labels + neg_labels, cmap='tab20', alpha=0.5)  # 区域填充
        plt.title('=0 Boundary Line and Divided Regions')
        plt.colorbar()
        plt.show()

    return {
        'n': n,
        'num_pos': num_pos,
        'num_neg': num_neg,
        'masks': masks,
        'middle_mask': middle_mask
    }


if __name__ == '__main__':
    # 示例使用：生成同心圆环数组并调用函数
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)
    z = (r - 1) * (r - 2)  # 示例数组

    result = divide_regions_by_zero(z, X=X, Y=Y, visualize=True)

    # 输出示例结果
    print(f"总区域数 n: {result['n']}")
    print(f"正区域数: {result['num_pos']}, 负区域数: {result['num_neg']}")
    print(f"中间 mask 形状: {result['middle_mask'].shape if result['middle_mask'] is not None else 'None'}, "
          f"非零元素数: {np.sum(result['middle_mask']) if result['middle_mask'] is not None else 'N/A'}")
    print("所有 mask 已生成，可供选择（result['masks'] 列表）。")
