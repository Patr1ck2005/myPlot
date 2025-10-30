import matplotlib.pyplot as plt
import numpy as np

# 假设这是你已经定义好的 plot_polar_line 函数
def plot_polar_line(ax, angles, values, index=0, **kwargs):
    angle_unit = kwargs.get('angle_unit', 'degrees')
    scale = kwargs.get('scale', 1.0)
    alpha_line = kwargs.get('alpha_line', 0.8)
    default_color = kwargs.get('default_color', None)
    linewidth_base = kwargs.get('linewidth_base', 1)
    set_r_lim = kwargs.get('set_r_lim', True)
    r_min = kwargs.get('r_min', 0)
    r_max = kwargs.get('r_max', None)
    angle_offset_degrees = kwargs.get('angle_offset_degrees', 0)

    n = len(angles)
    if len(values) != n:
        raise ValueError("values 长度必须与 angles 一致")

    if angle_unit == 'degrees':
        theta_rad = np.deg2rad(angles + angle_offset_degrees) # 应用偏移
    elif angle_unit == 'radians':
        theta_rad = angles + np.deg2rad(angle_offset_degrees)
    else:
        raise ValueError("angle_unit 必须是 'degrees' 或 'radians'")

    r_vals = np.maximum(0, values) * scale

    if default_color is None:
        default_color = plt.cm.tab10(index % 10)

    ax.plot(theta_rad, r_vals, color=default_color, linewidth=linewidth_base, alpha=alpha_line, label=f'Intensity {index+1}')

    if set_r_lim:
        if r_max is None:
            current_r_max = np.max(r_vals) if r_vals.size > 0 else 0
            r_max_val = np.maximum(current_r_max * 1.1, 0.1)
        else:
            r_max_val = r_max
        ax.set_rlim(r_min, r_max_val)

    return ax


if __name__ == '__main__':

    # ----- 示例：调整半圆朝向 -----

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    base_angles = np.linspace(0, 180, 100) # 基础数据为 0 到 180 度的半圆
    base_values = 10.0 + 0.8 * np.sin(np.deg2rad(base_angles * 2))

    # 1. 默认朝向（上半圆，0度在右，逆时针）
    ax0 = axes[0]
    plot_polar_line(ax0, base_angles, base_values, index=0, default_color='blue')
    ax0.set_title('Upper Half-Circle (Default Settings)')
    ax0.set_thetalim(np.deg2rad(0), np.deg2rad(180)) # 限制显示范围
    ax0.grid(True)


    # 2. 右半圆（0度在上方，顺时针，数据范围调整）
    ax1 = axes[1]
    # 调整输入角度数据范围，使其本身就描述右半圆
    angles_right = np.linspace(-90, 90, 100)
    values_right = 1.0 + 0.8 * np.sin(np.deg2rad(angles_right * 2))
    plot_polar_line(ax1, angles_right, values_right, index=0, default_color='red')
    ax1.set_theta_zero_location('N') # 0度在上方
    ax1.set_theta_direction(-1)      # 顺时针
    ax1.set_thetalim(np.deg2rad(-90), np.deg2rad(90)) # 限制显示范围
    ax1.set_title('Right Half-Circle (Data & Axis Settings)')
    ax1.grid(True)


    # 3. 左半圆（使用 angle_offset_degrees 旋转）
    ax2 = axes[2]
    # 使用原始的 0-180 数据，但通过 angle_offset_degrees 旋转 90 度
    plot_polar_line(ax2, base_angles, base_values, index=0, default_color='green', angle_offset_degrees=90)
    ax2.set_theta_zero_location('N') # 0度在上方
    ax2.set_theta_direction(-1)      # 顺时针
    ax2.set_thetalim(np.deg2rad(90), np.deg2rad(270)) # 限制显示范围
    ax2.set_title('Left Half-Circle (Rotated by angle_offset_degrees)')
    ax2.grid(True)


    # 4. 下半圆（0度在上方，通过 set_theta_zero_location 和 thetalim 调整）
    ax3 = axes[3]
    # 保持数据为 0-180 的上半圆，但将 0 度放在下方，并限制显示范围
    plot_polar_line(ax3, base_angles, base_values, index=0, default_color='purple')
    ax3.set_theta_zero_location('S') # 0度在下方
    ax3.set_theta_direction(-1)      # 顺时针
    ax3.set_thetalim(np.deg2rad(0), np.deg2rad(180)) # 限制显示范围
    ax3.set_title('Lower Half-Circle (theta_zero_location)')
    ax3.grid(True)


    plt.tight_layout()
    plt.show()
