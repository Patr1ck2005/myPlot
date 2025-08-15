import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def generate_vortex_metasurface_structure(size=20, topological_charge=1, unit_width=0.6, unit_height=0.2):
    """
    生成一个产生涡旋光的超透镜（metasurface）的结构示例图案。
    这是一个简化模型，使用几何相位（Pancharatnam-Berry phase）原理，通过旋转各向异性纳米柱（这里简化为矩形单元）来实现螺旋相位。

    参数:
    - size: 网格大小（单元数），默认20x20，表示超透镜由size x size个单元组成
    - topological_charge: 拓扑荷l（整数），决定涡旋的螺旋度，默认1
    - unit_width: 每个单元的宽度（相对值），默认0.6
    - unit_height: 每个单元的高度（相对值），默认0.2（宽度 > 高度表示各向异性）

    该函数会生成一个SVG文件 'vortex_metasurface_structure.svg'，其中每个矩形代表一个纳米柱，其旋转角度根据位置计算，以产生涡旋相位。
    """
    # 创建坐标网格（单元中心位置）
    x = np.linspace(-size // 2 + 0.5, size // 2 - 0.5, size)
    y = np.linspace(-size // 2 + 0.5, size // 2 - 0.5, size)
    X, Y = np.meshgrid(x, y)

    # 计算极坐标
    theta = np.arctan2(Y, X)

    # 对于几何相位，旋转角度 alpha = (l * theta) / 2
    # 这会引入相位 phi = 2 * alpha = l * theta（对于圆偏振光）
    rotation_angles = (topological_charge * theta) / 2 * (180 / np.pi)  # 转换为度，因为matplotlib使用度

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-size // 2, size // 2)
    ax.set_ylim(-size // 2, size // 2)
    ax.axis('off')  # 隐藏轴

    # 为每个单元添加旋转矩形
    for i in range(size):
        for j in range(size):
            # 矩形中心位置
            cx, cy = X[i, j], Y[i, j]
            radius = np.sqrt(cx ** 2 + cy ** 2)
            if radius > size / 2:
                continue  # 跳过超出范围的单元
            angle = rotation_angles[i, j]

            # 创建矩形（锚点在左下角，但我们通过xy调整为中心）
            rect = Rectangle(
                (cx - unit_width / 2, cy - unit_height / 2),  # 左下角位置
                unit_width,
                unit_height,
                angle=angle,  # 旋转角度（度）
                rotation_point='center',
                edgecolor='black',
                facecolor='gray',
                linewidth=0.5
            )
            ax.add_patch(rect)

    # 保存为SVG
    plt.savefig('vortex_metasurface_structure.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.title(f'Vortex Metasurface Structure (Topological Charge = {topological_charge})')
    plt.show()  # 可选：显示图像


# 生成结构图案
generate_vortex_metasurface_structure(
    size=16,
    unit_width=0.6,
    unit_height=0.2,
    topological_charge=3,
)
