import matplotlib.pyplot as plt


def plot_two_scales(
        x, y1, y2,
        figsize=(8, 6),
        x_label='X-Axis',
        x_ticks=None,
        y1_label='Y1-Axis', y2_label='Y2-Axis',
        y1_lim=None, y2_lim=None,
        y1_ticks=(), y2_ticks=(),
        y1_color='tab:blue', y2_color='tab:red',
        title='Dual Y-Axis Plot',
        y1_marker='o', y2_marker='s',
        save_name='DEFAULT',
        **kwargs,
):
    """
    绘制带有双 y 轴的通用函数

    参数:
    - x: 横轴的数据 (list or array)
    - y1: 左 y 轴的数据 (list or array)
    - y2: 右 y 轴的数据 (list or array)
    - x_label: 横轴的标签 (str)
    - y1_label: 左 y 轴的标签 (str)
    - y2_label: 右 y 轴的标签 (str)
    - y1_color: 左 y 轴线条的颜色 (str)
    - y2_color: 右 y 轴线条的颜色 (str)
    - title: 图表标题 (str)
    - y1_marker: 左 y 轴线的标记样式 (str)
    - y2_marker: 右 y 轴线的标记样式 (str)
    """
    plt.rcParams['font.size'] = 14
    # 创建图形和左 y 轴
    fig, ax1 = plt.subplots(figsize=figsize)

    if x_ticks is not None:
        ax1.set_xticks(x_ticks)
    ax1.set_xlabel(x_label)

    # 绘制左 y 轴的折线图
    ax1.set_ylabel(y1_label, color=y1_color)
    if y1_lim:
        ax1.set_ylim(*y1_lim)
    ax1.plot(
        x, y1,
        color=y1_color,
        marker=y1_marker,
        label=y1_label,
        **kwargs
    )
    ax1.tick_params(axis='y', labelcolor=y1_color)
    # ax1.grid(True, linestyle='--', alpha=0.5)

    # 创建右 y 轴
    ax2 = ax1.twinx()
    if y2_lim:
        ax2.set_ylim(*y2_lim)
    ax2.set_ylabel(y2_label, color=y2_color)
    ax2.plot(
        x, y2,
        color=y2_color,
        marker=y2_marker,
        label=y2_label,
        **kwargs
    )
    ax2.tick_params(axis='y', labelcolor=y2_color)

    # 添加标题
    plt.title(title)

    # 调整布局并显示图形
    plt.tight_layout()
    plt.savefig(f'./rsl/dual_y_axis_fig-{save_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.3, transparent=True)
    plt.show()


# 示例用法
if __name__ == "__main__":
    # 测试数据
    time = [0, 1, 2, 3, 4, 5]  # 时间 (s)
    speed = [0, 10, 20, 30, 40, 50]  # 速度 (m/s)
    acceleration = [0, 2, 4, 3, 2, 1]  # 加速度 (m/s^2)

    plot_two_scales(
        time, speed, acceleration,
        x_label='Time (s)',
        y1_label='Speed (m/s)',
        y2_label='Acceleration (m/s²)',
        y1_color='green',
        y2_color='orange',
        title='Time vs Speed and Acceleration',
        y1_marker='x',
        y2_marker='d'
    )
