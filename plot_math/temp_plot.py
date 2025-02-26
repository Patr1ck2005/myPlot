import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.optimize import fsolve

# ==============================
# 初始参数及函数定义
# ==============================
alpha = 1
gamma = 1
gamma3 = 1
v = 1
omega3 = -5


def update_derived():
    """根据当前参数计算派生参数 a, b"""
    a = 2 * gamma + gamma3
    b = 4 * alpha * np.sqrt(gamma * gamma3) - 2 * gamma * omega3
    return a, b


a, b = update_derived()


def c(x, y):
    return -v ** 2 * x ** 2 * gamma3


def eq1(x, y):
    return y ** 3 - omega3 * y ** 2 - (v ** 2 * x ** 2 + 2 * alpha ** 2) * y + v ** 2 * x ** 2 * omega3


def eq2(x, y):
    return a * y ** 2 + b * y + c(x, y)


def system(vars):
    x, y = vars
    return [eq1(x, y), eq2(x, y)]


# 构建网格
x_vals = np.linspace(-10, 10, 512)
y_vals = np.linspace(-10, 10, 512)
X, Y = np.meshgrid(x_vals, y_vals)


def compute_solutions():
    """利用 fsolve 在整个区域内搜索交点"""
    initial_guesses = []
    for i in np.linspace(-10, 10, 21):
        for j in np.linspace(-10, 10, 21):
            initial_guesses.append([i, j])
    sols = []
    for guess in initial_guesses:
        sol, infodict, ier, mesg = fsolve(system, guess, full_output=True)
        if ier == 1:
            x_sol, y_sol = sol
            if -10 <= x_sol <= 10 and -10 <= y_sol <= 10:
                if np.abs(eq1(x_sol, y_sol)) < 1e-5 and np.abs(eq2(x_sol, y_sol)) < 1e-5:
                    if not any(np.allclose(sol, s, atol=1e-6) for s in sols):
                        sols.append(sol)
    if len(sols) > 0:
        return np.array(sols)
    else:
        return np.empty((0, 2))


solutions = compute_solutions()

# 用于轨迹显示的全局变量
show_trajectory = False  # 轨迹显示开关，初始为关闭
trajectory_points = []  # 用于保存所有更新过程中获得的交点

# ==============================
# 建立图形及初始绘图
# ==============================
fig = plt.figure(figsize=(10, 8))
# 主绘图区
ax_main = fig.add_axes([0.1, 0.3, 0.8, 0.65])
contour1_main = ax_main.contour(X, Y, eq1(X, Y), levels=[0], colors='blue', linewidths=2)
contour2_main = ax_main.contour(X, Y, eq2(X, Y), levels=[0], colors='red', linewidths=2)
if solutions.size > 0:
    points_plot, = ax_main.plot(solutions[:, 0], solutions[:, 1], 'ko', markersize=8,
                                fillstyle='none', label='BICs')
else:
    points_plot, = ax_main.plot([], [], 'ko', markersize=8, fillstyle='none', label='BICs')
ax_main.set_title('eq1 & eq2')
ax_main.set_xlabel('x')
ax_main.set_ylabel('y')
ax_main.set_xlim(-10, 10)
ax_main.set_ylim(-10, 10)
ax_main.grid(True)
ax_main.legend(loc='upper right')

# ==============================
# 建立 UI 控件（滑动条 + 文本框 + ±按钮）
# ==============================
slider_params = {
    'alpha': {'val': alpha, 'min': 0.1, 'max': 5, 'step': 0.01},
    'gamma': {'val': gamma, 'min': 0.1, 'max': 5, 'step': 0.01},
    'gamma3': {'val': gamma3, 'min': 0.1, 'max': 5, 'step': 0.01},
    'v': {'val': v, 'min': 0.1, 'max': 5, 'step': 0.01},
    'omega3': {'val': omega3, 'min': -10, 'max': 10, 'step': 0.1},
}

# 用来存放 slider、文本框和按钮对象
sliders = {}
plus_buttons = {}
minus_buttons = {}

slider_height = 0.03  # 控件高度
slider_spacing = 0.05  # 控件之间的垂直间隔
start_y = 0.2  # 第一个控件的 y 坐标

for i, param in enumerate(slider_params):
    y_position = start_y - i * slider_spacing
    # 滑动条
    ax_slider = fig.add_axes([0.15, y_position, 0.55, slider_height])
    slider = Slider(ax_slider, param, slider_params[param]['min'],
                    slider_params[param]['max'],
                    valinit=slider_params[param]['val'],
                    valstep=slider_params[param]['step'],
                    valfmt='%0.5f')
    sliders[param] = slider
    # “－”按钮
    ax_minus = fig.add_axes([0.78, y_position, 0.04, slider_height])
    minus_button = Button(ax_minus, '-')
    minus_buttons[param] = minus_button
    # “＋”按钮
    ax_plus = fig.add_axes([0.83, y_position, 0.04, slider_height])
    plus_button = Button(ax_plus, '+')
    plus_buttons[param] = plus_button

# ----------------------------
# 新增：轨迹显示开关按钮
# ----------------------------
ax_toggle = fig.add_axes([0.05, 0.85, 0.12, 0.05])
toggle_button = Button(ax_toggle, 'trajectory: OFF')  # 初始为关闭状态


# ==============================
# 定义更新回调函数
# ==============================
def update_all(val):
    global alpha, gamma, gamma3, v, omega3, a, b, solutions, trajectory_points

    # 从各 slider 读取最新数值
    alpha = sliders['alpha'].val
    gamma = sliders['gamma'].val
    gamma3 = sliders['gamma3'].val
    v = sliders['v'].val
    omega3 = sliders['omega3'].val


    # 更新派生参数
    a, b = update_derived()

    # 重新计算等高线数据
    Z1 = eq1(X, Y)
    Z2 = eq2(X, Y)

    # 重新求解交点
    solutions = compute_solutions()

    # 如果轨迹显示打开，则保存本次的交点到历史记录中
    if show_trajectory and solutions.size > 0:
        # 将当前所有交点追加到轨迹中
        trajectory_points.append(solutions.copy())

    # 清除主绘图区并重绘
    ax_main.cla()
    ax_main.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
    ax_main.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
    if solutions.size > 0:
        ax_main.plot(solutions[:, 0], solutions[:, 1], 'ko', markersize=8,
                     fillstyle='none', label='BICs')

    # 如果轨迹显示打开，则将历史交点轨迹一起绘制出来（使用淡灰色）
    if show_trajectory and trajectory_points:
        traj_all = np.vstack(trajectory_points)
        ax_main.scatter(traj_all[:, 0], traj_all[:, 1], c='gray', marker='o', s=20, alpha=0.5, label='轨迹')

    ax_main.set_title('eq1 & eq2')
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    ax_main.set_xlim(-10, 10)
    ax_main.set_ylim(-10, 10)
    ax_main.grid(True)
    ax_main.legend(loc='upper right')
    fig.canvas.draw_idle()


# 为所有 slider 绑定更新回调
for slider in sliders.values():
    slider.on_changed(update_all)


# 为 “＋” 和 “－” 按钮绑定回调
def make_plus_callback(param):
    def plus(event):
        current = sliders[param].val
        step = slider_params[param]['step']
        new_val = min(current + step, slider_params[param]['max'])
        sliders[param].set_val(new_val)

    return plus


def make_minus_callback(param):
    def minus(event):
        current = sliders[param].val
        step = slider_params[param]['step']
        new_val = max(current - step, slider_params[param]['min'])
        sliders[param].set_val(new_val)

    return minus


for param in sliders:
    plus_buttons[param].on_clicked(make_plus_callback(param))
    minus_buttons[param].on_clicked(make_minus_callback(param))


# ----------------------------
# 轨迹显示开关按钮的回调函数
# ----------------------------
def toggle_trajectory(event):
    global show_trajectory, trajectory_points
    # 切换轨迹显示状态
    show_trajectory = not show_trajectory
    if show_trajectory:
        toggle_button.label.set_text('trajectory: ON')
        # 开启时初始化轨迹数据为空
        trajectory_points = []
    else:
        toggle_button.label.set_text('trajectory: OFF')
        # 关闭时清除历史轨迹数据
        trajectory_points = []
    # 触发一次更新，刷新绘图
    update_all(None)


toggle_button.on_clicked(toggle_trajectory)

plt.show()
