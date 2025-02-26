import numpy as np
import matplotlib.pyplot as plt

# 定义常数
alpha = 1
gamma = 1
gamma3 = 1
v = 1
omega3 = -5

a = 2 * gamma + gamma3
b = 4 * alpha * np.sqrt(gamma * gamma3) - 2 * gamma * omega3

# 定义辅助函数
def c(x, y):
    return -v**2 * x**2 * gamma3

def eq1(x, y):
    return y**3 - omega3 * y**2 - (v**2 * x**2 + 2 * alpha**2) * y + v**2 * x**2 * omega3

def eq2(x, y):
    return a * y**2 + b * y + c(x, y)

# 创建网格并计算函数值
x_vals = np.linspace(-10, 10, 512)
y_vals = np.linspace(-10, 10, 512)
X, Y = np.meshgrid(x_vals, y_vals)
Z1 = eq1(X, Y)
Z2 = eq2(X, Y)

# 绘制等高线
plt.figure(figsize=(8, 8))
contour1 = plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
contour2 = plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
plt.title('eq1 & eq2 等高线与交点 (不用 fsolve)', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.grid(True)

# ------------------------------
# 提取等高线的线段 (用 allsegs 属性)
# ------------------------------

def get_segments_from_array(arr):
    """
    将一个二维数组 (n_points x 2) 分割成连续线段列表。
    """
    segments = []
    for i in range(len(arr) - 1):
        segments.append((arr[i], arr[i + 1]))
    return segments

# 从 allsegs 中提取所有线段
segments1 = []
for seg_list in contour1.allsegs:
    for arr in seg_list:
        segments1.extend(get_segments_from_array(arr))

segments2 = []
for seg_list in contour2.allsegs:
    for arr in seg_list:
        segments2.extend(get_segments_from_array(arr))

# ------------------------------
# 定义计算两线段交点的函数
# ------------------------------
def line_intersect(p1, p2, p3, p4):
    """
    计算两线段 (p1, p2) 和 (p3, p4) 的交点。
    若线段相交，则返回交点坐标 (xi, yi)；否则返回 None。
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if np.abs(denom) < 1e-10:
        return None  # 平行或共线
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        xi = x1 + t * (x2 - x1)
        yi = y1 + t * (y2 - y1)
        return (xi, yi)
    else:
        return None

# ------------------------------
# 计算所有线段之间的交点
# ------------------------------
intersections = []
for seg1 in segments1:
    for seg2 in segments2:
        pt = line_intersect(seg1[0], seg1[1], seg2[0], seg2[1])
        if pt is not None:
            intersections.append(pt)

# 去除重复的交点（用一定的容差）
unique_points = []
for pt in intersections:
    if not any(np.allclose(pt, up, atol=1e-4) for up in unique_points):
        unique_points.append(pt)
unique_points = np.array(unique_points)

# 绘制交点
if unique_points.size > 0:
    plt.plot(unique_points[:, 0], unique_points[:, 1], 'ko', markersize=8, label='交点')
    plt.legend(fontsize=12)
else:
    print("没有找到交点")

plt.show()

# 输出所有交点的坐标
print("交点坐标:")
for pt in unique_points:
    print(pt)
