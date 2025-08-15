import numpy as np
import pyvista as pv

# 椭圆参数
a, b = 2, 1
n = 100  # 离散的点数

# 生成 x, y 坐标网格
x = np.linspace(-a, a, n)
y = np.linspace(-b, b, n)
X, Y = np.meshgrid(x, y)

# 限定区域：只保留椭圆内的点
mask = (X/a)**2 + (Y/b)**2 <= 1

# 构造曲面1：椭圆抛物面
Z1 = (X/a)**2 + (Y/b)**2
# 对超出椭圆区域的点赋值 nan（不会显示）
Z1[~mask] = np.nan

# 构造曲面2：倒置的椭圆抛物面
Z2 = 1 - (X/a)**2 - (Y/b)**2
Z2[~mask] = np.nan

# 使用 StructuredGrid 构造 PyVista 网格对象
grid1 = pv.StructuredGrid(X, Y, Z1)
grid2 = pv.StructuredGrid(X, Y, Z2)

# 设置绘图器
p = pv.Plotter()
p.add_mesh(grid1, color='blue', opacity=1, show_edges=True, label='Surface 1')
p.add_mesh(grid2, color='red', opacity=1, show_edges=True, label='Surface 2')

# 添加图例和标题
p.add_legend()
p.add_text("两个相交的椭圆曲面", font_size=14)

# 显示结果
p.show()
