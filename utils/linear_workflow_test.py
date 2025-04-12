import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 1. 生成一维均匀数组，取值从 0 到 1，共256个数
gradient = np.linspace(0, 1, 256)

gradient = np.power(gradient, 1)

# 2. 将一维数组复制成二维数组，生成一幅256x256的灰度图像，每一行均为渐变
img = np.tile(gradient, (256, 1))

# 3. 保存为png灰度图
plt.imsave("gradient.png", img, cmap='gray')
print("图像已保存为 gradient.png")

# 4. 读取保存的图像
img_read = plt.imread("gradient.png")

# 5. 取出第128行的像素值
row_index = 128
row_values = img_read[row_index]

# 打印第128行的像素值数组
print(f"读取图像中第 {row_index} 行的像素值：")
print(row_values)

# 6. 绘制第128行像素值曲线
plt.figure(figsize=(8, 4))
plt.plot(row_values, label=['R', 'G', 'B', 'A'], color='k')
plt.xlabel("列索引")
plt.ylabel("灰度值")
plt.title(f"第 {row_index} 行的像素值曲线")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 也可以保存曲线图为文件，例如：
plt.savefig("row_128_curve.png")
print("像素值曲线图已保存为 row_128_curve.png")

# 显示图形
plt.show()
