import pandas as pd
import matplotlib.pyplot as plt

# 定义数据文件名
filename = './data/EP_band-field.txt'

# 先读取所有行，找出第一个非注释行所在位置
with open(filename, 'r') as f:
    lines = f.readlines()

# 找到第一行不以 '%' 开头的行号
for i, line in enumerate(lines):
    if not line.strip().startswith('%'):
        header_line_index = i
        break

# 利用 pandas 读取数据，并跳过头部的注释行
data = pd.read_csv(filename, delim_whitespace=True, skiprows=header_line_index)

# 查看所有列名称
print("所有列名:")
print(data.columns.tolist())

# 定义你需要的参数和特征解（可按需要修改）
target_a = 'a=0.015'
target_lambda = 'lambda=3E-12'

# 在列名称中寻找包含目标参数和特征解的 Ez 列
target_cols = [col for col in data.columns if ('ewfd.Ez' in col) and (target_a in col) and (target_lambda in col)]
if len(target_cols) == 0:
    print("没有找到对应的 Ez 数据列，请检查参数设置！")
else:
    target_col = target_cols[0]
    print("目标列:", target_col)

    # 绘图：横坐标 X，纵坐标为目标 Ez 数据
    plt.figure(figsize=(8, 5))
    plt.plot(data['X'], data[target_col], 'b-', label=target_col)
    plt.xlabel('X (µm)')
    plt.ylabel('Ez (V/m)')
    plt.title(f'Ez场分布: {target_col}')
    plt.legend()
    plt.grid(True)
    plt.show()
