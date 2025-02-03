import pandas as pd
import numpy as np

# 读取原始数据
# df = pd.read_csv('sorted_VBG-band3D-final_design.csv', sep='\t')
df = pd.read_csv('sorted_VBG-band3D-final_design.csv', sep='\t')
# df = pd.read_csv('sorted_VBG-band3D-comparison_design.csv', sep='\t')
# df = pd.read_csv('expanded_VBG-band3D-comparison_design.csv', sep='\t')

m_max = df['m1'].max()
m_step = 0.005
# 假设原始数据中的 'm1' 和 'm2' 是坐标列
# 并且其他列是函数值

# 原始坐标范围
original_range = np.linspace(0, m_max, int(m_max / m_step + 1))

# 扩展后的坐标范围
expanded_range = np.linspace(-m_max, m_max, int(2 * m_max / m_step + 1))


# 函数来生成 C4 对称坐标
def generate_c4_symmetry_coords(x, y):
    """根据 C4 对称性扩展坐标 (x, y) 到 -2 到 2 范围"""
    coords = [(x, y), (-y, x), (-x, -y), (y, -x)]
    return coords


def generate_anti_symmetry_coords(x, y):
    coords = [(y, x)]
    return coords


# 创建一个字典来存储坐标和函数值
expanded_dict = {
    'coordinates': [],
    'function_values': []
}

for _, row in df.iterrows():
    m1 = row['m1']
    m2 = row['m2']

    anti_coords = generate_anti_symmetry_coords(m1, m2)
    for i, anti_coord in enumerate(anti_coords):
        # 存储坐标
        expanded_dict['coordinates'].append(anti_coord)

        # 存储对应的函数值，假设除了'm1'和'm2'以外的列为函数值
        function_values = row.drop(['m1', 'm2']).values
        m1 = anti_coord[0]
        m2 = anti_coord[1]
        theta = np.arctan(m1/m2) if m2 != 0 else np.pi/2
        function_values[2] -= 2*(theta-np.pi/4)
        expanded_dict['function_values'].append(function_values)
    expanded_dict['coordinates'].append((m2, m1))
    function_values = row.drop(['m1', 'm2']).values
    expanded_dict['function_values'].append(function_values)


# 转换为二维数组：coordinates 为 2D 数组
expanded_dict['coordinates'] = np.array(expanded_dict['coordinates'])

# 转换为三维数组：function_values 为 3D 数组
expanded_dict['function_values'] = np.array(expanded_dict['function_values'])

# 显示字典内容（可选）
print("Coordinates Array:")
print(expanded_dict['coordinates'])
print("\nFunction Values Array:")
print(expanded_dict['function_values'])

# 如果需要，也可以将扩展后的数据保存为新的 CSV 文件
expanded_df = pd.DataFrame(expanded_dict['coordinates'], columns=['m1', 'm2'])
for i, func_col in enumerate(df.columns[2:]):
    expanded_df[func_col] = expanded_dict['function_values'][:, i]

expanded_df.to_csv('temp-expanded_SOP.csv', sep='\t', index=False)

# 读取原始数据
df = pd.read_csv('temp-expanded_SOP.csv', sep='\t')

# 原始坐标范围
original_range = np.linspace(0, m_max, int(m_max / m_step + 1))

# 扩展后的坐标范围
expanded_range = np.linspace(-m_max, m_max, int(2 * m_max / m_step + 1))


# 创建一个字典来存储坐标和函数值
expanded_dict = {
    'coordinates': [],
    'function_values': []
}

for _, row in expanded_df.iterrows():
    m1 = row['m1']
    m2 = row['m2']
    # # 生成 C4 对称性的坐标
    expanded_coords = generate_c4_symmetry_coords(m1, m2)
    # 为每个生成的坐标创建新的行并保存函数值
    for i, coord in enumerate(expanded_coords):
        # 存储坐标
        expanded_dict['coordinates'].append(coord)

        # 存储对应的函数值，假设除了'm1'和'm2'以外的列为函数值
        function_values = row.drop(['m1', 'm2']).values
        function_values[2] += np.pi/2*i
        expanded_dict['function_values'].append(function_values)


# 转换为二维数组：coordinates 为 2D 数组
expanded_dict['coordinates'] = np.array(expanded_dict['coordinates'])

# 转换为三维数组：function_values 为 3D 数组
expanded_dict['function_values'] = np.array(expanded_dict['function_values'])

# 显示字典内容（可选）
print("Coordinates Array:")
print(expanded_dict['coordinates'])
print("\nFunction Values Array:")
print(expanded_dict['function_values'])

# 如果需要，也可以将扩展后的数据保存为新的 CSV 文件
expanded_df = pd.DataFrame(expanded_dict['coordinates'], columns=['m1', 'm2'])
for i, func_col in enumerate(df.columns[2:]):
    expanded_df[func_col] = expanded_dict['function_values'][:, i]

expanded_df.to_csv('expanded_VBG-final_design.csv', sep='\t', index=False)

print("坐标和函数值已成功存储并保存为 'expanded_VBG-final_design.csv'.")
