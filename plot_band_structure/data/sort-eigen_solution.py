import pandas as pd

# Step 1: 从 txt 文件加载数据
# filename = 'VBG-II-shrink_type-1stFP-meshed'
# filename = 'VBG_III_s1mple'
# filename = 'VBG-II-shrink_type-1stFP-uncoupled-meshed'
filename = 'VBG-band3D-final_design.csv'
# filename = 'merging_BICs-band3D.csv'
# filename = 'merging_BICs-band3D-0.01.csv'
# filename = 'VBG-band3D-comparison_design.csv'
# filename = 'VBG-band3D-homo_layer.csv'

data = pd.read_csv(filename, sep='\t')

# Step 2: 将特征频率列中的复数部分转换为适当的格式（假设用 'i' 表示虚数部分）
data['Eigenfrequency (THz)'] = data['特征频率 (THz)'].apply(lambda x: complex(x.replace('i', 'j')).real)

# Step 3: 为每组 (m1, m2) 计算特征解的序号
# 首先对每组 (m1, m2) 按特征频率排序
data['Eigenfrequency Rank'] = data.groupby(['m1', 'm2'])['Eigenfrequency (THz)'].rank(method='first', ascending=True).astype(int)

# Step 4: 查看结果
data = data.drop(['Eigenfrequency (THz)'], axis=1)

# Step 5: 将数据保存到一个新的文件
data.to_csv(f'sorted_{filename}', sep='\t', index=False)
