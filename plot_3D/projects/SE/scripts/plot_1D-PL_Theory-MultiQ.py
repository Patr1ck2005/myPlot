import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_3D.core.plot_pipeline import draw_shapes, add_annotations  # 导入函数

c_const = 299792458

# fontsize
fs = 12
plt.rcParams.update({'font.size': fs})

# 步骤1: 加载CSV数据
# csv_file = '../data/SE/1fold-TM-k_loss0-Purcell.csv'  # 指定路径
# csv_file = './rsl/2D/raw_data_20250914_210552/data_curve-1.csv'  # 指定路径
# csv_file = './rsl/2D/raw_data_20250914_210616/data_curve-1.csv'  # 指定路径
csv_file = '../../../rsl/2D/multi_Qs.csv'  # 指定路径
df = pd.read_csv(csv_file)

# 打印预览
print("原始数据预览:")
print(df.head())

# 步骤2: 简单后处理
# 2.1: 重命名列（假设'f_Hz', 'Y'；调整如果不同）
df.columns = ['X', 'Y1', 'Y2', 'Y3']

# 2.2: 去除NaN
df = df.dropna()
if df.empty:
    raise ValueError("数据为空，请检查CSV文件！")

# 2.3: 按f排序
df = df.sort_values(by='X')

# 2.5: 转换为数组
x = df['X'].values

# 步骤3: 创建figure和ax
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# 步骤4: 调用核心绘图（绘制形状）
ax = draw_shapes(
    ax=ax,
    x=x,
    y=df['Y1'].values,
    plot_type='line',
    color='blue',  # 单值颜色
    # marker='*',  # 星形标记
    # linestyle=':',  # 点线
    alpha=1.0,
    label='Normalized Data',  # 标签准备
    hide_default_ticks=False,  # 默认不隐藏；改True可得裸形状
    hide_default_ticklabels=False  # 默认不隐藏；改True可得裸形状
)

# 示例：如果需要多曲线，多次调用（如添加原Y）
ax = draw_shapes(ax, x, df['Y2'].values, plot_type='line', color='red', label='Original Y')
ax = draw_shapes(ax, x, df['Y3'].values, plot_type='line', color='gray', label='Original Y')

# 步骤5: 可选调用标注
ax = add_annotations(
    ax=ax,
    # title='Frequency vs Normalized Y',
    xlabel='X',
    ylabel=r'Y',
    show_legend=False,
    add_grid=False,
)

plt.yscale('log')

# 步骤6: 最终处理
plt.tight_layout()
save_path = '../simplified_pipeline_output.svg'
plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
plt.show()
plt.close(fig)

print(f"绘图完成！图像已保存为 {save_path}")
