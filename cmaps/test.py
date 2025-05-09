import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc  # 请确保已安装：pip install colorcet

# 1) 列出所有 Colorcet colormap 名称，打印出来手动挑选
all_cc_maps = list(cc.cm.keys())
print("Colorcet 提供的所有 colormap：")
print(sorted(all_cc_maps))

# 2) 在这里把手动挑好的循环 colormap 名称填入列表，比如：
cyclic_maps = [
    'rainbow',
    'hsv',
    'CET_C1',
    'CET_C2',
    'CET_C3',
    'CET_C4',
    'CET_C5',
    'CET_C6',
    'CET_C7',
    'CET_C8',
    'CET_C8',
    'CET_C9',
    'CET_C10',
    'CET_C11', 'circle_mgbm_67_c31', 'circle_mgbm_67_c31_r', 'circle_mgbm_67_c31_s25', 'circle_mgbm_67_c31_s25_r',
    'colorwheel', 'colorwheel_r', 'coolwarm', 'coolwarm_r', 'cwr', 'cwr_r', 'cyclic_bgrmb_35_70_c75',
    'cyclic_bgrmb_35_70_c75_r', 'cyclic_bgrmb_35_70_c75_s25', 'cyclic_bgrmb_35_70_c75_s25_r', 'cyclic_grey_15_85_c0',
    'cyclic_grey_15_85_c0_r', 'cyclic_grey_15_85_c0_s25', 'cyclic_grey_15_85_c0_s25_r', 'cyclic_isoluminant',
    'cyclic_isoluminant_r', 'cyclic_mrybm_35_75_c68', 'cyclic_mrybm_35_75_c68_r', 'cyclic_mrybm_35_75_c68_s25',
    'cyclic_mrybm_35_75_c68_s25_r', 'cyclic_mybm_20_100_c48', 'cyclic_mybm_20_100_c48_r', 'cyclic_mybm_20_100_c48_s25',
    'cyclic_mybm_20_100_c48_s25_r', 'cyclic_mygbm_30_95_c78', 'cyclic_mygbm_30_95_c78_r', 'cyclic_mygbm_30_95_c78_s25',
    'cyclic_mygbm_30_95_c78_s25_r', 'cyclic_mygbm_50_90_c46', 'cyclic_mygbm_50_90_c46_r', 'cyclic_mygbm_50_90_c46_s25',
    'cyclic_mygbm_50_90_c46_s25_r', 'cyclic_protanopic_deuteranopic_bwyk_16_96_c31',
    'cyclic_protanopic_deuteranopic_bwyk_16_96_c31_r', 'cyclic_protanopic_deuteranopic_wywb_55_96_c33',
    'cyclic_protanopic_deuteranopic_wywb_55_96_c33_r', 'cyclic_rygcbmr_50_90_c64', 'cyclic_rygcbmr_50_90_c64_r',
    'cyclic_rygcbmr_50_90_c64_s25', 'cyclic_rygcbmr_50_90_c64_s25_r', 'cyclic_tritanopic_cwrk_40_100_c20',
    'cyclic_tritanopic_cwrk_40_100_c20_r', 'cyclic_tritanopic_wrwc_70_100_c20', 'cyclic_tritanopic_wrwc_70_100_c20_r',
    'cyclic_wrkbw_10_90_c43', 'cyclic_wrkbw_10_90_c43_r', 'cyclic_wrkbw_10_90_c43_s25', 'cyclic_wrkbw_10_90_c43_s25_r',
    'cyclic_wrwbw_40_90_c42', 'cyclic_wrwbw_40_90_c42_r', 'cyclic_wrwbw_40_90_c42_s25', 'cyclic_wrwbw_40_90_c42_s25_r',
    'cyclic_ymcgy_60_90_c67', 'cyclic_ymcgy_60_90_c67_r', 'cyclic_ymcgy_60_90_c67_s25', 'cyclic_ymcgy_60_90_c67_s25_r',
    # ... 继续添加你感兴趣的名称 ...
]

# 如果你不确定有哪些是循环的，可以先只填入一两个，测试后继续补充

# 3) 构造涡旋相位场
n = 400
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
X, Y = np.meshgrid(x, y)
phase = np.arctan2(Y, X)
phase_norm = (phase + np.pi) / (2 * np.pi)

# 4w) 绘制
cols = int(np.ceil(np.sqrt(len(cyclic_maps))))
rows = (len(cyclic_maps) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = axes.flatten()

for ax, name in zip(axes, cyclic_maps):
    try:
        cmap = cc.cm[name]
    except KeyError:
        # 如果在 colorcet 中没有，退回 matplotlib
        cmap = name
    ax.imshow(phase_norm, origin='lower', extent=[-1, 1, -1, 1], cmap=cmap)
    ax.set_title(name, fontsize=10)
    ax.axis('off')

# 清除多余子图
for ax in axes[len(cyclic_maps):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
