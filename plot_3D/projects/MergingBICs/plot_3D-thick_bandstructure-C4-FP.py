from plot_3D.core.data_postprocess.data_filter import advanced_filter_eigensolution
from plot_3D.core.data_postprocess.data_grouper import *
from plot_3D.core.plot_3D_params_space_plt import *
from plot_3D.core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from plot_3D.core.prepare_plot import prepare_plot_data
from plot_3D.core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

# 写一个简单的函数，将一个坐标为kx, ky的Z矩形数组按照对称性进行补充
# 默认kx, ky均为从零到正的均匀分布
# 提供多种模式: 例如 'xy_mirror'表示对x和y均进行镜像补充, 'C4_rot'表示C4旋转补充
def symmetric_complete_coords(new_coords, Z, mode='C4_rot'):
    """
    将仅定义在第一象限 (x>=0, y>=0) 的矩形网格 Z 按指定对称性补全到全平面。
    假设 x、y 等距且从 0 开始递增。返回 (coords_out, Z_out)。

    支持模式:
      - 'xy_mirror': 对 x,y 同时镜像补全得到四象限
      - 'x_mirror' : 仅对 x 镜像（得到左右对称）
      - 'y_mirror' : 仅对 y 镜像（得到上下对称）
      - 'C4_rot'   : 在 'xy_mirror' 的基础上，进一步与其中心 90° 旋转的结果做平均，
                     近似满足 C4 旋转对称（要求 x,y 网格尺寸相同且步长一致；否则退化为 'xy_mirror'）

    兼容性说明：
      - 返回的坐标严格递增，形如 [-...,-dx,0,dx,...]，不会重复 0 点；
      - 返回的 Z_out 满足 Z_out.shape == (len(x_full), len(y_full))，可直接配合
        M1,M2 = np.meshgrid(x_full, y_full, indexing='ij') 使用。
    """
    import numpy as np

    # ---- 1) 取出 x,y 轴名称与数组 ----
    # 优先使用 'kx','ky'，否则使用字典顺序的前两个键
    keys = list(new_coords.keys())
    if 'kx' in new_coords and 'ky' in new_coords:
        x_key, y_key = 'kx', 'ky'
    else:
        if len(keys) < 2:
            raise ValueError("new_coords 至少需要包含两个一维坐标轴。")
        x_key, y_key = keys[0], keys[1]

    x = np.asarray(new_coords[x_key])
    y = np.asarray(new_coords[y_key])
    Z = np.asarray(Z)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x,y 必须为一维数组。")
    if Z.shape[0] != x.size or Z.shape[1] != y.size:
        raise ValueError(f"Z 的前两维必须分别等于 len({x_key}), len({y_key})。"
                         f" 当前 Z.shape={Z.shape}, |x|={x.size}, |y|={y.size}")

    # 粗略校验单调性与起点
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("假定 x,y 为严格递增的均匀网格。")
    if not (np.isclose(x[0], 0) and np.isclose(y[0], 0)):
        raise ValueError("假定 x[0]=0 且 y[0]=0。")

    # ---- 2) 工具: 镜像坐标与数据（不重复 0）----
    def mirror_axis(arr):
        # [-...,-dx, 0, dx,...]；当起点即为 0 时，左侧不包含重复 0
        left = -arr[::-1]
        left = left[1:] if np.isclose(arr[0], 0) else left
        return np.concatenate([left, arr], axis=0)

    def mirror_data_xy(Z_in):
        # 先沿 x 镜像拼接（行方向，axis=0）
        left_block = np.flip(Z_in[1:, ...], axis=0)  # 去掉 0 行避免重复
        Zx = np.concatenate([left_block, Z_in], axis=0)
        # 再沿 y 镜像拼接（列方向，axis=1）
        bottom_block = np.flip(Zx[:, 1:, ...], axis=1)  # 去掉 0 列避免重复
        Zxy = np.concatenate([bottom_block, Zx], axis=1)
        return Zxy

    def mirror_data_x(Z_in):
        left_block = np.flip(Z_in[1:, ...], axis=0)
        return np.concatenate([left_block, Z_in], axis=0)

    def mirror_data_y(Z_in):
        bottom_block = np.flip(Z_in[:, 1:, ...], axis=1)
        return np.concatenate([bottom_block, Z_in], axis=1)

    # ---- 3) 按模式补全 ----
    if mode == 'xy_mirror':
        x_full = mirror_axis(x)
        y_full = mirror_axis(y)
        Z_full = mirror_data_xy(Z)

    elif mode == 'x_mirror':
        x_full = mirror_axis(x)
        y_full = y.copy()
        Z_full = mirror_data_x(Z)

    elif mode == 'y_mirror':
        x_full = x.copy()
        y_full = mirror_axis(y)
        Z_full = mirror_data_y(Z)

    elif mode == 'C4_rot':
        # 先做 xy 镜像补全得到完整网格
        x_full = mirror_axis(x)
        y_full = mirror_axis(y)
        Z_full = mirror_data_xy(Z)

        # 若网格为正方形（点数一致）则与其中心 90° 旋转版本做平均，近似满足 C4
        if x_full.size == y_full.size:
            # 以网格中心为原点的 90° 旋转对应到数组上的 np.rot90
            # 注意：这里假设 x,y 步长一致；否则仍然返回 xy_mirror 的结果（不报错）
            try:
                Zr1 = np.rot90(Z_full, k=1)
                Zr2 = np.rot90(Z_full, k=2)
                Zr3 = np.rot90(Z_full, k=3)
                # 用 nanmean 防止 dtype=object 时出错；若是 object，则退化为逐元素挑选第一个非 None
                if Z_full.dtype != object:
                    Z_full = np.nanmean(np.stack([Z_full, Zr1, Zr2, Zr3], axis=0), axis=0)
                else:
                    # object 情况：尽量保持原值；若有空位可用旋转值补上
                    Z_candidates = [Z_full, Zr1, Zr2, Zr3]
                    Z_obj = Z_full.copy()
                    it = np.nditer(np.empty(Z_obj.shape), flags=['multi_index'])
                    for _ in it:
                        i, j = it.multi_index
                        if Z_obj[i, j] is None:
                            for cand in Z_candidates:
                                if cand[i, j] is not None:
                                    Z_obj[i, j] = cand[i, j]
                                    break
                    Z_full = Z_obj
            except Exception:
                # 如果旋转失败（例如 dtype 不支持），就保留 xy_mirror 的结果
                pass
        # 如果不是正方形，直接沿用 xy_mirror 的结果

    else:
        raise ValueError(f"未知的对称模式: {mode}")

    # ---- 4) 组装返回的坐标字典（保持其余键不变）----
    coords_out = dict(new_coords)
    coords_out[x_key] = np.asarray(x_full)
    coords_out[y_key] = np.asarray(y_full)

    return coords_out, Z_full




if __name__ == '__main__':
    # data_path = 'data/FP_PhC-600nmFP-0.1k.csv'
    # data_path = 'data/FP_PhC-full_600_700nm.csv'
    data_path = 'data/FP_PhC-full_685nm.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    def norm_freq(freq, period):
        return freq/(c_const/period)
    period = 1300
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=period*1e-9*1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=period*1e-9)
    df_sample["phi (rad)"] = df_sample["phi (rad)"].apply(lambda x: x % np.pi)
    # # 筛选m1<0.1的成分
    # df_sample = df_sample[df_sample["m1"] < 0.05]
    # 指定用于构造网格的参数以及目标数据列
    param_keys = ["m1", "m2", "buffer (nm)"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "fake_factor (1)", "频率 (Hz)"]

    # 构造数据网格，此处不进行聚合，每个单元格保存列表
    grid_coords, Z = create_data_grid(df_sample, param_keys, z_keys, deduplication=False)
    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)

    # 假设已得到grid_coords, Z
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords, Z,
        z_keys=z_keys,
        fixed_params={
            'buffer (nm)': 600-15,
            # 'buffer (nm)': 600,
        },  # 固定
        filter_conditions={
            "fake_factor (1)": {"<": 1},  # 筛选
            # "频率 (Hz)": {">": 0.0, "<": 0.58},  # 筛选
        }
    )

    deltas3 = (1e-3, 1e-3)  # n个维度的网格间距
    # 当沿维度 d 生长时，值差权重矩阵（n×n）
    # 例如：value_weights[d, j] = 在 grow_dir=d 时，对维度 j 的值差权重
    value_weights = np.array([
        [1, 1], [1, 1]   # 沿维度生长时
    ])
    # 当沿维度 d 生长时，导数不连续权重矩阵（n×n）
    deriv_weights = np.array([
        [1, 1], [1, 1]
    ])
    # 创建一个新的数组，用于存储更新后的结果
    Z_new = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        for j in range(Z_filtered.shape[1]):
            Z_new[i, j] = Z_filtered[i][j][0]  # 提取每个 lst_ij 的第 b 列

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas3,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=5
    )

    # 假设你已经得到了 grid_coords, Z
    new_coords, Z_target1 = group_solution(
        new_coords, Z_grouped,
        freq_index=0  # 第n个频率
    )
    new_coords, Z_target2 = group_solution(
        new_coords, Z_grouped,
        freq_index=1  # 第n个频率
    )
    new_coords, Z_target3 = group_solution(
        new_coords, Z_grouped,
        freq_index=2  # 第n个频率
    )
    new_coords, Z_target4 = group_solution(
        new_coords, Z_grouped,
        freq_index=3  # 第n个频率
    )
    new_coords, Z_target5 = group_solution(
        new_coords, Z_grouped,
        freq_index=4  # 第n个频率
    )
    # new_coords, Z_target6 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=5  # 第n个频率
    # )
    # new_coords, Z_target7 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=6  # 第n个频率
    # )
    # new_coords, Z_target8 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=7  # 第n个频率
    # )
    # new_coords, Z_target9 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=8  # 第n个频率
    # )
    # new_coords, Z_target10 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=9  # 第n个频率
    # )
    # new_coords, Z_target11 = group_solution(
    #     new_coords, Z_grouped,
    #     freq_index=10  # 第n个频率
    # )


    print("去掉 bg_n 后的参数：")
    for k, v in new_coords.items():
        print(f"  {k}: {v}")

    # data_path = prepare_plot_data(
    #     new_coords, [
    #         Z_target1,
    #         Z_target2,
    #         Z_target3,
    #         Z_target4,
    #         Z_target5,
    #         Z_target6,
    #         Z_target7,
    #         Z_target8,
    #         Z_target9,
    #         # Z_target10,
    #         # Z_target11,
    #     ], x_key="m1", fixed_params={},
    #     save_dir='./rsl/eigensolution',
    # )
    #
    # from plot_3D.projects.MergingBICs.plot_thickband import main
    #
    # main(data_path)

    # 暂时简单使用3D绘制数据
    import matplotlib.pyplot as plt
    fs = 9
    plt.rcParams.update({'font.size': fs})

    fig = plt.figure(figsize=(3, 4))
    ax = fig.add_subplot(111, projection='3d')
    m1_vals = new_coords['m1']
    m2_vals = new_coords['m2']

    target2_phi = np.zeros(((additional_Z_grouped.shape[0]), (additional_Z_grouped.shape[1])))
    target2_tanchi = np.zeros(((additional_Z_grouped.shape[0]), (additional_Z_grouped.shape[1])))
    target2_Qfactor = np.zeros(((additional_Z_grouped.shape[0]), (additional_Z_grouped.shape[1])))
    # 把形状51,51的列表中的[1][3]数据提取出来
    for i in range(additional_Z_grouped.shape[0]):
        for j in range(additional_Z_grouped.shape[1]):
            target2_phi[i][j] = additional_Z_grouped[i][j][1][3]
            target2_tanchi[i][j] = additional_Z_grouped[i][j][1][2]
            target2_Qfactor[i][j] = additional_Z_grouped[i][j][1][1]

    M1, M2 = np.meshgrid(m1_vals, m2_vals, indexing='ij')

    # fig = plt.figure()
    # plt.imshow(target2_phi)
    # plt.colorbar()
    # plt.show()

    # 绘制多个 Z_target, 同时使用 Qfactor 作为颜色映射
    # Z_targets = [Z_target1, Z_target2, Z_target3]
    additional_Zs = [target2_phi,]
    Z_targets = [Z_target2,]

    from polar_postprocess import from_legacy_and_save

    pkl_path = 'rsl/eigensolution/polar_fields.pkl'
    from_legacy_and_save(
        pkl_path=pkl_path,
        m1=new_coords['m1'],
        m2=new_coords['m2'],
        Z_target_complex=Z_target2,  # 你的目标频带（复数也行，内部取 real 做等频线）
        phi_Q1=target2_phi,  # 第一象限 φ
        tanchi_Q1=target2_tanchi,  # 第一象限 tanchi
        Q_Q1=target2_Qfactor,  # 第一象限 Q（自己按数据生成一个同shape数组）
        do_complete=True,
    )

    for idx, Z_target in enumerate(Z_targets):
        FREQ = np.empty(M1.shape)
        Qfactor = np.empty(M1.shape)
        Phi = np.empty(M1.shape)
        tanchi = np.empty(M1.shape)
        for i in range(M1.shape[0]):
            for j in range(M1.shape[1]):
                val = Z_target[i, j]
                FREQ[i, j] = val.real
                # Qfactor[i, j] = np.log10(val.real/val.imag/2 if val.imag != 0 else 0)
                # Qfactor[i, j] = np.log10(additional_Z_grouped[i][j][1][1])
                Phi[i, j] = additional_Zs[idx][i, j]
                # tanchi[i, j] = additional_Z_grouped[i][j][1][2]
                # tanchi[i, j] = additional_Zs[idx][i][j]
        # surf_color_data = Qfactor
        surf_color_data = np.mod(Phi, np.pi)
        # surf_color_data = tanchi
        # surf_colors = plt.cm.RdBu((surf_color_data - -1) / 2)
        # surf_colors = plt.cm.hot((surf_color_data - np.min(surf_color_data)) / (np.max(surf_color_data) - np.min(surf_color_data)))
        # surf_colors = plt.cm.hot((surf_color_data - 2) / (6 - 2))
        # surf_colors = plt.cm.hsv((surf_color_data - np.min(surf_color_data)) / (np.max(surf_color_data) - np.min(surf_color_data)))
        surf_colors = plt.cm.twilight((surf_color_data - np.min(surf_color_data)) / (np.max(surf_color_data) - np.min(surf_color_data)))
        surf = ax.plot_surface(M1, M2, FREQ, facecolors=surf_colors, rstride=1, cstride=1, alpha=0.8, label=f'Band {idx+1}')
    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap='twilight')
    mappable.set_array(surf_color_data)
    cbar = plt.colorbar(mappable, ax=ax)
    ax.set_xlabel('m1')
    ax.set_ylabel('m2')
    ax.set_zlabel('Frequency (normalized)')

    # 调整视角
    ax.view_init(elev=30, azim=45-20)

    # 设置比例
    ax.set_box_aspect([1, 1, 1])  # 设置xyz轴的比例

    plt.savefig('temp.svg', transparent=True)
    plt.show()







