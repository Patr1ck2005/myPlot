from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import *
from core.plot_3D_params_space_plt import *
from core.plot_3D_params_space_pv import plot_Z_diff_pyvista
from core.prepare_plot import prepare_plot_data
from core.process_multi_dim_params_space import *

import numpy as np

c_const = 299792458

if __name__ == '__main__':
    # data_path = './data/eigen-Hex_annular-a400nm_r0.25_r_w80nm_t180nm-0.3kr.csv'
    # data_path = 'data/eigen-Hex_annular-a380nm_r88nm_r_w80nm_t180nm-0.3kr.csv'
    # data_path = 'data/eigen-Hex_annular-a400nm_r102nm_r_w80nm_t180nm-0.3kr.csv'
    # data_path = 'data/eigen-Hex_annular-a400nm_r105nm_r_w80nm_t180nm-0.3kr.csv'
    # data_path = 'data/eigen-Hex_annular-a400nm_r102nm_r_w80nm_t177.5nm-0.3kr.csv'
    data_path = 'data/eigen-Hex_annular-a400nm_r105nm_r_w75nm_t177.5nm-0.3kr.csv'
    df_sample = pd.read_csv(data_path, sep='\t')

    # 对 "特征频率 (THz)" 进行简单转换，假设仅取实部，后续也可以根据需要修改数据处理过程
    def convert_complex(freq_str):
        return complex(freq_str.replace('i', 'j'))
    def norm_freq(freq, period):
        return freq/(c_const/period)
    # period = 1000 nm
    df_sample["特征频率 (THz)"] = df_sample["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=400*1e-9*1e12)
    df_sample["频率 (Hz)"] = df_sample["频率 (Hz)"].apply(norm_freq, period=400*1e-9)
    df_sample["k"] = df_sample["m1"]-df_sample["m2"]

    # # 筛选m1<0.1的成分
    # df_sample = df_sample[df_sample["m1"] < 0.3]

    # 指定用于构造网格的参数以及目标数据列
    # param_keys = ["k_r", "k_azimu"]
    param_keys = ["k"]
    z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "S_air_prop (1)", "频率 (Hz)"]

    X_KEY = 'k'

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
        fixed_params={},  # 固定
        # fixed_params={"m1": 0, "m2": 0, "loss_k": 1e-3*0},  # 固定
        filter_conditions={
            "S_air_prop (1)": {"<": 1.0},  # 筛选
            # "m1": {"<": .1},  # 筛选
            # "频率 (Hz)": {">": 0.0, "<": 1.0},  # 筛选
        }
    )

    deltas = (1e-3,)  # n个维度的网格间距
    # 当沿维度 d 生长时，值差权重矩阵（n×n）
    # 例如：value_weights[d, j] = 在 grow_dir=d 时，对维度 j 的值差权重
    value_weights = np.array([
        [1,],   # 沿维度生长时
    ])
    # 当沿维度 d 生长时，导数不连续权重矩阵（n×n）
    deriv_weights = np.array([
        [1,],
    ])
    # 创建一个新的数组，用于存储更新后的结果
    Z_new = np.empty_like(Z_filtered, dtype=object)
    # 使用直接的循环来更新 Z_new
    for i in range(Z_filtered.shape[0]):
        Z_new[i] = Z_filtered[i][0]  # 提取每个 lst_ij 的第 b 列

    ###############################################################################################################
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 10))
    # 通过散点的方式绘制出来，看看效果
    for i in range(Z_new.shape[0]):
        z_vals = Z_new[i]
        for val in z_vals:
            if val is not None:
                plt.scatter(new_coords[X_KEY][i], np.real(val), color='blue', s=10)
    plt.xlabel(X_KEY)
    plt.ylabel('Re(eigenfreq) (THz)')
    plt.title('Filtered Eigenfrequencies before Grouping')
    plt.grid(True)
    plt.show()
    ###############################################################################################################

    MAX_NUM = 9
    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new], deltas,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=MAX_NUM,
        nan_cost_penalty=1e1,
        auto_split_streams=False
    )

    Z_targets = []
    # selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    selected_indices = range(MAX_NUM)
    for freq_index in selected_indices:
        new_coords, Z_target = group_solution(
            new_coords, Z_grouped,
            freq_index=freq_index  # 第n个频率
        )
        Z_targets.append(Z_target)

    ###################################################################################################################
    from core.plot_workflow import PlotConfig
    from core.prepare_plot import prepare_plot_data
    from core.process_multi_dim_params_space import extract_adjacent_fields

    datasets = []
    for i, Z_target in enumerate(Z_targets):
        dataset = {'eigenfreq_real': Z_target.real, 'eigenfreq_imag': Z_target.imag}
        # z_keys = ["特征频率 (THz)", "品质因子 (1)", "tanchi (1)", "phi (rad)", "S_air_prop (1)", "频率 (Hz)"]
        # eigenfreq, qfactor, fake_factor, up_s3, u_factor = extract_adjacent_fields(
        #     additional_Z_grouped,
        #     z_keys=z_keys,
        #     band_index=i
        # )
        datasets.append(dataset)

    data_path = prepare_plot_data(
        new_coords, data_class='Eigensolution', dataset_list=datasets, fixed_params={},
        save_dir='./rsl/1_para_space',
    )

    # ============================================================================================================
    from core.plot_workflow import PlotConfig
    from core.plot_cls import OneDimFieldVisualizer

    SELECTED_NUM = 9

    config = PlotConfig(
        plot_params={'scale': 1},
        annotations={
            'xlabel': '', 'ylabel': '',
            'show_axis_labels': True, 'show_tick_labels': True,
            'ylim': (0.38, 0.49)
        },
    )
    config.update(tick_direction='in')
    plotter = OneDimFieldVisualizer(config=config, data_path=data_path)
    plotter.load_data()

    plotter.re_initialized_plot()
    plotter.new_2d_fig(figsize=(2, 2))
    # for i in range(SELECTED_NUM):
    #     plotter.plot(
    #         index=i, x_key=X_KEY, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
    #         # enable_fill=True, default_color='gray', alpha_fill=0.3, scale=1
    #     )
    plotter.plot(
        index=0, x_key=X_KEY, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        # enable_fill=True, default_color='gray', alpha_fill=0.3, scale=1
    )
    plotter.scatter(
        index=0, x_key=X_KEY, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
        # enable_fill=True, default_color='gray', alpha_fill=0.3, scale=1
    )
    plotter.adjust_view_2dim_auto()
    plotter.add_annotations()
    plotter.save_and_show()

    plotter.re_initialized_plot()
    plotter.new_2d_fig(figsize=(2, 3))
    for i in range(SELECTED_NUM):
        plotter.plot(
            index=i, x_key=X_KEY, z1_key='eigenfreq_real', z2_key='eigenfreq_imag', z3_key='eigenfreq_imag',
            enable_fill=True, gradient_fill=True, gradient_direction='z3', cmap='magma', alpha_fill=1, scale=1,
            global_color_vmin=0, global_color_vmax=5e-3,
        )
    for i in range(SELECTED_NUM):
        plotter.plot(
            index=i, x_key=X_KEY, z1_key='eigenfreq_real', z2_key='eigenfreq_imag',
            enable_fill=False, default_line_color='gray', alpha_fill=0.3, scale=1
        )
    period = 400e-9  # C6
    c_const = 299792458
    f1 = c_const / 930e-9 / (c_const / period)
    f2 = c_const / 900e-9 / (c_const / period)
    plotter.ax.axhspan(f1, f2, color='gray', alpha=0.3, linewidth=0, zorder=-1)
    plotter.adjust_view_2dim_auto()
    plotter.add_annotations()
    plotter.save_and_show()

