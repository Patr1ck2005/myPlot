from __future__ import annotations

import numpy as np
import pandas as pd

from batch_plot_fixed_params import batch_run_cases, print_batch_params_summary
from core.utils import convert_complex, norm_freq

# =========================================================
# 主入口：全部配置集中在这里
# =========================================================

if __name__ == "__main__":
    # -------------------------
    # 1. 文件与输出
    # -------------------------
    DATA_PATH = "data/Tri_Rod-I-search0.55-t_slab_space-vary_fill-t_tot-k-norm_mesh-UGR_E_BIC-(400t).csv"
    OUTPUT_ROOT = "./BATCH_OUTPUT"

    # -------------------------
    # 2. 基础参数
    # -------------------------
    PERIOD_NM = 500
    X_KEY = "t_slab_factor"
    PARAM_KEYS = ["k_r", "k_azimu", "t_slab_factor", "t_tot (nm)", "fill", "tri_factor", "substrate (nm)", "dpml (nm)"]
    Z_KEYS = ["特征频率 (THz)", "品质因子 (1)", "fake_factor (1)", "up_S3 (1)", "U_factor (1)"]
    df = pd.read_csv(DATA_PATH, sep="\t").copy()
    df["特征频率 (THz)"] = (
        df["特征频率 (THz)"].apply(convert_complex).apply(norm_freq, period=PERIOD_NM * 1e-9 * 1e12)
    )
    df["频率 (Hz)"] = np.real(df["特征频率 (THz)"])

    # -------------------------
    # 3. 分组与绘图参数
    # -------------------------
    MAX_NUM = 20
    DELTAS = (1e-3,)
    NAN_COST_PENALTY = 1e1
    AUTO_SPLIT_STREAMS = False
    FIGSIZE = (2, 2)

    # -------------------------
    # 4. 固定参数
    # -------------------------
    BASE_FIXED_PARAMS = {
        "t_tot (nm)": 400,
        "k_azimu": 0,
        "k_r": 0.01,
        "tri_factor": 0.0,
        "substrate (nm)": 3000,  # 1500, 3000
        "dpml (nm)": 600,  # 300, 600
    }

    # -------------------------
    # 5. 扫描参数：这里改成 fill 列表
    # -------------------------
    FILL_LIST = [0.6,  0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,  0.75]

    FIXED_PARAMS_LIST = [
        {
            **BASE_FIXED_PARAMS,
            "fill": fill_value,
        }
        for fill_value in FILL_LIST
    ]

    # -------------------------
    # 6. 过滤条件：手动集中传入
    # -------------------------
    FILTER_CONDITIONS = {
        "fake_factor (1)": {"<": 1},
        # "品质因子 (1)": {"<": 1e6},
        "特征频率 (THz)": {"<": 0.67},
    }

    # -------------------------
    # 7. case 命名规则
    # -------------------------
    KEYS_FOR_CASE_NAME = ["fill"]

    # -------------------------
    # 8. 批量运行
    # -------------------------
    batch_run_cases(
        df_sample=df,
        output_root=OUTPUT_ROOT,
        x_key=X_KEY,
        param_keys=PARAM_KEYS,
        z_keys=Z_KEYS,
        fixed_params_list=FIXED_PARAMS_LIST,
        filter_conditions=FILTER_CONDITIONS,
        deltas=DELTAS,
        max_num=MAX_NUM,
        nan_cost_penalty=NAN_COST_PENALTY,
        auto_split_streams=AUTO_SPLIT_STREAMS,
        keys_for_case_name=KEYS_FOR_CASE_NAME,
        figsize=FIGSIZE,
    )

    print_batch_params_summary(
        data_path=DATA_PATH,
        base_fixed_params=BASE_FIXED_PARAMS,
        fixed_params_list=FIXED_PARAMS_LIST,
    )

