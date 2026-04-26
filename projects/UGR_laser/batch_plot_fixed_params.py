from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data_postprocess.data_filter import advanced_filter_eigensolution
from core.data_postprocess.data_grouper import group_vectors_one_sided_hungarian
from core.process_multi_dim_params_space import (
    create_data_grid,
    group_solution,
    extract_adjacent_fields,
)
from core.prepare_plot import prepare_plot_data
from core.plot_workflow import PlotConfig
from core.plot_cls import OneDimFieldVisualizer
from core.utils import norm_freq, convert_complex


# =========================================================
# 基础工具
# =========================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_case_name(
    fixed_params: Dict[str, Any],
    keys_for_name: List[str],
) -> str:
    parts = []
    for k in keys_for_name:
        if k in fixed_params:
            safe_k = (
                k.replace(" ", "")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
            )
            v = fixed_params[k]
            if isinstance(v, float):
                v_str = f"{v:g}"
            else:
                v_str = str(v)
            parts.append(f"{safe_k}_{v_str}")

    return "case_" + "__".join(parts)


def summarize_batch_params(
    base_fixed_params: Dict[str, Any],
    fixed_params_list: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    """Infer scanned params by comparing each case against the base fixed params."""
    scanned: Dict[str, set] = {}

    for case in fixed_params_list:
        for k, v in case.items():
            if k not in base_fixed_params or base_fixed_params[k] != v:
                scanned.setdefault(k, set()).add(v)

    scanned_sorted: Dict[str, List[Any]] = {}
    for k, vals in scanned.items():
        try:
            scanned_sorted[k] = sorted(vals)
        except TypeError:
            scanned_sorted[k] = sorted(vals, key=lambda x: str(x))

    return scanned_sorted


def print_batch_params_summary(
    data_path: str,
    base_fixed_params: Dict[str, Any],
    fixed_params_list: List[Dict[str, Any]],
):
    scanned = summarize_batch_params(base_fixed_params, fixed_params_list)

    print("\n========== Batch Params ==========")
    print(f"[Source File: {data_path}]")

    print("[Fixed Params]")
    for k in sorted(base_fixed_params.keys()):
        print(f"- {k}: {base_fixed_params[k]}")

    print("[Scan Params]")
    if not scanned:
        print("- (none)")
    else:
        for k in sorted(scanned.keys()):
            values = scanned[k]
            preview = ", ".join(map(str, values[:10]))
            suffix = "" if len(values) <= 10 else f", ... (total {len(values)})"
            print(f"- {k}: [{preview}{suffix}]")

    print(f"[Cases] total={len(fixed_params_list)}")
    print("==================================\n")


def load_and_prepare_dataframe(data_path: str, period_nm: float) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep="\t").copy()

    df["特征频率 (THz)"] = (
        df["特征频率 (THz)"]
        .apply(convert_complex)
        .apply(norm_freq, period=period_nm * 1e-9 * 1e12)
    )
    df["频率 (Hz)"] = np.real(df["特征频率 (THz)"])
    return df


def build_grid(
    df_sample: pd.DataFrame,
    param_keys: List[str],
    z_keys: List[str],
):
    grid_coords, Z = create_data_grid(
        df_sample,
        param_keys,
        z_keys,
        deduplication=False
    )
    return grid_coords, Z


# =========================================================
# 单个 case 数据处理
# =========================================================

def run_single_case(
    grid_coords: Dict[str, np.ndarray],
    Z: np.ndarray,
    fixed_params: Dict[str, Any],
    filter_conditions: Dict[str, Dict[str, float]],
    z_keys: List[str],
    deltas: tuple,
    max_num: int,
    nan_cost_penalty: float,
    auto_split_streams: bool,
):
    new_coords, Z_filtered, min_lens = advanced_filter_eigensolution(
        grid_coords,
        Z,
        z_keys=z_keys,
        fixed_params=fixed_params,
        filter_conditions=filter_conditions,
    )

    value_weights = np.array([[1.0]])
    deriv_weights = np.array([[1.0]])

    Z_new = np.empty_like(Z_filtered, dtype=object)
    for i in range(Z_filtered.shape[0]):
        Z_new[i] = Z_filtered[i][0]

    Z_grouped, additional_Z_grouped = group_vectors_one_sided_hungarian(
        [Z_new],
        deltas,
        additional_data=Z_filtered,
        value_weights=value_weights,
        deriv_weights=deriv_weights,
        max_m=max_num,
        nan_cost_penalty=nan_cost_penalty,
        auto_split_streams=auto_split_streams,
    )

    Z_targets = []
    for freq_index in range(max_num):
        _, Z_target = group_solution(
            new_coords,
            Z_grouped,
            freq_index=freq_index,
        )
        Z_targets.append(Z_target)

    datasets = []
    for i, Z_target in enumerate(Z_targets):
        dataset = {
            "eigenfreq_real": Z_target.real,
            "eigenfreq_imag": Z_target.imag,
        }

        eigenfreq, qfactor, fake_factor, up_s3, u_factor = extract_adjacent_fields(
            additional_Z_grouped,
            z_keys=z_keys,
            band_index=i,
        )

        q_safe = np.where(np.real(qfactor) > 0, qfactor, np.nan)
        qlog = np.log10(q_safe)

        dataset["qlog"] = qlog.real.ravel()
        dataset["-u_factor"] = np.abs(u_factor.real.ravel())
        dataset["up_s3"] = up_s3.real.ravel()

        datasets.append(dataset)

    return new_coords, datasets


def prepare_case_data(
    new_coords: Dict[str, np.ndarray],
    datasets: List[Dict[str, np.ndarray]],
    case_output_dir: Path,
):
    data_dir = ensure_dir(case_output_dir / "data")
    data_path = prepare_plot_data(
        new_coords,
        data_class="Eigensolution",
        dataset_list=datasets,
        fixed_params={},
        save_dir=str(data_dir),
    )
    return data_path


# =========================================================
# 单个 case 绘图
# =========================================================

def save_current_plotter_fig(plotter: OneDimFieldVisualizer, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(plotter, "ax") and plotter.ax is not None:
        fig = plotter.ax.figure
        fig.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        return

    if hasattr(plotter, "fig") and plotter.fig is not None:
        plotter.fig.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close(plotter.fig)
        return

    raise RuntimeError("无法从 plotter 中提取 matplotlib figure，请检查 plotter 的内部属性。")


def plot_three_figures(
    data_path: str,
    case_output_dir: Path,
    x_key: str,
    max_num: int,
    figsize=(2, 2),
):
    config_plot = PlotConfig(
        plot_params={"scale": 1},
        annotations={
            "xlabel": "",
            "ylabel": "",
            "show_axis_labels": True,
            "show_tick_labels": True,
        },
    )
    config_plot.update(figsize=figsize, tick_direction="in")

    plotter = OneDimFieldVisualizer(config=config_plot, data_path=data_path)
    plotter.load_data()

    svg_paths = []

    # 图1
    plotter.new_2d_fig()
    for i in range(max_num):
        plotter.plot(
            index=i,
            x_key=x_key,
            z1_key="-u_factor",
            z3_key="qlog",
            cmap="nipy_spectral",
        )
    plotter.adjust_view_2dim_auto()
    plotter.ax.set_yscale("log")
    plotter.add_annotations()
    p1 = case_output_dir / "fig_01_uq.svg"
    save_current_plotter_fig(plotter, p1)
    svg_paths.append(p1)

    # 图2
    plotter.re_initialized_plot()
    plotter.new_2d_fig()
    for i in range(max_num):
        plotter.plot(
            index=i,
            x_key=x_key,
            z1_key="eigenfreq_real",
            z2_key="eigenfreq_imag",
        )
    plotter.adjust_view_2dim_auto()
    plotter.add_annotations()
    p2 = case_output_dir / "fig_02_freq.svg"
    save_current_plotter_fig(plotter, p2)
    svg_paths.append(p2)

    # 图3
    plotter.re_initialized_plot()
    plotter.new_2d_fig()
    for i in range(max_num):
        plotter.plot(
            index=i,
            x_key=x_key,
            z1_key="qlog",
            z2_key="qlog",
        )
    plotter.adjust_view_2dim_auto()
    plotter.add_annotations()
    p3 = case_output_dir / "fig_03_qlog.svg"
    save_current_plotter_fig(plotter, p3)
    svg_paths.append(p3)

    return svg_paths


# =========================================================
# 拼接总图
# =========================================================

def combine_svgs_to_grid(
    svg_rows,
    output_svg,
    cell_w=140,
    cell_h=140,
    n_cols=3,
):
    from svgutils.compose import Figure, SVG

    n_rows = len(svg_rows)
    total_w = cell_w * n_cols
    total_h = cell_h * n_rows

    elements = []

    for r, row in enumerate(svg_rows):
        y0 = r * cell_h

        for c, svg_path in enumerate(row):
            x0 = c * cell_w

            # 关键：先把 Unicode minus 替换成普通减号
            text = Path(svg_path).read_text(encoding="utf-8", errors="replace")
            text = text.replace("−", "-")   # U+2212
            Path(svg_path).write_text(text, encoding="utf-8")

            elements.append(
                SVG(str(svg_path)).move(x0, y0).scale(1.0)
            )

    fig = Figure(f"{total_w}px", f"{total_h}px", *elements)
    fig.save(str(output_svg))



# =========================================================
# 批处理主逻辑
# =========================================================

def batch_run_cases(
    *,
    df_sample: pd.DataFrame,
    # data_path: str,
    output_root: str,
    # period_nm: float,
    x_key: str,
    param_keys: List[str],
    z_keys: List[str],
    fixed_params_list: List[Dict[str, Any]],
    filter_conditions: Dict[str, Dict[str, float]],
    deltas: tuple,
    max_num: int,
    nan_cost_penalty: float,
    auto_split_streams: bool,
    keys_for_case_name: List[str],
    figsize=(2, 2),
):
    output_root = ensure_dir(output_root)

    # df_sample = load_and_prepare_dataframe(data_path, period_nm)
    grid_coords, Z = build_grid(df_sample, param_keys, z_keys)

    print("网格参数：")
    for key, arr in grid_coords.items():
        print(f"  {key}: {arr}")
    print("数据网格 Z 的形状：", Z.shape)


    all_svg_rows = []

    for idx, fixed_params in enumerate(fixed_params_list, start=1):
        case_name = make_case_name(fixed_params, keys_for_name=keys_for_case_name)
        case_output_dir = ensure_dir(output_root / case_name)

        print(f"[{idx}/{len(fixed_params_list)}] Running {case_name}")

        new_coords, datasets = run_single_case(
            grid_coords=grid_coords,
            Z=Z,
            fixed_params=fixed_params,
            filter_conditions=filter_conditions,
            z_keys=z_keys,
            deltas=deltas,
            max_num=max_num,
            nan_cost_penalty=nan_cost_penalty,
            auto_split_streams=auto_split_streams,
        )

        data_path_case = prepare_case_data(
            new_coords=new_coords,
            datasets=datasets,
            case_output_dir=case_output_dir,
        )

        svg_paths = plot_three_figures(
            data_path=data_path_case,
            case_output_dir=case_output_dir,
            x_key=x_key,
            max_num=max_num,
            figsize=figsize,
        )

        all_svg_rows.append(svg_paths)

    summary_svg = Path(output_root) / "summary.svg"
    combine_svgs_to_grid(all_svg_rows, summary_svg)
    print(f"Summary SVG saved to: {summary_svg}")



# =========================================================
# 主入口：全部配置集中在这里
# =========================================================

if __name__ == "__main__":
    # -------------------------
    # 1. 文件与输出
    # -------------------------
    # DATA_PATH = "data/PhC-Rod-I-detailed-t_slab_factor_space-vary_fill-t_tot-norm_mesh-(tri0.10).csv"
    # DATA_PATH = "data/PhC-Rod-I-t_slab_factor_space-vary_fill-t_tot-norm_mesh-supp1(tri0.05).csv"
    # DATA_PATH = "data/PhC-Rod-I-t_slab_factor_space-vary_fill-t_tot-norm_mesh-supp2(tri0.15).csv"
    # DATA_PATH = "data/PhC-Tri-I-detailed-t_slab_space-vary_fill-t_tot-norm_mesh-(tri0.10).csv"
    # DATA_PATH = "data/PhC-Tri-I-t_slab_space-vary_fill-t_tot-norm_mesh-supp1(tri0.05).csv"
    # DATA_PATH = "data/PhC-Trap-I-t_slab_space-vary_fill-t_tot-norm_mesh-1.csv"
    # DATA_PATH = "data/PhC-Tri_Rod-I-t_slab_space-vary_fill-t_tot-norm_mesh-detailed_UGR_C-(tri0.05).csv"
    # DATA_PATH = "data/PhC-Tri_Rod-I-t_slab_space-vary_fill-t_tot-norm_mesh-detailed_UGR_C-(tri0.05)-supp1.csv"
    # DATA_PATH = "data/PhC-Tri_Void-I-t_slab_space-vary_fill-t_tot-norm_mesh-detailed_UGR_B-(tri0.10).csv"
    # DATA_PATH = "data/PhC-Tri_Void-I-t_slab_space-vary_fill-t_tot-norm_mesh-(tri0.10)-supp1.csv"
    # DATA_PATH = "data/PhC-Tri_Rod-I-t_slab_space-vary_fill-t_tot-t_cladding-norm_mesh-detailed_UGR_C-(tri0.03)-supp3.csv"
    # DATA_PATH = "data/PhC-Tri_Void-I-t_slab_space-vary_fill-t_tot-norm_mesh-detailed_UGR_D-(tri0.10).csv"
    # DATA_PATH = "data/PhC-Tri_Void-I-t_slab_space-vary_fill-t_tot-norm_mesh-detailed_UGR_D-(tri0.10)-supp1.csv"
    # DATA_PATH = "data/PhC-Tri_Rod-I-t_slab_space-vary_fill-t_tot(330,370nm)-norm_mesh-(tri0.10)-supp.csv"
    # DATA_PATH = "data/PhC-Tri_Rod-I-search0.50-t_slab_space-vary_fill-t_tot-norm_mesh-detailed_UGR_E-(tri0.05).csv"
    # DATA_PATH = "data/PhC-Tri_Rod-I-search0.50-t_slab_space-vary_fill-t_tot-norm_mesh-detailed_UGR_E-(tri0.05)-supp1.csv"
    # DATA_PATH = "data/PhC-Tri_Rod-I-search0.60-t_slab_space-vary_fill-t_tot-norm_mesh-detailed_UGR_E-(tri0.10)-supp.csv"
    # DATA_PATH = "data/PhC-Tri_Rod-I-search0.55-t_slab_space-vary_fill-t_tot-tri_factor-norm_mesh-detailed_UGR_E-(400t).csv"
    DATA_PATH = "data/PhC-Tri_Rod-I-search0.55-t_slab_space-vary_fill-t_tot-tri_factor-ultrahigh_mesh-detailed_UGR_E-(400t).csv"
    OUTPUT_ROOT = "./BATCH_OUTPUT"

    # -------------------------
    # 2. 基础参数
    # -------------------------
    PERIOD_NM = 500
    X_KEY = "t_slab_factor"
    PARAM_KEYS = ["m1", "m2", "t_slab_factor", "t_tot (nm)", "fill", "tri_factor", "substrate (nm)", "dpml (nm)"]
    # PARAM_KEYS = ["m1", "m2", "t_slab (nm)", "t_tot (nm)", "fill", "trap_asym", "substrate (nm)"]
    Z_KEYS = ["特征频率 (THz)", "品质因子 (1)", "fake_factor (1)", "up_S3 (1)", "U_factor (1)"]
    df_sample = load_and_prepare_dataframe(DATA_PATH, PERIOD_NM)

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
        "m1": 0.00,
        "m2": 0.00,
        "t_tot (nm)": 400,
        "tri_factor": 0.02,
        "substrate (nm)": 3000,  # 1500, 3000, 6000
        "dpml (nm)": 900,  # 300, 600, 900
    }

    # -------------------------
    # 5. 扫描参数：这里改成 fill 列表
    # -------------------------
    # FILL_LIST = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    # FILL_LIST = [0.56, 0.58, 0.59, 0.6,  0.61, 0.62, 0.64,]
    # FILL_LIST = [0.65,0.70,0.72,0.75,0.78,0.80,0.85]
    # FILL_LIST = [0.59,0.595,0.60]
    # FILL_LIST = [0.50, 0.55, 0.60, 0.65, 0.70]
    # FILL_LIST = [0.50,0.53,0.55,0.58,0.60,0.63,0.65,0.68,0.70]
    # FILL_LIST = [0.671, 0.673, 0.674, 0.675, 0.676, 0.677, 0.678, 0.679, 0.680, 0.681, 0.682,0.684]
    FILL_LIST = [0.675, 0.676, 0.677, 0.678, 0.679, 0.68,  0.681, 0.682, 0.683]

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
        df_sample=df_sample,
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

