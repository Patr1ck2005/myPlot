# interactive_preview.py
# -*- coding: utf-8 -*-
from typing import Dict, Tuple, List, Optional, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox

# 引用你已实现的核心函数
from .spectrum_fit_core import (
    compute_curve_physics_core,
    fit_curve_physics_core,
)
from plot_3D.projects.SE.scripts.physics_core import SystemParams

def interactive_preview(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sp_template: SystemParams,
    z_range: Tuple[float, float],
    mode: Literal["Prad", "Ptot"] = "Prad",

    # 参数配置（需要可视化/可拟合的参数名、初值和边界）
    param_names: List[str],
    p0: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    fixed: Optional[Dict[str, float]] = None,

    # 计算/拟合选项
    fast: bool = True,
    z_samples: int = 1025,
    method: Literal["trapz", "quad"] = "trapz",
    normalize_by_max: bool = True,
    fit_range: Optional[Tuple[float, float]] = None,

    # —— 新增：控制模型曲线的平滑程度（预览与拟合输出都会使用） ——
    output_samples: Optional[int] = None,                   # 例如 2001
    output_range: Optional[Tuple[float, float]] = None,     # 例如 (x.min(), x.max())

    # 预览曲线外观
    data_marker_size: float = 3.0,
):
    """
    打开一个交互式窗口：
      - 左侧曲线：Data vs Model（模型可在更密的 x 网格上绘制，便于平滑）
      - 右侧面板：参数滑块、Normalize 勾选、Fit 按钮、Fit-range 文本框
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()

    # 预览/输出用的模型 x 网格（若不指定，则与数据一致）
    if output_samples is not None:
        if output_range is None:
            xmin_out, xmax_out = fit_range if fit_range is not None else (float(np.min(x)), float(np.max(x)))
        else:
            xmin_out, xmax_out = float(output_range[0]), float(output_range[1])
            if xmin_out > xmax_out:
                xmin_out, xmax_out = xmax_out, xmin_out
        x_model_grid = np.linspace(xmin_out, xmax_out, int(output_samples))
    else:
        x_model_grid = x

    # 拟合区间初值
    if fit_range is None:
        fit_range = (float(np.min(x)), float(np.max(x)))
    fx_min, fx_max = fit_range

    # 归一化：只对显示/拟合使用，不改原始 y
    def norm(yv: np.ndarray) -> np.ndarray:
        if not normalize_by_max:
            return yv
        m = float(np.max(np.abs(yv))) if yv.size else 1.0
        if m <= 0: m = 1.0
        return yv / m

    # 当前参数字典（滑块状态）初始化
    cur_params: Dict[str, float] = {k: float(p0[k]) for k in param_names}
    if fixed:
        for k, v in fixed.items():
            if k not in cur_params:
                cur_params[k] = float(v)

    # 计算一次初始模型（在平滑网格上）
    y_model = compute_curve_physics_core(
        x_model_grid, cur_params,
        sp_template=sp_template,
        z_range=z_range,
        mode=mode,
        fast=fast,
        z_samples=z_samples,
        method=method,
    )

    # 准备数据/模型（按需归一化）
    y_show = norm(y)
    y_model_show = norm(y_model)

    # 掩膜函数（用于拟合区间）
    def mask_range(xarr: np.ndarray, fr: Tuple[float, float]):
        a, b = float(fr[0]), float(fr[1])
        return (xarr >= a) & (xarr <= b)

    # --- 布局（右侧面板略加宽，文字更小，数值不再溢出） ---
    plt.close('all')
    fig = plt.figure(figsize=(11.5, 6.4))  # 略加大宽度
    # 主图区域
    ax_main = fig.add_axes([0.07, 0.12, 0.60, 0.80])
    # 右侧控件面板
    panel_left = 0.70
    panel_width = 0.28
    ax_controls_top = 0.88
    line_height = 0.06

    # 绘制数据与模型（模型用平滑网格）
    (sc_data,) = ax_main.plot(x, y_show, 'o', ms=data_marker_size, alpha=0.8, label='Data')
    (ln_model,) = ax_main.plot(x_model_grid, y_model_show, '-', lw=2.0, label='Model')

    # 拟合范围阴影
    shade = ax_main.axvspan(fx_min, fx_max, color='gray', alpha=0.12, label='Fit range')

    # 基于拟合范围, 设置绘图范围
    x_margin = (fx_max - fx_min) * 0.05
    ax_main.set_xlim(fx_min - x_margin, fx_max + x_margin)

    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y (normalized)' if normalize_by_max else 'y')
    ax_main.set_title('Interactive Preview: Data vs Model')
    ax_main.legend(loc='best')

    # ---- Normalize 勾选 ----
    ax_chk = fig.add_axes([panel_left, ax_controls_top, panel_width, line_height])
    chk = CheckButtons(ax_chk, labels=['Normalize by max'], actives=[normalize_by_max])

    # ---- Fit range 文本框（格式：min,max）----
    ax_txt = fig.add_axes([panel_left, ax_controls_top - 1.0*line_height, panel_width, line_height])
    tb = TextBox(ax_txt, 'Fit range [min,max]: ', initial=f"{fx_min},{fx_max}")

    # ---- Fit 按钮 ----
    ax_fit = fig.add_axes([panel_left, ax_controls_top - 2.05*line_height, 0.12, line_height])
    btn_fit = Button(ax_fit, 'Fit')

    # ---- Reset 按钮（重置为 p0）----
    ax_rst = fig.add_axes([panel_left + 0.13, ax_controls_top - 2.05*line_height, 0.12, line_height])
    btn_rst = Button(ax_rst, 'Reset')

    # ---- 参数滑块 ----
    sliders: Dict[str, Slider] = {}
    base_y = ax_controls_top - 3.3 * line_height
    for i, name in enumerate(param_names):
        lb, ub = (-np.inf, np.inf)
        if bounds and (name in bounds):
            lb, ub = bounds[name]
        # 兜底：若无界，给一个宽松可视范围
        if not np.isfinite(lb) or not np.isfinite(ub):
            c = float(cur_params[name])
            span = 1.0 if abs(c) < 1e-12 else abs(c) * 2.0
            if not np.isfinite(lb): lb = c - span
            if not np.isfinite(ub): ub = c + span
            if lb == ub: ub = lb + 1.0

        ax_sl = fig.add_axes([panel_left, base_y - i*line_height, panel_width, 0.80*line_height])
        sliders[name] = Slider(
            ax_sl, label=name,
            valmin=float(lb), valmax=float(ub),
            valinit=float(cur_params[name]),
            valstep=None,
            valfmt="%.4g",                 # 数值短格式，避免溢出
        )
        # 缩小字体 & 将数值文本放到滑块轴内部靠右
        sliders[name].label.set_fontsize(9)
        sliders[name].valtext.set_fontsize(9)
        sliders[name].valtext.set_position((0.98, 0.5))  # 轴坐标
        sliders[name].valtext.set_ha('right')

    # --- 状态更新函数 ---
    def read_fit_range_from_box():
        nonlocal fx_min, fx_max
        try:
            text = tb.text.strip()
            parts = text.split(',')
            if len(parts) == 2:
                a = float(parts[0]); b = float(parts[1])
                if a > b: a, b = b, a
                fx_min, fx_max = a, b
        except Exception:
            pass

    def update_model_from_sliders(_=None):
        nonlocal shade, fx_min, fx_max, x_model_grid

        # 1) 更新参数 -> 计算模型（在平滑网格上）
        for n, sl in sliders.items():
            cur_params[n] = float(sl.val)

        # 若用户修改了拟合范围且指定了 output_range=None，则可选把模型网格跟随拟合范围
        # 这里默认不自动改变 x_model_grid；如需联动，把下面这段取消注释：
        # read_fit_range_from_box()
        # if output_samples is not None and output_range is None:
        #     x_model_grid = np.linspace(fx_min, fx_max, int(output_samples))

        ym = compute_curve_physics_core(
            x_model_grid, cur_params,
            sp_template=sp_template,
            z_range=z_range,
            mode=mode,
            fast=fast,
            z_samples=z_samples,
            method=method,
        )

        # 2) 归一化与曲线更新
        normalize = chk.get_status()[0]
        y_plot = y if not normalize else norm(y)
        ym_plot = ym if not normalize else norm(ym)

        sc_data.set_ydata(y_plot)
        ln_model.set_data(x_model_grid, ym_plot)

        # 3) 读取文本框中的拟合范围，并重建阴影
        read_fit_range_from_box()
        if shade is not None:
            try:
                shade.remove()
            except Exception:
                pass
        shade = ax_main.axvspan(fx_min, fx_max, color='gray', alpha=0.12)

        ax_main.set_ylabel('y (normalized)' if normalize else 'y')
        # 基于拟合范围, 设置绘图范围
        x_margin = (fx_max - fx_min) * 0.05
        ax_main.set_xlim(fx_min - x_margin, fx_max + x_margin)

        fig.canvas.draw_idle()

    # 滑块联动
    for sl in sliders.values():
        sl.on_changed(update_model_from_sliders)

    # 勾选框/文本框联动
    chk.on_clicked(update_model_from_sliders)
    tb.on_submit(lambda _: update_model_from_sliders())

    # Reset：回到 p0
    def on_reset(event):
        for n, sl in sliders.items():
            sl.reset()
        tb.set_val(f"{fit_range[0]},{fit_range[1]}")
        update_model_from_sliders()
    btn_rst.on_clicked(on_reset)

    # Fit：以当前滑块值为初值，调用拟合，回填到滑块并刷新
    def on_fit(event):
        read_fit_range_from_box()
        m = mask_range(x, (fx_min, fx_max))
        if not np.any(m):
            return
        normalize = chk.get_status()[0]
        p0_now = {n: float(sliders[n].val) for n in param_names}
        res = fit_curve_physics_core(
            x=x, y=y,
            sp_template=sp_template,
            z_range=z_range,
            mode=mode,
            param_names=param_names,
            p0=p0_now,
            bounds=bounds,
            fixed=fixed,
            fit_range=(fx_min, fx_max),
            normalize_by_max=normalize,
            fast=fast,
            z_samples=z_samples,
            method=method,
            # —— 把平滑设置传给拟合输出（这样 res.x_fit/res.y_fit 也是平滑网格）——
            output_samples=output_samples,
            output_range=output_range,
        )
        # 回填滑块
        for n in param_names:
            if n in res.params:
                sliders[n].set_val(float(res.params[n]))
        # 直接用 res 的平滑曲线刷新（更快看到拟合结果）
        ln_model.set_data(res.x_fit, norm(res.y_fit) if normalize else res.y_fit)
        # 基于拟合范围, 设置绘图范围
        x_margin = (fx_max - fx_min) * 0.05
        ax_main.set_xlim(fx_min - x_margin, fx_max + x_margin)
        fig.canvas.draw_idle()
    btn_fit.on_clicked(on_fit)

    # 初次刷新
    update_model_from_sliders()
    plt.show()

