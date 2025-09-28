# spectrum_fit_core.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Literal
import numpy as np

# 依赖 SciPy 拟合（curve_fit）
try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# 复用你的 physics_core 计算核
from plot_3D.projects.SE.scripts.physics_core import SystemParams, NonHermitianTwoLevel

# ---------------------------
# 工具
# ---------------------------
def _normalize_by_max(y: np.ndarray) -> Tuple[np.ndarray, float]:
    m = float(np.max(np.abs(y))) if y.size else 1.0
    if m <= 0.0:
        m = 1.0
    return y / m, m

def _pack_theta(param_names: List[str], theta: np.ndarray,
                fixed: Optional[Dict[str, float]]) -> Dict[str, float]:
    p = {k: float(v) for k, v in zip(param_names, theta)}
    if fixed:
        for k, v in fixed.items():
            if k not in p:
                p[k] = float(v)
    return p

# ---------------------------
# 曲线计算：y(x) = ∫_z P(x,z; params) dz
# 内部调用 physics_core 的 power_on_grid_fast 实现矢量化
# ---------------------------
def compute_curve_physics_core(
    x_array: np.ndarray,
    params: Dict[str, float],
    *,
    sp_template: SystemParams,
    z_range: Tuple[float, float],
    mode: Literal["Prad", "Ptot"] = "Prad",
    fast: bool = True,
    z_samples: int = 1025,
    method: Literal["trapz", "quad"] = "trapz",
    # enforce_d2_zero: bool = True,
    d1: float = 1.0,
    d2: float = 0.0,
) -> np.ndarray:
    """
    使用 physics_core 计算 y(x)=∫ P(x,z) dz，其中：
    - x 对应 physics_core 的 omega
    - z 对应 physics_core 的 k
    - params 中至少可包含：omega0, delta, gamma0, gamma_rad（名称可自定，只要与 sp_template 字段对应即可）
    """
    # 组装 SystemParams（尽量中性，不限定物理量）
    sp = SystemParams(
        omega0=float(params.get("omega0", sp_template.omega0)),
        dispersion_v=float(params.get("dispersion_v", sp_template.dispersion_v)),
        delta=float(params.get("delta", sp_template.delta)),
        gamma0=float(params.get("gamma0", sp_template.gamma0)),
        d1=d1,
        d2=d2,
    )
    gamma_rad = float(params.get("gamma_rad", 0.5))

    model = NonHermitianTwoLevel(sp)

    x = np.asarray(x_array, float).ravel()
    zmin, zmax = float(z_range[0]), float(z_range[1])

    # 快路径：一次性生成 Z(x, z) 并沿 z 做 trapz
    if fast or method == "trapz":
        z = np.linspace(zmin, zmax, int(z_samples))
        # 注意：power_on_grid_fast 的形参顺序是 (mode, gamma_rad, k_vals, omega_vals)
        Z = model.power_on_grid_fast(mode, gamma_rad, z, x)  # shape = (n_x, n_z)
        return np.trapz(Z, z, axis=1).astype(float)

    # 回退（若你真想用逐点 quad，可自行扩展为对每个 x 积分）
    # 这里保持简洁统一：使用等距网格 + trapz
    z = np.linspace(zmin, zmax, int(z_samples))
    Z = model.power_on_grid_fast(mode, gamma_rad, z, x)
    return np.trapz(Z, z, axis=1).astype(float)

# ---------------------------
# 拟合结果
# ---------------------------
@dataclass
class FitResult:
    x_fit: np.ndarray          # 与输入 x 顺序一致
    y_fit: np.ndarray          # 拟合得到的曲线
    params: Dict[str, float]   # 最终参数（包含 fixed）
    pcov: np.ndarray           # 协方差矩阵（需要 SciPy）

def fit_curve_physics_core(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sp_template: SystemParams,
    z_range: Tuple[float, float],
    mode: Literal["Prad", "Ptot"] = "Prad",

    # 参数配置
    param_names: List[str],
    p0: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    fixed: Optional[Dict[str, float]] = None,

    # 拟合与计算选项
    fit_range: Optional[Tuple[float, float]] = None,
    normalize_by_max: bool = True,
    fast: bool = True,
    z_samples: int = 1025,
    method: Literal["trapz", "quad"] = "trapz",
    maxfev: int = 200000,

    # —— 新增：控制输出曲线的平滑程度 ——
    output_samples: Optional[int] = None,                    # 例如 2001
    output_range: Optional[Tuple[float, float]] = None,      # 例如 (x.min(), x.max())
) -> FitResult:
    """
    用 y(x) = ∫_z P(x,z; params) dz 去拟合 (x, y)。

    新增:
      - output_samples: 若提供，则在指定范围内生成新的等距 x 网格用于输出 y_fit；
                        否则在输入 x 上输出 y_fit（保持原行为）。
      - output_range:   与 output_samples 搭配使用；未提供则取 (x.min(), x.max())。
    """
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy is required for curve fitting (scipy.optimize.curve_fit).")

    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()

    # ---------- 选择拟合区间 ----------
    if fit_range is not None:
        xmin, xmax = float(fit_range[0]), float(fit_range[1])
        m = (x >= xmin) & (x <= xmax)
        if not np.any(m):
            raise ValueError("No points inside fit_range.")
        x_fit = x[m]
        y_target = y[m]
    else:
        x_fit = x
        y_target = y

    # ---------- 归一化目标（仅用于拟合） ----------
    if normalize_by_max:
        y_target, _ = _normalize_by_max(y_target)

    # ---------- 初值与边界 ----------
    theta0 = np.array([float(p0[name]) for name in param_names], dtype=float)
    if bounds:
        lb = np.array([float(bounds.get(n, (-np.inf, np.inf))[0]) for n in param_names], float)
        ub = np.array([float(bounds.get(n, (-np.inf, np.inf))[1]) for n in param_names], float)
    else:
        lb = np.full_like(theta0, -np.inf)
        ub = np.full_like(theta0,  np.inf)

    # ---------- 残差模型 ----------
    def model_eval(x_eval: np.ndarray, *theta_vec):
        params = _pack_theta(param_names, np.asarray(theta_vec, float), fixed)
        y_model = compute_curve_physics_core(
            x_eval, params,
            sp_template=sp_template,
            z_range=z_range,
            mode=mode,
            fast=fast,
            z_samples=z_samples,
            method=method,
        )
        if normalize_by_max:
            y_model, _ = _normalize_by_max(y_model)
        return y_model

    # ---------- 拟合 ----------
    popt, pcov = curve_fit(
        f=model_eval,
        xdata=x_fit,
        ydata=y_target,
        p0=theta0,
        bounds=(lb, ub),
        maxfev=maxfev,
    )

    # ---------- 生成输出网格并计算最终曲线 ----------
    if output_samples is not None:
        if output_range is None:
            xmin_out, xmax_out = fit_range if fit_range is not None else (float(np.min(x)), float(np.max(x)))
        else:
            xmin_out, xmax_out = float(output_range[0]), float(output_range[1])
            if xmin_out > xmax_out:
                xmin_out, xmax_out = xmax_out, xmin_out
        x_out = np.linspace(xmin_out, xmax_out, int(output_samples))
    else:
        x_out = x  # 保持与输入一致

    best_params = _pack_theta(param_names, popt, fixed)
    y_fit = compute_curve_physics_core(
        x_out, best_params,
        sp_template=sp_template,
        z_range=z_range,
        mode=mode,
        fast=fast,
        z_samples=z_samples,
        method=method,
    )
    if normalize_by_max:
        y_fit, _ = _normalize_by_max(y_fit)

    return FitResult(
        x_fit=x_out,
        y_fit=y_fit,
        params=best_params,
        pcov=pcov,
    )

