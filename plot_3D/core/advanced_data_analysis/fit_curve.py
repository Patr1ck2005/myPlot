# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# =========================
# 模型：gamma 用 FWHM 表示
# =========================
def lorentzian(x, A, x0, gamma, y0):
    """Lorentzian with gamma as FWHM."""
    g2 = (0.5 * gamma) ** 2
    return y0 + A * g2 / ((x - x0) ** 2 + g2)

def fano(x, A, x0, gamma, q, y0):
    """Fano lineshape with gamma as FWHM. eps = 2*(x-x0)/gamma."""
    eps = 2.0 * (x - x0) / gamma
    return y0 + A * ((q + eps) ** 2) / (1.0 + eps ** 2)


# =========================
# 结果结构
# =========================
@dataclass
class FitResult:
    model: Literal["lorentzian", "fano"]
    x_fit: np.ndarray           # 按 x 排序后的点（与输入 x 等长）
    y_fit: np.ndarray           # 拟合曲线 y(model, x_fit, *p)
    params: Dict[str, float]    # 拟合参数
    perr: Dict[str, float]      # 参数标准差（来自协方差对角线）
    pcov: np.ndarray            # 协方差矩阵
    rss: float                  # 残差平方和
    aic: float                  # AIC
    bic: float                  # BIC


# =========================
# 初值与预处理
# =========================
def _robust_initial_guess(x: np.ndarray, y: np.ndarray, model: str):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # 背景：两端中位数
    edge_n = max(3, len(y_sorted) // 20)
    y0 = float(np.median(np.r_[y_sorted[:edge_n], y_sorted[-edge_n:]]))

    # 峰位与幅度
    i_max = int(np.argmax(y_sorted))
    x0 = float(x_sorted[i_max])
    A_guess = float(np.max(y_sorted) - y0)

    # FWHM 粗估
    half = y0 + 0.5 * A_guess
    above = np.where(y_sorted >= half)[0]
    if len(above) >= 2:
        gamma = float(x_sorted[above[-1]] - x_sorted[above[0]])
        if gamma <= 0:
            gamma = float((np.max(x_sorted) - np.min(x_sorted)) / 10.0)
    else:
        gamma = float((np.max(x_sorted) - np.min(x_sorted)) / 10.0)

    if model == "lorentzian":
        p0 = [A_guess if A_guess != 0 else 1.0, x0, max(gamma, 1e-6), y0]
        bounds = ([-np.inf, np.min(x_sorted), 1e-9, -np.inf],
                  [ np.inf, np.max(x_sorted),  np.inf,  np.inf])
        names = ["A", "x0", "gamma", "y0"]
    else:  # fano
        q_guess = 1.0 * np.sign(A_guess if A_guess != 0 else 1.0)
        p0 = [A_guess if A_guess != 0 else 1.0, x0, max(gamma, 1e-6), q_guess, y0]
        bounds = ([-np.inf, np.min(x_sorted), 1e-9, -np.inf, -np.inf],
                  [ np.inf, np.max(x_sorted),  np.inf,  np.inf,  np.inf])
        names = ["A", "x0", "gamma", "q", "y0"]

    return x_sorted, y_sorted, p0, bounds, names


def _finalize_result(model: str, names, popt, pcov, x_sorted, y_sorted, fun) -> FitResult:
    y_fit = fun(x_sorted, *popt)
    resid = y_sorted - y_fit
    rss = float(np.sum(resid ** 2))
    n = len(y_sorted)
    k = len(popt)
    # 信息准则（高斯残差假设）
    rss_safe = rss if rss > 0 else 1e-24
    aic = n * np.log(rss_safe / n) + 2 * k
    bic = n * np.log(rss_safe / n) + k * np.log(n)
    perr_vals = np.sqrt(np.maximum(np.diag(pcov), 0.0)) if pcov is not None else np.full(k, np.nan)
    params = {k_: float(v) for k_, v in zip(names, popt)}
    perr = {k_: float(e) for k_, e in zip(names, perr_vals)}
    return FitResult(
        model=model,
        x_fit=x_sorted,
        y_fit=y_fit,
        params=params,
        perr=perr,
        pcov=pcov,
        rss=rss,
        aic=float(aic),
        bic=float(bic),
    )


# =========================
# 核心拟合 API
# =========================
def fit_lineshape(
    x: np.ndarray,
    y: np.ndarray,
    model: Literal["lorentzian", "fano"] = "lorentzian",
    sigma: Optional[np.ndarray] = None,
    absolute_sigma: bool = False,
    maxfev: int = 100000,
    fit_range: Optional[Tuple[float, float]] = None,   # <--- 新增
) -> FitResult:
    """
    对给定 (x, y) 进行 Lorentzian 或 Fano 拟合。
    fit_range : (xmin, xmax)，仅在该区间内取点进行拟合。
    """
    # -------------------------
    # 筛选拟合范围
    # -------------------------
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if fit_range is not None:
        xmin, xmax = fit_range
        mask = (x >= xmin) & (x <= xmax)
        if not np.any(mask):
            raise ValueError("No data points found within fit_range")
        x, y = x[mask], y[mask]
        if sigma is not None:
            sigma = np.asarray(sigma).ravel()[mask]

    # -------------------------
    # 原有拟合流程
    # -------------------------
    model = model.lower()
    assert model in {"lorentzian", "fano"}
    x_sorted, y_sorted, p0, bounds, names = _robust_initial_guess(x, y, model)
    fun = lorentzian if model == "lorentzian" else fano

    popt, pcov = curve_fit(
        fun,
        x_sorted,
        y_sorted,
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        maxfev=maxfev,
    )
    return _finalize_result(model, names, popt, pcov, x_sorted, y_sorted, fun)


def fit_both_and_compare(
    x: np.ndarray,
    y: np.ndarray,
    criterion: Literal["rss", "aic", "bic"] = "aic",
    fit_range: Optional[Tuple[float, float]] = None,   # <--- 新增
) -> Tuple[FitResult, FitResult, FitResult]:
    """同时拟合 Lorentzian 与 Fano，在 fit_range 内比较两者。"""
    res_l = fit_lineshape(x, y, model="lorentzian", fit_range=fit_range)
    res_f = fit_lineshape(x, y, model="fano", fit_range=fit_range)

    def score(res: FitResult):
        if criterion == "rss":
            return res.rss
        elif criterion == "bic":
            return res.bic
        else:
            return res.aic

    best = res_l if score(res_l) <= score(res_f) else res_f
    return res_l, res_f, best


# =========================
# 绘图对比
# =========================
def plot_comparison(x: np.ndarray, y: np.ndarray, res_l: FitResult, res_f: FitResult, best: FitResult = None):
    """
    在同一张图中对比原始数据与两种拟合曲线。
    - 英文坐标/图例，便于发表与交流
    """
    plt.figure(figsize=(7.2, 4.8))
    # 原始数据
    plt.scatter(x, y, s=16, alpha=0.8, label="Data")
    # 拟合曲线（已按 x 排序）
    plt.plot(res_l.x_fit, res_l.y_fit, lw=2, label="Lorentzian fit")
    plt.plot(res_f.x_fit, res_f.y_fit, lw=2, linestyle="--", label="Fano fit")

    # 标注最佳模型
    title = "Lineshape Fit Comparison"
    if best is not None:
        title += f"  (best: {best.model})"

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# 示例：直接运行检验
# =========================
if __name__ == "__main__":
    # 生成一组带噪声的 Fano 数据来测试
    rng = np.random.default_rng(42)
    x = np.linspace(-5, 5, 301)
    true_fano = {"A": 3.0, "x0": 0.7, "gamma": 1.2, "q": 1.0, "y0": 0.2}
    y = fano(x, **true_fano) + 0.05 * rng.standard_normal(x.size)

    # 同时拟合两种模型并比较
    res_l, res_f, best = fit_both_and_compare(x, y, criterion="aic")

    # 打印参数
    print("[Lorentzian] params:", res_l.params, "perr:", res_l.perr, "AIC:", res_l.aic, "BIC:", res_l.bic)
    print("[Fano]       params:", res_f.params, "perr:", res_f.perr, "AIC:", res_f.aic, "BIC:", res_f.bic)
    print("Best model:", best.model)

    # 绘图对比
    plot_comparison(x, y, res_l, res_f, best)
