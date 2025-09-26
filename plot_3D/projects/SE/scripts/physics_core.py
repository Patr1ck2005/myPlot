# physics_core.py
# -*- coding: utf-8 -*-
"""
Core physics logic for a 2-level non-Hermitian model:
- Heff = H0 - i/2 * (GammaAbs + K K^\dagger)
- G(omega, k) = (omega I - Heff)^{-1}
- Prad(omega, k) = d^\dagger G^\dagger (K K^\dagger) G d
- Ptot(omega, k) = d^\dagger G^\dagger [ -Im(Heff)/2 ] G d
- Band structure from H0 or from Re(eigvals(Heff))

This module contains no plotting; it only exposes physics computations.

Dependencies:
- numpy
- (optional) scipy.integrate.quad for k-integration helpers
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import numpy as np

try:
    from scipy.integrate import quad
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ===================== Parameters =====================

@dataclass
class SystemParams:
    """系统参数（模型固定量）"""
    omega0: float = 0.0
    dispersion_v: float = 1
    delta:  float = 0.0
    gamma0: float = 1e-3     # 材料吸收损耗（对角 GammaAbs 的一半）
    d1:     float = 1.0      # 偶极子 x 分量
    d2:     float = 0.0      # 偶极子 y 分量


# ===================== Core Class =====================

class NonHermitianTwoLevel:
    """
    二能级非厄米模型的物理核心：
    - 负责生成 Heff、G
    - 计算 Prad / Ptot
    - 计算能带（H0 或 Heff 的本征值）
    - 提供 k-积分与(omega,k)网格采样的辅助函数（不绘图）
    """

    def __init__(self, params: SystemParams):
        self.sp = params

    # ---------- Hamiltonians & Green's Function ----------

    def heff(self, k: float, gamma_rad: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回 (Heff, KKT)：
        H0 = [[omega0 + delta, k],
              [k,             omega0 - delta]]
        GammaAbs = diag(2*gamma0, 2*gamma0)
        KKT = [[0, 0],
               [0, 2*gamma_rad]]
        Heff = H0 - i/2 * (GammaAbs + KKT)

        注：Heff 不显含 omega（在本模型设定下）。
        """
        sp = self.sp
        H0 = np.array([[sp.omega0 + sp.delta, sp.dispersion_v*k],
                       [sp.dispersion_v*k,                     sp.omega0 - sp.delta]], dtype=complex)
        GammaAbs = np.array([[2*sp.gamma0, 0],
                             [0,           2*sp.gamma0]], dtype=complex)
        KKT = np.array([[0,               0],
                        [0, 2*float(gamma_rad)]], dtype=complex)
        Heff = H0 - 0.5j * (GammaAbs + KKT)
        return Heff, KKT

    def g_matrix(self, omega: float, k: float, gamma_rad: float) -> np.ndarray:
        """
        格林函数 G = (omega I - Heff)^{-1}
        奇异时返回 NaN 矩阵，便于上层可视化/掩蔽。
        """
        Heff, _ = self.heff(k, gamma_rad)
        Ginv = omega * np.identity(2, dtype=complex) - Heff
        try:
            return np.linalg.inv(Ginv)
        except np.linalg.LinAlgError:
            return np.full((2, 2), np.nan, dtype=complex)

    # ---------- Power Quantities ----------

    def prad(self, omega: float, k: float, gamma_rad: float) -> float:
        """
        Prad(omega, k) = d^† G^† (KKT) G d
        返回实数（取实部）。
        """
        Heff, KKT = self.heff(k, gamma_rad)
        G = self.g_matrix(omega, k, gamma_rad)
        d = np.array([self.sp.d1, self.sp.d2], dtype=complex)
        val = np.conjugate(d).T @ np.conjugate(G).T @ KKT @ G @ d
        return float(np.real(val))

    def ptot(self, omega: float, k: float, gamma_rad: float) -> float:
        """
        Ptot(omega, k) = d^† G^† [ -2Im(Heff) ] G d
        """
        Heff, _ = self.heff(k, gamma_rad)
        G = self.g_matrix(omega, k, gamma_rad)
        d = np.array([self.sp.d1, self.sp.d2], dtype=complex)
        kernel = -2*np.imag(Heff)
        val = np.conjugate(d).T @ np.conjugate(G).T @ kernel @ G @ d
        return float(np.real(val))

    # ---------- Bands ----------

    def bands_h0(self, k_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        H0 的解析本征值：omega0 ± sqrt(delta^2 + k^2)
        返回 (band_plus, band_minus) ，均为实数数组。
        """
        sp = self.sp
        gap = np.sqrt(sp.delta**2 + k_vals**2)
        return sp.omega0 + gap, sp.omega0 - gap

    def bands_heff(self, k_vals: np.ndarray, gamma_rad: float,
                   sort_by_real: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Heff 的本征值（复数），常用于绘制 Re(eigvals(Heff)) 作为“能带”。
        - sort_by_real=True：按实部从小到大排序并拆成 (plus, minus)
        - sort_by_real=False：不排序，按返回顺序拆分（不推荐）
        返回 (band_plus, band_minus)，每个为复数数组。
        """
        plus_list, minus_list = [], []
        for kk in k_vals:
            Heff, _ = self.heff(kk, gamma_rad)
            vals = np.linalg.eigvals(Heff)
            if sort_by_real:
                # 按实部排序，确保带标号一致
                order = np.argsort(np.real(vals))
                vals = vals[order]
            # 定义 minus = 较小实部，plus = 较大实部
            minus_list.append(vals[0])
            plus_list.append(vals[1])
        return np.array(plus_list, dtype=complex), np.array(minus_list, dtype=complex)

    # ---------- Helpers: grids & integration (no plotting) ----------

    def power_on_grid(self,
                      power_fn: Callable[[float, float, float], float],
                      gamma_rad: float,
                      k_min: float, k_max: float, k_samples: int,
                      omega_min: float, omega_max: float, omega_samples: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 (omega, k) 网格上的功率密度值：
        返回 (k_grid, omega_grid, Z) ，其中 Z.shape = (omega_samples, k_samples)
        """
        k_grid = np.linspace(k_min, k_max, int(k_samples))
        omega_grid = np.linspace(omega_min, omega_max, int(omega_samples))
        Z = np.zeros((omega_grid.size, k_grid.size), dtype=float)
        for i, om in enumerate(omega_grid):
            for j, kk in enumerate(k_grid):
                Z[i, j] = power_fn(om, kk, gamma_rad)
        return k_grid, omega_grid, Z

    def integrate_over_k(self,
                         power_fn: Callable[[float, float, float], float],
                         gamma_rad: float,
                         k_range: Tuple[float, float],
                         omega_array: np.ndarray,
                         method: str = "quad") -> np.ndarray:
        """
        对每个 omega，将 power_fn(omega, k, gamma_rad) 在 k ∈ [kmin,kmax] 上积分。
        - method="quad": 使用 scipy.integrate.quad（若不可用将回退到离散法）
        - method="trapz": 使用 numpy.trapz 在均匀网格上离散积分

        返回 shape = (len(omega_array),) 的实数数组。
        """
        kmin, kmax = float(k_range[0]), float(k_range[1])
        results = []

        if method == "quad" and _HAS_SCIPY:
            for om in omega_array:
                val, _err = quad(lambda kk: power_fn(om, kk, gamma_rad), kmin, kmax)
                results.append(val)
            return np.asarray(results, dtype=float)

        # 回退或指定离散法：固定网格 + trapz
        # 自定义一个较细的 k 网格（可按需调整密度）
        k_grid = np.linspace(kmin, kmax, 2049)
        for om in omega_array:
            y = np.array([power_fn(om, kk, gamma_rad) for kk in k_grid], dtype=float)
            results.append(np.trapz(y, k_grid))
        return np.asarray(results, dtype=float)

    # ---------- Ultra-fast vectorized grid power (no Python loops) ----------

    def _ad_params(self, omega_array: np.ndarray, gamma_rad: float):
        """
        生成广播友好的 a(omega), d(omega), 以及常量：
        A = [[a, -v*k], [-v*k, d]], 其中
        a = (omega - (omega0+delta)) + i*gamma0
        d = (omega - (omega0-delta)) + i*(gamma0 + gamma_rad)
        """
        sp = self.sp
        omega = np.asarray(omega_array, dtype=float)  # shape: (n_omega,)
        a = (omega - (sp.omega0 + sp.delta)) + 1j * sp.gamma0
        d = (omega - (sp.omega0 - sp.delta)) + 1j * (sp.gamma0 + gamma_rad)
        return a, d

    def _solve_u_grid(self, omega_grid: np.ndarray, k_grid: np.ndarray, gamma_rad: float):
        """
        求解 u = G d 向量的网格（使用 2x2 显式逆公式），返回 (u1, u2) 复数组，形状皆为 (n_omega, n_k)
        注：u 是 G(omega,k) 作用在偶极向量 d=[d1,d2]^T 上的结果。
        """
        sp = self.sp
        d1, d2 = complex(sp.d1), complex(sp.d2)
        v = float(sp.dispersion_v)

        # a(omega), d(omega) 先计算出来，然后与 k 广播
        a_om, d_om = self._ad_params(omega_grid, gamma_rad)  # (n_omega,)
        k = np.asarray(k_grid, dtype=float)                   # (n_k,)

        # 广播到 (n_omega, n_k)
        a = a_om[:, None]                 # (n_omega, 1)
        d = d_om[:, None]                 # (n_omega, 1)
        vk = v * k[None, :]               # (1, n_k)

        # A = [[a, -vk], [-vk, d]],  A^{-1} = (1/Δ) [[d, vk], [vk, a]], Δ = a*d - (vk)^2
        Delta = a * d - (vk ** 2)        # (n_omega, n_k)

        # 防止极点处溢出 -> 在非常小的 |Δ| 处做一个极小正则
        eps = 1e-18
        mask = (np.abs(Delta) < eps)
        if np.any(mask):
            Delta = Delta + eps * mask

        u1 = (d * d1 + vk * d2) / Delta  # (n_omega, n_k)
        u2 = (vk * d1 + a * d2) / Delta  # (n_omega, n_k)
        return u1, u2

    def power_on_grid_fast(self,
                           mode: str,
                           gamma_rad: float,
                           k_vals: np.ndarray,
                           omega_vals: np.ndarray) -> np.ndarray:
        """
        向量化计算 Z(omega,k)，shape=(n_omega, n_k)
        - mode='Prad'：M = KKT = diag(0, 2*gamma_rad)
        - mode='Ptot'：M = GammaAbs + KKT = diag(2*gamma0, 2*gamma0 + 2*gamma_rad)
        公式：P = u^† M u = m1*|u1|^2 + m2*|u2|^2
        """
        m = mode.strip().lower()
        u1, u2 = self._solve_u_grid(np.asarray(omega_vals, float),
                                    np.asarray(k_vals, float),
                                    float(gamma_rad))
        if m == "prad":
            m1 = 0.0
            m2 = 2.0 * float(gamma_rad)
        elif m == "ptot":
            m1 = 2.0 * float(self.sp.gamma0)
            m2 = 2.0 * (float(self.sp.gamma0) + float(gamma_rad))
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'Ptot' or 'Prad'.")

        Z = (m1 * (u1.real**2 + u1.imag**2) +
             m2 * (u2.real**2 + u2.imag**2))  # 等价于 m1*|u1|^2 + m2*|u2|^2
        return Z.astype(float)

    def power_on_grid(self,
                      power_fn: Callable[[float, float, float], float],
                      gamma_rad: float,
                      k_min: float, k_max: float, k_samples: int,
                      omega_min: float, omega_max: float, omega_samples: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        兼容旧接口：如果 power_fn 来自 get_power_function(model, mode)，
        则自动走 vectorized 快路径；否则回退到逐点循环。
        """
        k_grid = np.linspace(k_min, k_max, int(k_samples))
        omega_grid = np.linspace(omega_min, omega_max, int(omega_samples))

        # 检查是否是标准模式函数
        fast_mode = None
        try:
            # 通过函数名粗略判断（也可在 get_power_function 中打标签）
            fn_name = getattr(power_fn, "__name__", "").lower()
            if "prad" in fn_name:
                fast_mode = "Prad"
            elif "ptot" in fn_name:
                fast_mode = "Ptot"
        except Exception:
            pass

        if fast_mode is not None:
            Z = self.power_on_grid_fast(fast_mode, gamma_rad, k_grid, omega_grid)
            return k_grid, omega_grid, Z

        # 回退：保留你原来的逐点法
        Z = np.zeros((omega_grid.size, k_grid.size), dtype=float)
        for i, om in enumerate(omega_grid):
            for j, kk in enumerate(k_grid):
                Z[i, j] = power_fn(om, kk, gamma_rad)
        return k_grid, omega_grid, Z



# ===================== Factory for callable modes =====================

def get_power_function(model: NonHermitianTwoLevel, mode: str) -> Callable[[float, float, float], float]:
    """
    根据模式字符串返回相应的功率计算可调用对象：
    - "Ptot" -> model.ptot
    - "Prad" -> model.prad
    """
    m = mode.strip().lower()
    if m == "ptot":
        return model.ptot
    if m == "prad":
        return model.prad
    raise ValueError(f"Unknown mode: {mode!r}. Use 'Ptot' or 'Prad'.")


# ===================== Minimal usage example =====================

if __name__ == "__main__":
    sp = SystemParams(omega0=0.0, delta=0.0, gamma0=1e-3, d1=1.0, d2=0.0)
    model = NonHermitianTwoLevel(sp)

    # 例：计算某点的 Prad 与 Ptot
    om, kk, g = 0.05, 0.2, 0.5
    print("Prad(om, k, g) = ", model.prad(om, kk, g))
    print("Ptot(om, k, g) = ", model.ptot(om, kk, g))

    # 例：Heff 能带（复数），并查看其实部
    k_vals = np.linspace(-1, 1, 5)
    b_plus, b_minus = model.bands_heff(k_vals, gamma_rad=g, sort_by_real=True)
    print("Re[band_plus] =", np.real(b_plus))
    print("Re[band_minus] =", np.real(b_minus))
