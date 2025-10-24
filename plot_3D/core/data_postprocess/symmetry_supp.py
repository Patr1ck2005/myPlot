import numpy as np

def _wrap_angle_pi(phi):
    # 将角度 wrap 到 (-pi/2, pi/2] 或 [0,pi) 都行；这里用 (-pi/2, pi/2]
    out = (phi + np.pi/2) % np.pi - np.pi/2
    return out

def complete_C4_polarization(kx, ky, phi_Q1, chi_Q1, mode='xy_mirror+ C4'):
    """
    基于物理对称（时间反演 + C4 旋转）把 Q1 的 (phi, chi) 扩展到全平面。
    约定：
      - phi: 线偏振方位角，模 pi；关于原点偶；C4 旋转时加 pi/2
      - chi: 圆偏振度 ~ S3/S0；关于原点奇；C4 旋转不变
    输入：
      kx, ky: 一维、严格递增、从 0 开始（包含 0）
      phi_Q1, chi_Q1: 形状 (len(kx), len(ky))，对应 kx>=0, ky>=0
    输出：
      kx_full, ky_full, phi_full, chi_full  （可直接 meshgrid(indexing='ij') ）
    """
    kx = np.asarray(kx); ky = np.asarray(ky)
    phi_Q1 = np.asarray(phi_Q1); chi_Q1 = np.asarray(chi_Q1)
    assert kx.ndim==1 and ky.ndim==1 and np.all(np.diff(kx)>0) and np.all(np.diff(ky)>0)
    assert np.isclose(kx[0], 0) and np.isclose(ky[0], 0)
    assert phi_Q1.shape[:2] == (kx.size, ky.size)
    # 先做 xy 镜像（含物理奇偶）
    def mirror_axis(arr):
        left = -arr[::-1]
        left = left[1:] if np.isclose(arr[0],0) else left
        return np.concatenate([left, arr], axis=0)

    kx_full = mirror_axis(kx)
    ky_full = mirror_axis(ky)

    # 先把 Q1 放到右上象限
    phi = phi_Q1.copy()
    chi = chi_Q1.copy()

    # 关于 x 轴镜像 (ky -> -ky)：phi 偶，chi 奇（镜像翻手性）
    # 右下象限：
    phi_RB = _wrap_angle_pi(phi[:, 1:])             # phi 偶
    chi_RB = -chi[:, 1:]                            # chi 奇

    # 组合右半平面
    phi_R = np.concatenate([phi_RB, phi], axis=1)
    chi_R = np.concatenate([chi_RB, chi], axis=1)

    # 关于 y 轴镜像 (kx -> -kx)：phi 偶，chi 奇
    # 左半平面：
    phi_L = _wrap_angle_pi(np.flip(phi_R[1:, :], axis=0))
    chi_L = -np.flip(chi_R[1:, :], axis=0)

    # 拼成全平面
    phi_full = np.concatenate([phi_L, phi_R], axis=0)
    chi_full = np.concatenate([chi_L, chi_R], axis=0)

    if mode.lower().startswith('xy_mirror'):
        # 只做奇偶扩展
        return kx_full, ky_full, phi_full, chi_full

    if 'c4' in mode.lower():
        # 用 C4 再约束一遍：R_{90°} 作用在 (S1,S2) 相当于 phi -> phi + pi/2；chi 不变
        # 数值上：把旋转后的结果与当前结果做一致化（例如取平均/强制替换）
        # 这里采用“弱一致化”：对四个 90° 旋转的版本做多数投票式平均
        def rot90_field(F):
            return np.rot90(F)  # 形状 (Ny,Nx) → 我们要的是 (Nx,Ny)，因此再转置回来
        # 为了轴对齐，要求网格近似方阵；否则直接返回 mirror 结果
        if phi_full.shape[0] == phi_full.shape[1]:
            # 旋转索引映射：R90: (i,j) -> (j, N-1-i)
            N = phi_full.shape[0]
            # 构造 R90 约束后的 phi
            phi_r = np.rot90(phi_full, k=1) + np.pi/2
            phi_r = _wrap_angle_pi(phi_r)
            # 合并（对角度采用“向量平均”避免跳变）
            v1 = np.stack([np.cos(2*phi_full), np.sin(2*phi_full)], axis=0)
            v2 = np.stack([np.cos(2*phi_r),   np.sin(2*phi_r)],   axis=0)
            v = v1 + v2
            phi_full = 0.5*np.arctan2(v[1], v[0])
            # chi：R90 不变；但数值上可以与旋转版本做平均以降噪
            chi_r = np.rot90(chi_full, k=1)
            chi_full = 0.5*(chi_full + chi_r)
        # 若非方阵，不强制 C4 旋转一致化
    return kx_full, ky_full, phi_full, chi_full
