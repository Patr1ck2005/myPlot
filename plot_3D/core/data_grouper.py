import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import linear_sum_assignment

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def estimate_derivative_one_sided(Zg, ni, nj, nk, s, axis, delta):
    """
    在邻点 (ni,nj,nk) 处，用单边差分估计沿 axis 方向的一阶导数。
    axis='x' 时，代表当前格点来自 (ni+1,nj,nk)，
    so we use a backward difference at (ni,nj,nk): (Zg[ni] - Zg[ni-1]) / delta.
    If backward is unavailable, fall back to forward difference.
    """
    if axis == 'x':
        # backward
        if ni-1 >= 0:
            return (Zg[ni, nj, nk, s] - Zg[ni-1, nj, nk, s]) / delta
        # forward
        elif ni+1 < Zg.shape[0]:
            return (Zg[ni+1, nj, nk, s] - Zg[ni, nj, nk, s]) / delta

    elif axis == 'y':
        if nj-1 >= 0:
            return (Zg[ni, nj, nk, s] - Zg[ni, nj-1, nk, s]) / delta
        elif nj+1 < Zg.shape[1]:
            return (Zg[ni, nj+1, nk, s] - Zg[ni, nj, nk, s]) / delta

    # 如果既无法后向也无法前向，返回 0
    return 0.0

def group_surfaces_one_sided_hungarian(Z, dx=1.0, dy=1.0, lam=10.0):
    """
    使用匈牙利算法 + 单边差分导数估计来完成分组。
    Z: list of shape [nx][ny][nz] each storing a list of m complex values.
    Returns Zg: ndarray of shape (nx,ny,nz,m).
    lam: 导数连续性权重.
    """
    nx, ny, nz = len(Z), len(Z[0]), len(Z[0][0])
    m = len(Z[0][0][0])

    # 分组结果和已赋值标记
    Zg = np.zeros((nx, ny, nz, m), dtype=complex)
    assigned = np.zeros((nx, ny, nz), dtype=bool)

    # 初始化起点 (0,0,0)
    Zg[0,0,0,:] = np.array(sorted(Z[0][0][0], key=lambda c: c.real))
    assigned[0,0,0] = True

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if assigned[i,j,k]:
                    continue

                candidates = Z[i][j][k][:]
                # 优先选 x 方向邻点
                if i > 0 and assigned[i-1,j,k]:
                    ni, nj, nk, axis, delta = i-1, j, k, 'x', dx
                # 否则选 y 方向邻点
                elif j > 0 and assigned[i,j-1,k]:
                    ni, nj, nk, axis, delta = i, j-1, k, 'y', dy
                else:
                    # 回退到起点
                    ni, nj, nk, axis, delta = 0, 0, 0, 'x', dx

                # 拿到参考值和导数
                v_prev = Zg[ni, nj, nk, :]
                d_prev = np.array([
                    estimate_derivative_one_sided(Zg, ni, nj, nk, s, axis, delta)
                    for s in range(m)
                ])

                # 构造 m×m 代价矩阵
                C = np.zeros((m, m))
                for s in range(m):
                    for c_idx, c in enumerate(candidates):
                        cost_val  = abs(c - v_prev[s])
                        cost_der  = abs((c - v_prev[s]) / delta - d_prev[s])
                        C[s, c_idx] = cost_val + lam * cost_der

                # 匈牙利算法全局最优匹配
                row_ind, col_ind = linear_sum_assignment(C)
                for s, c_idx in zip(row_ind, col_ind):
                    Zg[i, j, k, s] = candidates[c_idx]

                assigned[i, j, k] = True

    return Zg

if __name__ == '__main__':
    # Demo 参数
    nx, ny, nz, m = 100, 50, 1, 3
    dx, dy = 1.0, 1.0
    lam = 10.0

    # 构建 x,y 网格
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 生成 m 条真实曲面并打乱
    true_surfaces = np.zeros((nx, ny, m), dtype=complex)
    for s in range(m):
        true_surfaces[:,:,s] = np.sin(X + s) * np.cos(Y + s)

    Z = [[[None] for _ in range(ny)] for _ in range(nx)]
    for i in range(nx):
        for j in range(ny):
            vals = list(true_surfaces[i,j,:])
            random.shuffle(vals)
            Z[i][j][0] = vals

    # 分组
    Zg = group_surfaces_one_sided_hungarian(Z, dx=dx, dy=dy, lam=lam)

    # 可视化：固定 y 行的折线图
    j0 = ny // 2
    plt.figure(figsize=(8, 4))

    # 分组前
    plt.subplot(1, 2, 1)
    for i in range(nx):
        for val in Z[i][j0][0]:
            plt.scatter(x[i], val.real, s=5)
    plt.title(f'分组前 (y={y[j0]:.2f})')
    plt.xlabel('x'); plt.ylabel('值')

    # 分组后
    plt.subplot(1, 2, 2)
    for s in range(m):
        plt.plot(x, Zg[:, j0, 0, s].real, label=f'曲面{s}')
    plt.title(f'分组后 (y={y[j0]:.2f})')
    plt.xlabel('x'); plt.legend()

    plt.tight_layout()
    plt.show()
