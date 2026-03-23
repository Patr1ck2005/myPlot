import numpy as np
import matplotlib.pyplot as plt

def ssh_obc_hamiltonian(N, t1, t2):
    """
    SSH 模型 OBC 哈密顿量
    格点顺序: A1, B1, A2, B2, ..., AN, BN
    总维数: 2N
    """
    dim = 2 * N
    H = np.zeros((dim, dim), dtype=float)

    for n in range(N):
        A = 2 * n
        B = 2 * n + 1

        # 胞内跃迁
        H[A, B] = t1
        H[B, A] = t1

        # 胞间跃迁
        if n < N - 1:
            A_next = 2 * (n + 1)
            H[B, A_next] = t2
            H[A_next, B] = t2

    return H


# =========================
# 参数设置
# =========================
N = 40          # 单胞数
t2 = 1.0        # 固定 t2
ratio_list = np.linspace(0.0, 2.0, 300)   # 横轴: t1/t2

all_eigs = []

# =========================
# 扫描 t1/t2
# =========================
for ratio in ratio_list:
    t1 = ratio * t2
    H = ssh_obc_hamiltonian(N, t1, t2)
    eigvals = np.linalg.eigvalsh(H)
    all_eigs.append(eigvals)

all_eigs = np.array(all_eigs)   # shape = (len(ratio_list), 2N)

# =========================
# 绘图
# =========================
plt.figure(figsize=(8, 6))

for n in range(all_eigs.shape[1]):
    plt.plot(ratio_list, all_eigs[:, n], color='black', linewidth=0.8)

plt.axvline(1.0, linestyle='--', linewidth=1.2, color='red', label=r'$t_1/t_2=1$')
plt.axhline(0.0, linestyle='--', linewidth=1.0, color='gray')

plt.xlabel(r'$t_1/t_2$', fontsize=13)
plt.ylabel('Energy', fontsize=13)
plt.title(f'SSH spectrum under OBC (N={N})', fontsize=14)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()
