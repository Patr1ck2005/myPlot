import numpy as np
import matplotlib.pyplot as plt

# =========================
# 参数设置
# =========================
t1 = 0.6   # 胞内跳跃
t2 = 1.4   # 胞间跳跃
N  = 30    # 单胞数（总格点数 = 2N）

# 说明：
# t1 < t2 -> 拓扑相，会出现边缘态
# t1 > t2 -> 平庸相，不会有边缘态


# =========================
# 1. PBC 下的 SSH 体能带
# =========================
def ssh_bulk_bands(k, t1, t2):
    """
    SSH 模型在周期性边界条件下的 Bloch 哈密顿量本征值
    E(k) = ±| t1 + t2 e^{-ik} |
    """
    val = np.sqrt(t1**2 + t2**2 + 2*t1*t2*np.cos(k))
    return -val, val


# =========================
# 2. OBC 下有限 SSH 链哈密顿量
# =========================
def ssh_obc_hamiltonian(N, t1, t2):
    """
    构造有限长度 SSH 链（开放边界）哈密顿量
    格点顺序: A1, B1, A2, B2, ..., AN, BN
    总维数: 2N
    """
    dim = 2 * N
    H = np.zeros((dim, dim), dtype=float)

    for n in range(N):
        A = 2 * n
        B = 2 * n + 1

        # 胞内跃迁 t1: A_n <-> B_n
        H[A, B] = t1
        H[B, A] = t1

        # 胞间跃迁 t2: B_n <-> A_{n+1}
        if n < N - 1:
            A_next = 2 * (n + 1)
            H[B, A_next] = t2
            H[A_next, B] = t2

    return H


# =========================
# 3. 计算 PBC 能带
# =========================
k_list = np.linspace(-np.pi, np.pi, 500)
E_minus = []
E_plus = []

for k in k_list:
    e1, e2 = ssh_bulk_bands(k, t1, t2)
    E_minus.append(e1)
    E_plus.append(e2)

E_minus = np.array(E_minus)
E_plus = np.array(E_plus)


# =========================
# 4. 计算 OBC 本征谱
# =========================
H = ssh_obc_hamiltonian(N, t1, t2)
eigvals, eigvecs = np.linalg.eigh(H)

# 找到最接近零能的两个态（拓扑相下通常就是边缘态）
idx_sorted = np.argsort(np.abs(eigvals))
edge_idx1 = idx_sorted[0]
edge_idx2 = idx_sorted[1]

psi1 = eigvecs[:, edge_idx1]
psi2 = eigvecs[:, edge_idx2]

prob1 = np.abs(psi1)**2
prob2 = np.abs(psi2)**2


# =========================
# 5. 绘图
# =========================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# ---- (a) PBC 体能带 ----
ax = axes[0]
ax.plot(k_list, E_plus, label=r'$E_+(k)$')
ax.plot(k_list, E_minus, label=r'$E_-(k)$')
ax.set_title('SSH bulk bands (PBC)')
ax.set_xlabel(r'$k$')
ax.set_ylabel('Energy')
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
ax.grid(alpha=0.3)
ax.legend()

# ---- (b) OBC 全部本征谱 ----
ax = axes[1]
x = np.arange(len(eigvals))
ax.scatter(x, eigvals, s=22, label='All eigenstates')

# 高亮最接近零能的两个态
ax.scatter(edge_idx1, eigvals[edge_idx1], s=70, marker='o', label='Edge mode 1')
ax.scatter(edge_idx2, eigvals[edge_idx2], s=70, marker='o', label='Edge mode 2')

ax.axhline(0, linestyle='--', linewidth=1)
ax.set_title(f'Finite SSH spectrum (OBC), N={N}')
ax.set_xlabel('Eigenstate index')
ax.set_ylabel('Energy')
ax.grid(alpha=0.3)
ax.legend()

# ---- (c) 边缘态波函数分布 ----
ax = axes[2]
sites = np.arange(1, 2*N + 1)
ax.plot(sites, prob1, '-o', ms=3, label=f'Edge state 1, E={eigvals[edge_idx1]:.3e}')
ax.plot(sites, prob2, '-o', ms=3, label=f'Edge state 2, E={eigvals[edge_idx2]:.3e}')
ax.set_title('Probability distribution of edge modes')
ax.set_xlabel('Site index')
ax.set_ylabel(r'$|\psi|^2$')
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
