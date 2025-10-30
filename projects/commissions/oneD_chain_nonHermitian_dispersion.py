import numpy as np
import matplotlib.pyplot as plt

# 参数定义（调整以见效果）
N = 100
kappa = 0.1
gamma = 0.01  # 减小，观察清晰峰而不扭曲
a = 1.0  # 设1简化
k_far = np.pi / 2  # 设大值，观察峰移动！试0, pi/4, pi/2

# 构建H0
H0 = np.zeros((N, N), dtype=complex)
for i in range(N-1):
    H0[i, i+1] = kappa
    H0[i+1, i] = kappa

# 构建K
j = np.arange(N)
K = np.sqrt(gamma) * np.exp(1j * k_far * j * a)  # 加a

# 构建V
K_dagger = K.conj()[:, np.newaxis]
K_row = K[np.newaxis, :]
V = -1j / 2 * np.dot(K_dagger, K_row)

# V相位图
plt.imshow(np.angle(V), cmap='hsv')
plt.colorbar(label='Phase of V')
plt.title(f'V (k_far={k_far:.2f})')
plt.show()

# H
H = H0 + V

# eig
vals, vecs = np.linalg.eig(H)

# 改进qs: 用投影法（更准于过阻尼模式）
q_tests = np.linspace(-np.pi, np.pi, 500)
qs = np.zeros(N)
for i in range(N):
    vec = vecs[:, i]
    vec = vec / np.linalg.norm(vec)  # 归一
    overlaps = np.abs([np.dot(vec.conj(), np.exp(1j * qt * j)) / N for qt in q_tests])  # 归一内积
    qs[i] = q_tests[np.argmax(overlaps)]

# 排序
sort_idx = np.argsort(qs)
qs_sorted = qs[sort_idx]
re_vals = np.real(vals[sort_idx])
im_vals = np.imag(vals[sort_idx])

# 找出superradiant模式
lossy_idx = np.argmin(im_vals)
print(f"Superradiant模式: ω = {vals[lossy_idx]}, q ≈ {qs[lossy_idx]:.2f} (应近k_far={k_far:.2f})")

# 理论
q_theory = np.linspace(-np.pi, np.pi, 1000)
omega_theory = 2 * kappa * np.cos(q_theory)

# 绘图（加突出lossy点）
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(qs_sorted, re_vals, color='blue', label='Re(ω)')
plt.scatter(qs_sorted[lossy_idx], re_vals[lossy_idx], color='red', s=100, label='Superradiant')
plt.plot(q_theory, omega_theory, 'r--', label='gamma=0')
plt.xlabel('q')
plt.ylabel('Re(ω)')
plt.title('实部')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(qs_sorted, im_vals, color='green', label='Im(ω)')
plt.scatter(qs_sorted[lossy_idx], im_vals[lossy_idx], color='red', s=100, label='Superradiant')
plt.xlabel('q')
plt.ylabel('Im(ω)')
plt.title('虚部 (损耗)')
plt.ylim(-gamma*N - 0.1, 0.1)  # 扩展ylim显示outlier
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 输出
print("所有Im(ω):", np.sort(np.imag(vals))[:5], "...", np.sort(np.imag(vals))[-5:])
print("注意: 大多数Im≈0，只一个大负值！这就是效果。")
