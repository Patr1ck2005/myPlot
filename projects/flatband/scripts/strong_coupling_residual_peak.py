import numpy as np
import matplotlib.pyplot as plt

# --- 模型参数 ---
omega_ex = 2000.0   # 激子共振频率
gamma_ex = 5.0      # 激子线宽 (窄)

omega_cav = 2000.0  # 腔模/等离激元频率
gamma_cav = 60.0    # 腔模线宽 (宽)

g = 40.0            # 耦合强度 (产生 Rabi 分裂)

# --- 关键的物理假设: 空间非均匀性 ---
# 我们假设只有一部分激子参与了耦合，另一部分作为"剩余激子"存在
f_coupled = 0.8     # 参与耦合的比例 (比如 60% 的分子在热点中)
f_residual = 0.2    # 剩余激子的比例 (比如 40% 的分子在体相中)

# 频率扫描
omega = np.linspace(1800, 2200, 1000)

# --- 1. 计算参与耦合的部分 (Coupled System -> UPB & LPB) ---
# 使用耦合模式理论 (CMT) 计算这部分的吸收
denom = (omega - omega_cav + 1j*gamma_cav) * (omega - omega_ex + 1j*gamma_ex) - g**2
# 腔模的分量 (通过腔模激发)
a_cav = (omega - omega_ex + 1j*gamma_ex) / denom
# 激子的分量
a_ex = -g / denom
# 耦合系统的吸收 (主要由由于腔模衰减和激子衰减贡献)
# 这里简化: 假设主要通过腔模的远场激发进行探测
absorption_coupled = f_coupled * np.imag(a_cav) # 取虚部代表损耗/消光

# 为了更符合物理直觉的吸收谱形式 (Im[chi]):
# 上面的公式是场振幅，我们用更标准的极化率形式来表示吸收谱
# 耦合系统的极化率 chi_coupled ~ 1 / ( ... - g^2/...)
chi_coupled = 1 / ((omega - omega_cav - 1j*gamma_cav) - g**2 / (omega - omega_ex - 1j*gamma_ex))
spec_coupled = f_coupled * np.imag(chi_coupled)


# --- 2. 计算剩余激子部分 (Residual Excitons -> Original Peak) ---
# 这是一个标准的 Lorentz 线型，不受耦合强度 g 的影响
chi_residual = 1 / (omega - omega_ex - 1j*gamma_ex)
spec_residual = f_residual * np.imag(chi_residual)


# --- 3. 总光谱 ---
spec_total = spec_coupled + spec_residual

# --- 绘图 ---
plt.style.use('seaborn-v0_8-whitegrid')
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制
ax.plot(omega, spec_total, 'k-', linewidth=2.5, label='总吸收谱 (实验观测值)')
ax.fill_between(omega, spec_coupled, color='blue', alpha=0.3, label='耦合极化激元贡献 (UPB + LPB)')
ax.fill_between(omega, spec_residual, color='red', alpha=0.5, label='剩余激子贡献 (Residual Excitons)')

# 标注
ax.axvline(omega_ex, color='red', linestyle='--', alpha=0.5)
ax.text(omega_ex, np.max(spec_total)*1.02, '原始激子频率', ha='center', color='red')

ax.set_title('基于"剩余激子"假设的强耦合三峰光谱', fontsize=14)
ax.set_xlabel('频率 (meV)')
ax.set_ylabel('吸收 / 消光 (a.u.)')
ax.legend()
plt.tight_layout()
plt.show()
