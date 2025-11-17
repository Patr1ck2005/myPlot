import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# # chinese font
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# =====================#
#     基本设置
# =====================#

# k 轴范围（只看一维 |k|，模拟 Γ-X 或 Γ-M）
k_max = 1.5
num_k = 400
k = np.linspace(-k_max, k_max, num_k)

# 初始参数（你可以随便调）
wA0_init = 1.0   # 裸 A 带 Γ 点频率
wB0_init = 1.3   # 裸 B 带 Γ 点频率
aA_init  = 0.5   # 裸 A 带曲率 alpha_A
aB_init  = -0.5  # 裸 B 带曲率 alpha_B
g_init   = 0.15  # A/B 之间耦合强度

# =====================#
#   核心计算函数
# =====================#

def compute_bands_and_eigvecs(k, wA0, wB0, aA, aB, g):
    """
    计算：
    - 裸带 wA(k), wB(k)
    - 混合后的两条带 w_plus, w_minus
    - 低频带的本征矢 (c_A, c_B) 在每个 k 上（规范化，c_A>0）
    """
    wA = wA0 + aA * k**2
    wB = wB0 + aB * k**2

    delta = wA - wB
    rad = np.sqrt((delta / 2.0) ** 2 + g**2)

    w_plus  = 0.5 * (wA + wB) + rad
    w_minus = 0.5 * (wA + wB) - rad

    # 低频带本征矢：从 (H - w_minus)v = 0 求得
    # (wA - w_minus) c_A + g c_B = 0 -> c_B/c_A = (w_minus - wA)/g
    r = (w_minus - wA) / g  # c_B/c_A

    # 归一化
    cA = 1.0 / np.sqrt(1.0 + np.abs(r)**2)
    cB = r * cA

    return wA, wB, w_plus, w_minus, cA, cB, r

def compute_beta_at_gamma(wA0, wB0, aA, aB, g):
    """
    使用解析公式计算 Γ 点处低频带的二阶曲率 beta
    beta(Δ0) = 2*bar_alpha - 2 * Δ0 * delta_alpha / sqrt(Δ0^2 + 4 g^2)
    """
    Delta0 = wA0 - wB0
    bar_alpha   = 0.5 * (aA + aB)
    delta_alpha = 0.5 * (aA - aB)

    denom = np.sqrt(Delta0**2 + 4.0 * g**2)
    beta = 2.0 * bar_alpha - 2.0 * Delta0 * delta_alpha / denom
    return beta

# =====================#
#     画布和子图
# =====================#

plt.close('all')
fig = plt.figure(figsize=(10, 7))
fig.subplots_adjust(left=0.08, right=0.75, bottom=0.30, top=0.93)

# 子图1：能带
ax_band = fig.add_subplot(2, 1, 1)
ax_band.set_title("Coupled A/B Bands and Mixed Modes")
ax_band.set_xlabel("k (arb. units)")
ax_band.set_ylabel("ω")

# 子图2：低频带 A/B 权重
ax_weight = fig.add_subplot(2, 1, 2)
ax_weight.set_xlabel("k")
ax_weight.set_ylabel("|c|^2 (lower band)")
ax_weight.set_ylim(-0.05, 1.05)

# 先用初始参数计算
wA, wB, w_plus, w_minus, cA, cB, r = compute_bands_and_eigvecs(
    k, wA0_init, wB0_init, aA_init, aB_init, g_init
)

beta0 = compute_beta_at_gamma(wA0_init, wB0_init, aA_init, aB_init, g_init)
idx_gamma = np.argmin(np.abs(k))  # 最近的 k≈0 索引

# =====================#
#    初始绘图对象
# =====================#

# 能带：画上裸带和混合带
line_wA, = ax_band.plot(k, wA,  linestyle="--", label="bare A")
line_wB, = ax_band.plot(k, wB,  linestyle="--", label="bare B")
line_wm, = ax_band.plot(k, w_minus, label="mixed lower band")
line_wp, = ax_band.plot(k, w_plus,  label="mixed upper band")
ax_band.legend(loc="upper right")

# 权重：低频带中 |c_A|^2 和 |c_B|^2
line_wA2, = ax_weight.plot(k, np.abs(cA)**2, label="|c_A|^2 (lower)")
line_wB2, = ax_weight.plot(k, np.abs(cB)**2, label="|c_B|^2 (lower)")
ax_weight.legend(loc="upper right")

# 在 band 图上标记 Γ 点位置
ax_band.axvline(0.0, linestyle=":", alpha=0.5)

# 在右侧加文本框，显示 Γ 点的 r_gamma 和 beta 等信息
info_text = fig.text(
    0.77, 0.55,
    "",
    fontsize=10,
    va="top",
    family="monospace"
)


def update_info_text(wA0, wB0, aA, aB, g, r_gamma, beta):
    text = (
        f"Parameters:\n"
        f"  ωA0 = {wA0:6.3f}\n"
        f"  ωB0 = {wB0:6.3f}\n"
        f"  αA  = {aA:6.3f}\n"
        f"  αB  = {aB:6.3f}\n"
        f"  g   = {g:6.3f}\n"
        f"\n"
        f"Γ-point (lower band):\n"
        f"  r_Γ = c_B/c_A = {r_gamma:7.3f}\n"
        f"  sign(r_Γ) < 0 → A/B 反相\n"
        f"\n"
        f"Curvature at Γ (lower):\n"
        f"  β = d²ω/dk² ≈ {beta:7.3f}\n"
        f"  β>0: 向上开口\n"
        f"  β<0: 向下开口\n"
        f"  β≈0: 平带临界态\n"
    )
    info_text.set_text(text)

# 初始化 info_text
update_info_text(
    wA0_init, wB0_init, aA_init, aB_init, g_init,
    r[idx_gamma], beta0
)

# 设置 y 轴范围（根据初值预估一个范围）
all_omega = np.concatenate([wA, wB, w_plus, w_minus])
ax_band.set_ylim(all_omega.min() - 0.2, all_omega.max() + 0.2)

# =====================#
#       滑动条
# =====================#

axcolor = 'lightgoldenrodyellow'

ax_wA0 = plt.axes([0.10, 0.20, 0.60, 0.03], facecolor=axcolor)
ax_wB0 = plt.axes([0.10, 0.16, 0.60, 0.03], facecolor=axcolor)
ax_aA  = plt.axes([0.10, 0.12, 0.60, 0.03], facecolor=axcolor)
ax_aB  = plt.axes([0.10, 0.08, 0.60, 0.03], facecolor=axcolor)
ax_g   = plt.axes([0.10, 0.04, 0.60, 0.03], facecolor=axcolor)

s_wA0 = Slider(ax_wA0, 'ωA0', 0.0, 2.0, valinit=wA0_init)
s_wB0 = Slider(ax_wB0, 'ωB0', 0.0, 2.0, valinit=wB0_init)
s_aA  = Slider(ax_aA,  'αA', -1.0, 1.0, valinit=aA_init)
s_aB  = Slider(ax_aB,  'αB', -1.0, 1.0, valinit=aB_init)
s_g   = Slider(ax_g,   'g',   0.0, 1.0,  valinit=g_init)

# =====================#
#      更新函数
# =====================#

def update(val):
    wA0 = s_wA0.val
    wB0 = s_wB0.val
    aA  = s_aA.val
    aB  = s_aB.val
    g   = s_g.val if s_g.val != 0 else 1e-6  # 避免 g=0 的除零

    # 重新计算
    wA, wB, w_plus, w_minus, cA, cB, r = compute_bands_and_eigvecs(
        k, wA0, wB0, aA, aB, g
    )

    beta = compute_beta_at_gamma(wA0, wB0, aA, aB, g)

    # 更新曲线数据
    line_wA.set_ydata(wA)
    line_wB.set_ydata(wB)
    line_wm.set_ydata(w_minus)
    line_wp.set_ydata(w_plus)

    line_wA2.set_ydata(np.abs(cA)**2)
    line_wB2.set_ydata(np.abs(cB)**2)

    # 自适应 y 轴范围（防止看不见）
    all_omega = np.concatenate([wA, wB, w_plus, w_minus])
    ax_band.set_ylim(all_omega.min() - 0.2, all_omega.max() + 0.2)

    # 更新 info 文本（用最近的 k≈0 点）
    idx_gamma = np.argmin(np.abs(k))
    r_gamma = r[idx_gamma]
    update_info_text(wA0, wB0, aA, aB, g, r_gamma, beta)

    fig.canvas.draw_idle()

# 把滑动条和更新函数连起来
s_wA0.on_changed(update)
s_wB0.on_changed(update)
s_aA.on_changed(update)
s_aB.on_changed(update)
s_g.on_changed(update)

# 加一个 reset 按钮
reset_ax = plt.axes([0.72, 0.02, 0.10, 0.04])
button_reset = Button(reset_ax, 'Reset')

def reset(event):
    s_wA0.reset()
    s_wB0.reset()
    s_aA.reset()
    s_aB.reset()
    s_g.reset()

button_reset.on_clicked(reset)

plt.show()
