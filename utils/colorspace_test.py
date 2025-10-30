import numpy as np
import matplotlib.pyplot as plt
import colorspacious as cs

def complex_to_lch(theta, mag, L=70, C_max=70, chroma_gamma=1.0,
                   gamut_strategy="desaturate", max_iter=10):
    """
    将相位角/幅值映射到 CIELCh，再转 sRGB。
    参数
    ----
    theta : ndarray
        相位角（弧度），通常在 [-pi, pi]。
    mag : ndarray
        幅值，建议在 [0, 1]。
    L : float
        Lightness（0-100），数值越大越亮。
    C_max : float
        幅值=1 时的目标最大色度（通常 40~80 之间比较安全）。
    chroma_gamma : float
        Chroma 非线性：C = C_max * mag**gamma（>1 更保守，<1 更鲜艳）。
    gamut_strategy : {'clip', 'desaturate'}
        - 'clip'：直接裁剪到 [0,1]，速度快但可能失真。
        - 'desaturate'：对越界像素逐步降低 C，直到落入 sRGB。
    max_iter : int
        'desaturate' 的最多迭代次数。

    返回
    ----
    rgb : ndarray, float32
        sRGB in [0,1]，形状与 theta 相同 + (3,)。
    """
    # 角度 -> [0, 360)
    H = (np.degrees(theta) + 360.0) % 360.0
    # 幅值 -> 色度
    C = np.asarray(C_max, dtype=np.float32) * (np.asarray(mag, dtype=np.float32) ** chroma_gamma)

    # 组装 LCh
    Lch = np.stack([np.full_like(C, float(L)), C, H], axis=-1)  # (..., 3)

    # 初次转换
    rgb = cs.cspace_convert(Lch, start="CIELCh", end="sRGB1").astype(np.float32)

    if gamut_strategy == "clip":
        return np.clip(rgb, 0.0, 1.0)

    # 逐步降饱和（只降超出色域处的 C）
    if gamut_strategy == "desaturate":
        mask = (rgb < 0.0) | (rgb > 1.0)
        mask = np.any(mask, axis=-1)
        Lch_work = Lch.copy()
        for _ in range(max_iter):
            if not np.any(mask):
                break
            # 对越界像素降低 C
            Lch_work[mask, 1] *= 0.85
            rgb = cs.cspace_convert(Lch_work, start="CIELCh", end="sRGB1").astype(np.float32)
            mask = (rgb < 0.0) | (rgb > 1.0)
            mask = np.any(mask, axis=-1)
        return np.clip(rgb, 0.0, 1.0)

    raise ValueError("gamut_strategy 必须是 'clip' 或 'desaturate'。")


def make_test_pattern(width=1200, height=400,
                      L=70, C_max=70, chroma_gamma=1.0,
                      gamut_strategy="desaturate"):
    """
    生成矩形测试图案：
      - 水平长边：相位角 theta ∈ [-pi, pi]
      - 竖直方向：幅值 mag ∈ [0, 1]（底部 0，顶部 1）
    """
    theta = np.linspace(-np.pi, np.pi, width, endpoint=True)
    mag = np.linspace(0.0, 1.0, height, endpoint=True)

    T, M = np.meshgrid(theta, mag)  # 形状 (H, W)

    rgb = complex_to_lch(
        theta=T,
        mag=M,
        L=L,
        C_max=C_max,
        chroma_gamma=chroma_gamma,
        gamut_strategy=gamut_strategy,
        max_iter=12
    )

    return rgb


def main():
    # 你可以调这些参数来感受差异
    L = 70            # 亮度（50~75 常用）
    C_max = 70        # 最大色度（太大容易越界，40~80 较稳妥）
    gamma = 0.9       # 幅值->色度的非线性（<1 更鲜艳，>1 更内敛）
    gamut = "desaturate"  # 'clip' 或 'desaturate'

    rgb = make_test_pattern(
        width=1200, height=400,
        L=L, C_max=C_max, chroma_gamma=gamma,
        gamut_strategy=gamut
    )

    plt.figure(figsize=(10, 3.2))
    plt.imshow(rgb, origin='lower', extent=(-np.pi, np.pi, 0.0, 1.0), aspect='auto')
    plt.xlabel("angle θ (radians)")
    plt.ylabel("magnitude")
    plt.title(f"CIELCh mapping (L={L}, C_max={C_max}, gamma={gamma}, gamut={gamut})")
    plt.tight_layout()
    plt.show()

    # 也保存一份图片
    plt.imsave("lch_test_pattern.png", np.clip(rgb, 0, 1))

if __name__ == "__main__":
    main()
