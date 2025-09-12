import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_c8_quasicrystal(
        size_pixels=512,  # 图像的像素大小 (例如 512x512)
        k_magnitude=2.0 * np.pi,  # 波矢量的大小，决定图案的尺度/频率
        num_waves=8,  # 对称性阶数 (C8 为 8)
        phase_offset_per_wave=0.0,  # 每个波的额外相位偏移，如 0.0, np.pi/num_waves, np.pi/(2*num_waves)
        amplitudes=None  # 每个波的振幅列表，None 表示所有振幅为 1.0
):
    """
    使用改进的平面波干涉法生成N重旋转对称的准晶图案。

    参数:
    size_pixels (int): 生成图像的边长像素数。
    k_magnitude (float): 平面波波数的大小。
    num_waves (int): 旋转对称的阶数 (N)。
    phase_offset_per_wave (float): 每个波的额外相位偏移量，以弧度计。
                                   例如，如果设置为 alpha，则第 j 个波的相位为 j * alpha。
                                   此参数对图案的准周期性影响显著。
    amplitudes (list/None): 每个波的振幅列表。如果为 None，则所有振幅默认为 1.0。
                            可以用于引入非均匀性或权重。

    返回:
    numpy.ndarray: 生成的准晶图案的二维数组。
    """

    # 创建空间网格
    x = np.linspace(-size_pixels / 2, size_pixels / 2, size_pixels)
    y = np.linspace(-size_pixels / 2, size_pixels / 2, size_pixels)
    X, Y = np.meshgrid(x, y)

    # 初始化强度数组
    intensity = np.zeros((size_pixels, size_pixels))

    # 设置振幅，如果未指定则都为1
    if amplitudes is None:
        amplitudes = [1.0] * num_waves
    elif len(amplitudes) != num_waves:
        raise ValueError(f"Amplitudes list must have {num_waves} elements or be None.")

    # 叠加平面波
    for j in range(num_waves):
        # 计算当前波的旋转角度
        angle = j * (2 * np.pi / num_waves)

        # 计算当前波的波矢量分量 (k_x, k_y)
        k_x = k_magnitude * np.cos(angle)
        k_y = k_magnitude * np.sin(angle)

        # 计算当前波的相位 (包含基准相位和额外偏移)
        # 这里的 phase_offset_per_wave 乘以 j 是一个常见的引入准周期性的方式
        # 如果 phase_offset_per_wave = 0，图案可能是周期性的或相对简单
        current_phase = j * phase_offset_per_wave

        # 计算平面波的相位项 k_j * r + phi_j
        wave_phase_term = k_x * X + k_y * Y + current_phase

        # 叠加当前平面波的贡献
        intensity += amplitudes[j] * np.cos(wave_phase_term)

    # 归一化强度到 [0, 1] 区间以便可视化
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())

    return intensity


# --- 示例使用 ---

if __name__ == "__main__":
    # --- 基础 C8 准晶图案 ---
    print("生成基础 C8 准晶图案 (相位偏移: pi/8)...")
    c8_pattern_basic = generate_c8_quasicrystal(
        size_pixels=512,
        k_magnitude=2 * np.pi / 20,  # 调整 k_magnitude 可以改变图案的“精细度”
        num_waves=8,
        phase_offset_per_wave=np.pi / 8  # 一个典型的相位偏移，有助于生成准晶特征
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(c8_pattern_basic, cmap=cm.viridis, origin='lower')
    plt.title("C8 Quasicrystal Pattern (Phase Offset: $\\pi/8$)")
    plt.axis('off')
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.show()

    # --- 带有不同相位偏移的 C8 准晶图案 (演示“改进”) ---
    print("生成不同相位偏移的 C8 准晶图案 (演示改进)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 案例 1: 零相位偏移 (可能更接近周期性)
    c8_pattern_zero_phase = generate_c8_quasicrystal(
        size_pixels=512,
        k_magnitude=2 * np.pi / 20,
        num_waves=8,
        phase_offset_per_wave=0.0
    )
    axes[0].imshow(c8_pattern_zero_phase, cmap=cm.viridis, origin='lower')
    axes[0].set_title("C8 Quasicrystal (Phase Offset: 0)")
    axes[0].axis('off')

    # 案例 2: 略微不同的相位偏移
    c8_pattern_alt_phase1 = generate_c8_quasicrystal(
        size_pixels=512,
        k_magnitude=2 * np.pi / 20,
        num_waves=8,
        phase_offset_per_wave=np.pi / 16  # 更小的相位偏移
    )
    axes[1].imshow(c8_pattern_alt_phase1, cmap=cm.viridis, origin='lower')
    axes[1].set_title("C8 Quasicrystal (Phase Offset: $\\pi/16$)")
    axes[1].axis('off')

    # 案例 3: 进一步的相位偏移
    c8_pattern_alt_phase2 = generate_c8_quasicrystal(
        size_pixels=512,
        k_magnitude=2 * np.pi / 20,
        num_waves=8,
        phase_offset_per_wave=np.pi  # 更大的相位偏移，可能产生不同结构
    )
    axes[2].imshow(c8_pattern_alt_phase2, cmap=cm.viridis, origin='lower')
    axes[2].set_title("C8 Quasicrystal (Phase Offset: $\\pi/4$)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # --- 带有自定义振幅的 C8 准晶图案 ---
    print("生成自定义振幅的 C8 准晶图案...")
    # 例如，让某些方向的波更强，破坏完美旋转对称性，但整体轮廓仍保持
    # 注意：如果振幅不均匀，图案可能不再严格满足C8群的平凡表示，
    # 而是某些C8群的子群表示，或根本没有严格C8对称性，但视觉上仍有8重特征
    custom_amplitudes = [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8]  # 交替振幅
    c8_pattern_custom_amp = generate_c8_quasicrystal(
        size_pixels=512,
        k_magnitude=2 * np.pi / 20,
        num_waves=8,
        phase_offset_per_wave=np.pi / 8,
        amplitudes=custom_amplitudes
    )
    plt.figure(figsize=(8, 8))
    plt.imshow(c8_pattern_custom_amp, cmap=cm.viridis, origin='lower')
    plt.title("C8 Quasicrystal (Custom Amplitudes)")
    plt.axis('off')
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.show()

