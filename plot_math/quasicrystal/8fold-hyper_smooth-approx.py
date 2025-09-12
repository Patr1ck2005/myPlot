import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def sqrt2_convergent(order: int):
    if order < 1:
        raise ValueError("order must be >= 1")
    a_prev2, b_prev2 = 0, 1
    a_prev1, b_prev1 = 1, 0
    a0 = 1
    a, b = a0 * a_prev1 + 1 * a_prev2, a0 * b_prev1 + 1 * b_prev2
    if order == 1:
        return a, b
    for _ in range(order - 1):
        a, b, a_prev1, b_prev1 = 2 * a + a_prev1, 2 * b + b_prev1, a, b
    return a, b

def generate_quasicrystal_8fold_np(
    width=512, height=512,
    mode="quasi",
    order=2,
    L=None,
    base_k=1.0,
    phase=math.pi,
    phase_shift=0,
    threshold=None
):
    if L is None:
        L = min(width, height)

    if mode == "approximant":
        a, b = sqrt2_convergent(order)
        M, p = a, b
        wave_ints = np.array([
            ( M, 0), ( 0, M), (-M, 0), ( 0,-M),
            ( p, p), (-p, p), ( p,-p), (-p,-p),
        ], dtype=float)

        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        xx_mod = xx % L
        yy_mod = yy % L

        z = np.zeros((height, width), dtype=float)
        for i, (m, n) in enumerate(wave_ints):
            z += np.cos(2 * np.pi * (m * xx_mod + n * yy_mod) / L + i * phase + phase_shift)

    elif mode == "quasi":
        phis = np.arange(8) * (np.pi / 4)
        x = np.linspace(-2*np.pi, 2*np.pi, width)
        y = np.linspace(-2*np.pi, 2*np.pi, height)
        xx, yy = np.meshgrid(x, y)

        z = np.zeros((height, width), dtype=float)
        for i, phi in enumerate(phis):
            z += np.cos(base_k * (xx * np.cos(phi) + yy * np.sin(phi)) + i * phase + phase_shift)

    else:
        raise ValueError("mode must be 'quasi' or 'approximant'")

    # plt.imshow(z, cmap='bwr')
    # plt.colorbar()
    # plt.show()
    # plt.close()

    # z = np.abs(z)
    # # normalize
    # z = 1 - (z / 8.0)

    # plt.imshow(z, cmap='bwr')
    # plt.colorbar()
    # plt.show()
    # plt.close()

    # normalize
    z = (z / 8.0 + 1) / 2
    z = np.clip(z * 255, 0, 255).astype(np.uint8)

    if threshold is not None:
        z = np.where(z > threshold, 255, 0).astype(np.uint8)
        print('fill factor', z.mean()/255)

    return Image.fromarray(z, mode='L').convert("RGB")

if __name__ == "__main__":
    img1 = generate_quasicrystal_8fold_np(
        width=512*2, height=512*2,
        mode="approximant",
        order=3,
        L=128*4,
        phase=math.pi*0,
        phase_shift=math.pi,
        threshold=89
    )
    img1.save("qc_8fold_approximant_order2.png")

    img2 = generate_quasicrystal_8fold_np(
        width=512*8, height=512*8,
        mode="quasi",
        base_k=36.0,
        phase=math.pi*0,
        phase_shift=math.pi*1,
        threshold=110
    )
    img2.save("qc_8fold_quasi.png")
