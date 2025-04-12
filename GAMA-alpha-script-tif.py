import os
import numpy as np
from PIL import Image
import tifffile


def composite_background_highbit(img_array, bg_color):
    """
    在高精度数据上合成背景，不先转换为 uint8。

    参数：
      - img_array: numpy 数组，形状 (H, W, 4)，为 TIFF 原始数据（float 或 integer）。
      - bg_color: 背景颜色 tuple，长度为3。
            对于浮点型数据，背景颜色应为 (1.0, 1.0, 1.0) 或 (0.0, 0.0, 0.0)；
            对于整数数据，背景颜色应为 (max, max, max) 或 (0, 0, 0)，其中 max 为该数据类型的最大值。

    返回：
      - 合成后的 RGB 数据，数据类型与计算保持 float32（若整数则先转换为 float32 再计算）。
    """
    # 判断数据类型
    if np.issubdtype(img_array.dtype, np.floating):
        # 假定浮点数据在 [0,1] 范围内，alpha 通道也位于 [0,1]
        alpha = img_array[..., 3]
        # alpha = alpha ** (1/2.2)  # gamma 校正
        out = np.empty(img_array.shape[:2] + (3,), dtype=img_array.dtype)
        # 对每个 RGB 通道做线性叠加
        for c in range(3):
            out[..., c] = img_array[..., c] * alpha + bg_color[c] * (1 - alpha)
        return out
    elif np.issubdtype(img_array.dtype, np.integer):
        # 对于整数数据，按最大值归一化。先转换为 float32 进行合成计算
        info = np.iinfo(img_array.dtype)
        max_val = info.max
        alpha = img_array[..., 3].astype(np.float32) / max_val
        # alpha = alpha ** (1/2.2)  # gamma 校正
        # 如果背景颜色传入的是整数（如 255），转换为 float32
        bg_color = tuple(float(c) for c in bg_color)
        out = np.empty(img_array.shape[:2] + (3,), dtype=np.float32)
        for c in range(3):
            out[..., c] = img_array[..., c].astype(np.float32) * alpha + bg_color[c] * (1 - alpha)
        return out
    else:
        raise ValueError("不支持的数据类型：{}".format(img_array.dtype))


def convert_to_uint8(array):
    """
    将合成后的图像数组转换到 uint8 范围 [0,255]。
    如果数据为 float 型，假定数据范围已经在 [0,1]（例如浮点型合成后的数据）；
    如果为整数，则线性归一化到 [0,255]。
    """
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        # 若数据为浮点且值大致在 [0,1]，直接乘 255
        array = np.clip(array, 0, 1) * 255
        return array.astype(np.uint8)
    elif np.issubdtype(array.dtype, np.integer):
        # 对于整数型，先将范围映射到 [0,255]
        info = np.iinfo(array.dtype)
        array = (array.astype(np.float32) / info.max) * 255.0
        array = np.clip(array, 0, 255)
        return array.astype(np.uint8)
    else:
        raise ValueError("不支持的数据类型：{}".format(array.dtype))


def apply_gamma_correction(array, gamma=2.2):
    """
    对 TIFF 图像应用 Gamma 校正。
    参数：
      - array: 32位数组
      - gamma: Gamma 校正值，默认 2.2
    返回：
      - 校正后的 32 位数组
    """
    inv_gamma = 1.0 / gamma
    return pow(array, inv_gamma)


def process_tiff_image(image_path, output_dir, gamma=2.2, apply_gamma=False):
    """
    处理单个 TIFF 图片：
      1. 读取 32 位 TIFF 原始数据（支持浮点或整数），要求数据为 RGBA 格式（shape: H×W×4）。
      2. 在原始 32 位数据上分别叠加白色和黑色背景进行合成，
         合成公式为：out = fg * α + bg * (1 – α)（α 归一化到 [0,1]）。
      3. 将合成结果转换为 uint8（可选在转换后对结果应用 Gamma 校正）。
      4. 分别保存为 JPG 和 PNG 格式，文件名后增加 _white 和 _black。
    """
    try:
        # 读取 TIFF 文件，得到原始 numpy 数组
        img_array = tifffile.imread(image_path)
    except Exception as e:
        print(f"读取 {os.path.basename(image_path)} 出错：{e}")
        return

    # 检查是否为 RGBA 数据
    if img_array.ndim != 3 or img_array.shape[-1] != 4:
        print(f"跳过 {os.path.basename(image_path)}（没有 alpha 通道或数据维度不符合要求）")
        return

    # 根据原始数据类型确定背景颜色
    if np.issubdtype(img_array.dtype, np.floating):
        bg_white = (1.0, 1.0, 1.0)
        bg_black = (0.0, 0.0, 0.0)
    elif np.issubdtype(img_array.dtype, np.integer):
        info = np.iinfo(img_array.dtype)
        bg_white = (info.max, info.max, info.max)
        bg_black = (0, 0, 0)
    else:
        print(f"不支持的数据类型：{img_array.dtype}")
        return

    # 在原始 32 位数据上合成背景
    comp_white = composite_background_highbit(img_array, bg_white)
    comp_black = composite_background_highbit(img_array, bg_black)

    # 可选：对输出图像进行 Gamma 校正（应在转换到 uint8 前进行）
    if apply_gamma:
        comp_white = apply_gamma_correction(comp_white, gamma)
        comp_black = apply_gamma_correction(comp_black, gamma)

    # 将合成后的高精度数据转换为 uint8 用于保存
    comp_white_uint8 = convert_to_uint8(comp_white)
    comp_black_uint8 = convert_to_uint8(comp_black)

    # 转换为 PIL Image（RGB模式）
    img_white = Image.fromarray(comp_white_uint8, mode="RGB")
    img_black = Image.fromarray(comp_black_uint8, mode="RGB")

    # 构造输出文件名（在原文件名基础上添加后缀）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    white_jpg = os.path.join(output_dir, f"{base_name}_white.jpg")
    white_png = os.path.join(output_dir, f"{base_name}_white.png")
    black_jpg = os.path.join(output_dir, f"{base_name}_black.jpg")
    black_png = os.path.join(output_dir, f"{base_name}_black.png")

    try:
        img_white.save(white_jpg, quality=95)
        img_white.save(white_png)
        img_black.save(black_jpg, quality=95)
        img_black.save(black_png)
        print(f"已处理 {os.path.basename(image_path)} ，生成：")
        print(f"    {white_jpg}")
        print(f"    {white_png}")
        print(f"    {black_jpg}")
        print(f"    {black_png}")
    except Exception as e:
        print(f"保存 {os.path.basename(image_path)} 时出错：{e}")


def process_tiff_directory(input_dir, output_dir, gamma=2.2, apply_gamma=False):
    """
    遍历指定目录下所有 .tif/.tiff 文件，对每个文件调用 process_tiff_image 进行处理。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_ext = ('.tif', '.tiff')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_ext):
            image_path = os.path.join(input_dir, filename)
            process_tiff_image(image_path, output_dir, gamma, apply_gamma)


def main(working_directory, apply_gamma=False):
    """
    将指定目录下的 TIFF 文件处理后存入 working_directory/TIFF_composite 下。
    参数 apply_gamma 控制是否对输出图像应用 Gamma 校正。
    """
    output_directory = os.path.join(working_directory, 'TIFF_composite')
    process_tiff_directory(working_directory, output_directory, gamma=2.2, apply_gamma=apply_gamma)
    print("所有 TIFF 图片处理完成！")


if __name__ == '__main__':
    # 修改为实际工作目录，并根据需要决定是否应用 gamma 校正（True 表示应用）
    main(r'D:\DELL\Documents\myPlots\examples', apply_gamma=True)
