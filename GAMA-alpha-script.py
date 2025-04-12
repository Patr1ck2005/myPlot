import os
import numpy as np
from PIL import Image
import tifffile


def process_alpha_channel(array, gamma=2.2):
    """
    对图像数组中的 alpha 通道进行处理：
    1. 如果 alpha 通道的数据类型不是 uint8，则先归一化到 [0, 255]。
    2. 对归一化后的 alpha 通道进行 Gamma 校正（不改变 RGB 通道）。
    """
    # 仅处理 alpha 通道（假设 alpha 通道位于最后一个维度）
    alpha = array[..., 3]

    # 如果数据类型不是 uint8，则归一化
    if alpha.dtype != np.uint8:
        amin = np.min(alpha)
        amax = np.max(alpha)
        if amax > amin:
            alpha = ((alpha - amin) / (amax - amin)) * 255.0
        else:
            alpha = np.zeros_like(alpha)
        alpha = alpha.astype(np.uint8)

    # 应用 Gamma 校正
    alpha_corr = np.power(alpha / 255.0, 1 / gamma) * 255.0
    array[..., 3] = alpha_corr.astype(np.uint8)
    return array


def process_png(image_path, gamma=2.2):
    """
    使用 PIL 处理 PNG 文件，要求图像具有 alpha 通道，
    处理方式只校正 alpha 通道，RGB 保持不变。
    """
    with Image.open(image_path) as img:
        if 'A' in img.getbands():
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            img_array = np.array(img)
            img_array = process_alpha_channel(img_array, gamma)
            return Image.fromarray(img_array)
        else:
            print(f"跳过：{os.path.basename(image_path)}（没有 alpha 通道）")
            return None


def process_tiff(image_path, gamma=2.2):
    """
    使用 tifffile 处理 TIFF 文件，支持 32 位 TIFF，
    只处理具有 alpha 通道的图像。
    """
    try:
        # 读取 TIFF 文件
        img_array = tifffile.imread(image_path)
    except Exception as e:
        print(f"读取 {os.path.basename(image_path)} 时出错：{e}")
        return None

    # 检查图像数组是否至少为三维且最后通道数为 4
    if img_array.ndim == 3 and img_array.shape[-1] == 4:
        # 对 alpha 通道进行处理
        processed_array = process_alpha_channel(img_array, gamma)
        return processed_array
    else:
        print(f"跳过：{os.path.basename(image_path)}（没有 alpha 通道或维度不符合要求）")
        return None


def process_images(input_dir, output_dir, gamma=2.2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ('.png', '.tif', '.tiff')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(input_dir, filename)
            try:
                if filename.lower().endswith('.png'):
                    result = process_png(image_path, gamma)
                    if result is not None:
                        out_path = os.path.join(output_dir, filename)
                        result.save(out_path)
                        print(f"已处理并保存：{filename}")
                else:  # 处理 tiff 文件
                    processed_array = process_tiff(image_path, gamma)
                    if processed_array is not None:
                        out_path = os.path.join(output_dir, filename)
                        tifffile.imwrite(out_path, processed_array)
                        print(f"已处理并保存：{filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错：{e}")


def main(working_directory):
    # 输出目录设为输入目录下的 GAMA-alpha 文件夹
    output_directory = os.path.join(working_directory, 'GAMA-alpha')
    process_images(working_directory, output_directory)
    print("所有图片处理完成！")


if __name__ == '__main__':
    # 修改为你的实际目录路径
    main(r'D:\DELL\Documents\myPlots\examples')
