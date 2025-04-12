import os
from PIL import Image
import numpy as np

def apply_gamma_correction_to_alpha(image, gamma=2.2):
    """
    对图像的 alpha 通道进行 Gamma 校正，不改变 RGB 通道。
    """
    img_array = np.array(image)
    # 检查图像是否具有 alpha 通道（数组最后一维为4）
    if img_array.shape[2] == 4:
        a_channel = img_array[..., 3]
        # 对 alpha 通道进行 Gamma 校正
        corrected_alpha = np.power(a_channel / 255.0, 1 / gamma) * 255
        img_array[..., 3] = corrected_alpha.astype(np.uint8)
        corrected_image = Image.fromarray(img_array)
        return corrected_image
    else:
        return image

def process_images(input_dir, output_dir, gamma=2.2):
    """
    遍历目录，处理扩展名为 .png、.tif 和 .tiff 的图像，
    仅处理有 alpha 通道的图像进行 Gamma 校正。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图片扩展名
    valid_extensions = ('.png',)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(input_dir, filename)
            try:
                with Image.open(image_path) as img:
                    # 仅对带 alpha 通道的图像进行处理（模式为 RGBA）
                    if img.mode == 'RGBA':
                        corrected_img = apply_gamma_correction_to_alpha(img, gamma)
                        output_path = os.path.join(output_dir, filename)
                        corrected_img.save(output_path)
                        print(f"已处理并保存：{filename}")
                    else:
                        print(f"跳过：{filename}（没有 alpha 通道）")
            except Exception as e:
                print(f"处理 {filename} 时出错：{e}")

def main(working_directory):
    # 设置输出目录为工作目录下的 GAMA-alpha 文件夹
    output_directory = os.path.join(working_directory, 'GAMA-alpha')
    process_images(working_directory, output_directory)
    print("所有图片处理完成！")

if __name__ == '__main__':
    # 修改为实际目录路径
    main(r'D:\DELL\Documents\myPlots\examples')
