import numpy as np
from PIL import Image


def circle_crop(square_array):
    N = square_array.shape[0]  # 假设数组是方形的
    center = (N // 2, N // 2)  # 圆心为数组的中心
    radius = N // 2  # 半径设为数组的一半

    # 创建一个坐标网格
    Y, X = np.ogrid[:N, :N]  # 生成坐标网格
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)  # 计算每个点到圆心的距离

    # 创建掩码，圆内的点为1，圆外的点为0
    mask = dist_from_center <= radius

    # 通过掩码裁剪数组，圆形区域内的元素保留，其他设置为0
    result = square_array * mask  # 直接应用掩码

    return result


def compute_circular_average(image_array):
    # 假设image_array是一个方形的NumPy数组
    # 我们将其裁剪为一个圆形
    cropped_array = circle_crop(image_array)

    # 计算圆形区域内的平均值
    average_value = np.mean(cropped_array[cropped_array != 0])

    return average_value


def load_img(img_path):
    # Open the image using PIL
    img = Image.open(img_path)

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Normalize the array to the range [0, 1]
    img_array = img_array / 255.0

    return img_array


def load_2D_data(
        data_filename,
        data_type,
        work_dir='./data',
        crop_rate=1.0
):
    if data_type == '.npy':
        data_2D = np.load(f'{work_dir}/{data_filename}' + data_type)
    else:
        data_2D = load_img(f'{work_dir}/{data_filename}' + data_type)[:, :, 0]
    if 1 > crop_rate > 0:
        margin = int((1 - crop_rate) / 2 * data_2D.shape[0])
        data_2D = data_2D[margin:-margin, margin:-margin]
    return data_2D


def clear_ax_ticks(ax):
    # 去除坐标轴标签和刻度
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # 去除边框
    for spine in ax.spines.values():
        spine.set_visible(False)
