import cv2
import scipy.signal as signal
import numpy as np


# 这个函数将应用IIR低通滤波器到图像上
# 使用OpenCV读取和保存彩色图像的函数，不转换为灰度图像
def apply_iir_filter(image_path, order=2, cutoff_frequency=0.1):
    # 使用OpenCV读取图像（彩色）
    image = cv2.imread(image_path)

    # 设计巴特沃斯低通滤波器
    b, a = signal.butter(order, cutoff_frequency)

    # 初始化处理后的图像数组
    filtered_image_array = np.zeros_like(image)

    # 对每个颜色通道分别应用滤波器
    for channel in range(3):
        filtered_image_array[:, :, channel] = signal.filtfilt(b, a, image[:, :, channel], axis=0)
        filtered_image_array[:, :, channel] = signal.filtfilt(b, a, filtered_image_array[:, :, channel], axis=1)

    # 将处理后的图像保存为新文件
    filtered_image_path = "filtered_color_image.jpg"
    cv2.imwrite(filtered_image_path, filtered_image_array)

    return filtered_image_path


# 你可以像这样调用这个函数：
# filtered_image_path = apply_iir_filter_color("path_to_noisy_image.jpg")


filtered_image = apply_iir_filter("./noised000var100/00000000.png")
