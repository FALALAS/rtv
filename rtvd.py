import cv2
import numpy as np
import os
import time

start_time = time.time()


# 构建高斯金字塔
def build_gaussian_pyramid(frame, levels):
    pyramid = []
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    pyramid.append(gray_frame)
    for _ in range(levels):
        gray_frame = cv2.pyrDown(gray_frame)
        pyramid.append(gray_frame)
    return pyramid


def apply_motion_compensation_with_pyramid(current_frame, prev_frame, levels=3):
    if prev_frame is None:
        return build_gaussian_pyramid(current_frame, levels)

    pyramid_current = build_gaussian_pyramid(current_frame, levels)
    pyramid_prev = build_gaussian_pyramid(prev_frame, levels)

    # 从最低分辨率开始估计光流
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    current_level_flow = flow.calc(pyramid_prev[-1], pyramid_current[-1], None)

    # 在每一层上应用光流
    for i in range(levels - 1, -1, -1):
        h, w = pyramid_current[i].shape[:2]
        flow_map = np.column_stack((np.meshgrid(np.arange(w), np.arange(h))))
        new_coords = flow_map + cv2.resize(current_level_flow, (w, h), interpolation=cv2.INTER_LINEAR)
        pyramid_current[i] = cv2.remap(pyramid_current[i], new_coords, None, cv2.INTER_LINEAR)

        if i > 0:
            current_level_flow = cv2.pyrUp(current_level_flow)

    return pyramid_current


def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []

    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)

    # 添加最底层的高斯层
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def laplacian_pyramid_fusion(pyramid1, pyramid2):
    fused_pyramid = []

    # 融合每一层的拉普拉斯金字塔
    for p1, p2 in zip(pyramid1, pyramid2):
        fused = cv2.addWeighted(p1, 0.5, p2, 0.5, 0)
        fused_pyramid.append(fused)

    # 重建图像
    fused_frame = fused_pyramid[-1]
    for i in range(len(fused_pyramid) - 1, 0, -1):
        fused_frame = cv2.pyrUp(fused_frame)
        fused_frame = cv2.add(fused_frame, fused_pyramid[i - 1])

    return fused_frame


noised_folder = "noised000var100"  # 指定文件夹路径
output_folder = "rtvdvar100"
os.makedirs(output_folder, exist_ok=True)

prev_frame = None
prev_output_pyramid = None
levels = 3  # 根据需要调整层数

num_images = 100  # 假设有 100 帧

for i in range(num_images):
    filename = f'{i:08d}.png'
    noised_path = os.path.join(noised_folder, filename)
    current_frame = cv2.imread(noised_path)
    # 检查图片是否被成功加载
    if current_frame is None:
        print(f"无法读取图像文件 {filename}")
        continue

    current_frame_pyramid = build_gaussian_pyramid(current_frame, levels)

    if prev_frame is None:
        compensated_pyramid = apply_motion_compensation_with_pyramid(current_frame_gray, prev_frame, levels)

    if prev_output_pyramid is not None:
        compensated_laplacian_pyramid = build_laplacian_pyramid(compensated_pyramid)
        prev_laplacian_pyramid = build_laplacian_pyramid(prev_output_pyramid)
        fused_frame = laplacian_pyramid_fusion(compensated_laplacian_pyramid, prev_laplacian_pyramid)
    else:
        fused_frame = current_frame

    prev_frame = current_frame_gray
    prev_output_pyramid = build_gaussian_pyramid(fused_frame, levels)

    # 保存处理后的帧
    output_filename = f"{output_folder}/{i:08d}.png"
    cv2.imwrite(output_filename, fused_frame)
    current_time = time.time()  # 获取当前时间
    elapsed_time = current_time - start_time  # 计算经过的时间
    print(f"已处理到第 {i} 帧，用时 {elapsed_time:.2f} 秒")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
