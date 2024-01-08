import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def build_gaussian_pyramid(image, num_levels):
    gaussian_pyramid = [image]
    for i in range(1, num_levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid


def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    num_levels = len(gaussian_pyramid)
    for i in range(num_levels - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def save_pyramid_images(pyramid, base_name, directory=""):
    for i, image in enumerate(pyramid):
        filename = f"{base_name}_level_{i + 1}.png"
        filepath = directory + filename
        cv2.imwrite(filepath, image)


clean_folder = '000'
clean_path = os.path.join(clean_folder, '00000000.png')
# Sample image
sample_image = cv2.imread(clean_path)

# Build Gaussian and Laplacian pyramids
num_levels = 5
gaussian_pyramid = build_gaussian_pyramid(sample_image, num_levels)
laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)

save_pyramid_images(gaussian_pyramid, "gaussian")
save_pyramid_images(laplacian_pyramid, "laplacian")
