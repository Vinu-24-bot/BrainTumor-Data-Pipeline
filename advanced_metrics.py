import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def calculate_ssim(image1, image2):
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    if image1.dtype != np.uint8:
        imax = max(float(image1.max()), 1e-6)
        image1 = (image1 / imax * 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        imax = max(float(image2.max()), 1e-6)
        image2 = (image2 / imax * 255).astype(np.uint8)
    ssim_value, _ = ssim(image1, image2, full=True)
    return ssim_value


def butterworth_filter(image, cutoff_frequency=30, order=2, high_pass=True):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u, v, indexing='ij')
    u = u - crow
    v = v - ccol
    D = np.sqrt(u**2 + v**2)
    if high_pass:
        H = 1 / (1 + (cutoff_frequency / (D + 1e-8)) ** (2 * order))
    else:
        H = 1 / (1 + (D / (cutoff_frequency + 1e-8)) ** (2 * order))

    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    F_filtered = F_shifted * H
    F_filtered_shifted = np.fft.ifftshift(F_filtered)
    filtered_image = np.abs(np.fft.ifft2(F_filtered_shifted))
    denom = max(filtered_image.max() - filtered_image.min(), 1e-6)
    filtered_image = (filtered_image - filtered_image.min()) / denom * 255
    return filtered_image.astype(np.uint8)


def calculate_bf_score(segmented_image, ground_truth=None, cutoff_frequency=30, order=2):
    if segmented_image.dtype != np.uint8:
        segmented_image = (segmented_image > 0).astype(np.uint8) * 255

    edges_x = cv2.Sobel(segmented_image)
