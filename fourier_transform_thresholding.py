import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy as cp
import cv2


def thresholding_fft(A, thresh):
    cp_A = cp.copy(A)
    percent = np.percentile(np.abs(cp_A), thresh)
    cp_A[np.abs(cp_A) <= percent] = 0
    return cp_A


if __name__ == "__main__":
    # noinspection PyTypeChecker
    boat = np.asarray(Image.open('images/Boat.png').convert('RGB').convert('L'))
    # noinspection PyTypeChecker
    museum = np.asarray(Image.open('images/Museum.png').convert('L'))

    boat, museum = boat.astype(np.double), museum.astype(np.double)
    # boat_fft = np.fft.fft2(boat)
    # name = "boat"
    boat_fft = np.fft.fft2(museum)
    name = "museum"

    """boat_fft_10 = thresholding_fft(boat_fft, 10)
    boat_recovered_10 = np.abs(np.fft.ifft2(boat_fft_10)).astype(np.uint8)
    boat_fft_10 = Image.fromarray(boat_recovered_10)
    boat_fft_10.save(f'images/{name}_fft_10.png')

    boat_fft_50 = thresholding_fft(boat_fft, 50)
    boat_recovered_50 = np.abs(np.fft.ifft2(boat_fft_50)).astype(np.uint8)
    boat_fft_50 = Image.fromarray(boat_recovered_50)
    boat_fft_50.save(f'images/{name}_fft_50.png')

    boat_fft_90 = thresholding_fft(boat_fft, 90)
    boat_recovered_90 = np.abs(np.fft.ifft2(boat_fft_90)).astype(np.uint8)
    boat_fft_90 = Image.fromarray(boat_recovered_90)
    boat_fft_90.save(f'images/{name}_fft_90.png')

    boat_fft_95 = thresholding_fft(boat_fft, 95)
    boat_recovered_95 = np.abs(np.fft.ifft2(boat_fft_95)).astype(np.uint8)
    boat_fft_95 = Image.fromarray(boat_recovered_95)
    boat_fft_95.save(f'images/{name}_fft_95.png')"""

    boat_fft_99 = thresholding_fft(boat_fft, 99)
    boat_recovered_99 = np.abs(np.fft.ifft2(boat_fft_99)).astype(np.uint8)
    boat_fft_99 = Image.fromarray(boat_recovered_99)
    boat_fft_99.save(f'images/{name}_fft_99.png')

    # plt for testing purpose
    """plt.figure(figsize=(7, 7))
    plt.imshow(boat_recovered_90, cmap='gray')
    plt.title("90")
    plt.show()"""
