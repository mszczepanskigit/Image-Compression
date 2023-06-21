import pywt
import numpy as np
from PIL import Image
import copy as cp
import matplotlib.pyplot as plt


def wt(matrix):
    cA, (cH, cV, cD) = pywt.dwt2(data=matrix, wavelet='haar')
    top_row = np.concatenate((cA, cH), axis=1)
    bottom_row = np.concatenate((cV, cD), axis=1)
    matrix_transformed = np.concatenate((top_row, bottom_row), axis=0)
    """old_min = np.min(matrix_transformed)
    old_max = np.max(matrix_transformed)
    new_min = 0
    new_max = 255
    matrix_transformed = ((matrix_transformed - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min"""
    return matrix_transformed


def normalizee(matrix):
    old_min = np.min(matrix)
    old_max = np.max(matrix)
    new_min = 0
    new_max = 255
    return ((matrix - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    # return matrix


def get_coeffs(matrix):
    M = cp.copy(matrix)
    cut = M.shape[0] // 2
    cA = M[:cut, : cut]
    cH = M[:cut, cut:]
    cV = M[cut:, :cut]
    cD = M[cut:, cut:]
    return cA, (cH, cV, cD)


if __name__ == "__main__":
    # noinspection PyTypeChecker
    boat = np.asarray(Image.open('images/Boat.png').convert('RGB').convert('L'))

    boat = boat.astype(np.double)

    boat_1 = wt(boat)

    boat_2_tmp = wt(boat_1[0:256, 0:256])
    boat_2 = cp.copy(boat_1)
    boat_2[0:256, 0:256] = boat_2_tmp

    boat_3_tmp = wt(boat_2[0:128, 0:128])
    boat_3 = cp.copy(boat_2)
    boat_3[0:128, 0:128] = boat_3_tmp

    boat_4_tmp = wt(boat_3[0:64, 0:64])
    boat_4 = cp.copy(boat_3)
    boat_4[0:64, 0:64] = boat_4_tmp

    boat_5_tmp = wt(boat_4[0:32, 0:32])
    boat_5 = cp.copy(boat_4)
    boat_5[0:32, 0:32] = boat_5_tmp

    boat_6_tmp = wt(boat_5[0:16, 0:16])
    boat_6 = cp.copy(boat_5)
    boat_6[0:16, 0:16] = boat_6_tmp

    boat_7_tmp = wt(boat_6[0:8, 0:8])
    boat_7 = cp.copy(boat_6)
    boat_7[0:8, 0:8] = boat_7_tmp

    boat_8_tmp = wt(boat_7[0:4, 0:4])
    boat_8 = cp.copy(boat_7)
    boat_8[0:4, 0:4] = boat_8_tmp

    boat_9_tmp = wt(boat_8[0:2, 0:2])
    boat_9 = cp.copy(boat_8)
    boat_9[0:2, 0:2] = boat_9_tmp

    # Thresholding
    thresh = 10
    boat_9[np.abs(boat_9) < thresh] = 0
    print(f"We have {np.count_nonzero(boat_9 == 0)} zeros out of {int(boat_9.shape[0] * boat_9.shape[1])}, "
          f"which is roughly ${np.round((np.count_nonzero(boat_9 == 0)/int(boat_9.shape[0] * boat_9.shape[1]))*100,2)}\%$.")

    aboat_10 = cp.copy(boat_9)
    aboat_tmp = aboat_10[0:2, 0:2]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='haar')
    aboat_10[0:2, 0:2] = tmp

    aboat_9 = cp.copy(aboat_10)
    aboat_tmp = aboat_9[0:4, 0:4]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='haar')
    aboat_9[0:4, 0:4] = tmp

    aboat_8 = cp.copy(aboat_9)
    aboat_tmp = aboat_8[0:8, 0:8]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='haar')
    aboat_8[0:8, 0:8] = tmp

    aboat_7 = cp.copy(aboat_8)
    aboat_tmp = aboat_7[0:16, 0:16]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='haar')
    aboat_7[0:16, 0:16] = tmp

    aboat_6 = cp.copy(aboat_7)
    aboat_tmp = aboat_6[0:32, 0:32]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='haar')
    aboat_6[0:32, 0:32] = tmp

    aboat_5 = cp.copy(aboat_6)
    aboat_tmp = aboat_5[0:64, 0:64]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='haar')
    aboat_5[0:64, 0:64] = tmp

    aboat_4 = cp.copy(aboat_5)
    aboat_tmp = aboat_4[0:128, 0:128]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='haar')
    aboat_4[0:128, 0:128] = tmp

    aboat_3 = cp.copy(aboat_4)
    aboat_tmp = aboat_3[0:256, 0:256]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='haar')
    aboat_3[0:256, 0:256] = tmp

    aboat_2 = cp.copy(aboat_3)
    aboat_tmp = aboat_2[0:512, 0:512]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='haar')
    aboat_2[0:512, 0:512] = tmp

    boat_haar = Image.fromarray(normalizee(aboat_2).astype(np.uint8))
    boat_haar.save(f'./haar_boat_thresh_{thresh}.png')

    """plt.figure(figsize=(7, 7))
    plt.imshow(aboat_2.astype(np.uint8), cmap='gray')
    plt.title("")
    plt.show()"""
