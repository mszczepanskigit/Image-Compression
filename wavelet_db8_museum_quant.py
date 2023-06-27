import pywt
import numpy as np
from PIL import Image
import copy as cp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def wt(matrix):
    cA, (cH, cV, cD) = pywt.dwt2(data=matrix, wavelet='db8', mode='periodization')
    top_row = np.concatenate((cA, cH), axis=1)
    bottom_row = np.concatenate((cV, cD), axis=1)
    matrix_transformed = np.concatenate((top_row, bottom_row), axis=0)
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


def vector_quantization(coeffs, num_clusters):
    vectorized_coeffs = coeffs.reshape(-1, 1)

    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(vectorized_coeffs)

    quantized_coeffs = kmeans.cluster_centers_[kmeans.labels_].reshape(coeffs.shape)

    return quantized_coeffs


if __name__ == "__main__":
    # noinspection PyTypeChecker
    boat = np.asarray(Image.open('images/Museum.png').convert('L'))

    boat_squared = np.zeros((4096, 4096))
    boat_squared[:boat.shape[0], :boat.shape[1]] = boat

    boat = boat_squared.astype(np.double)

    boat_1 = wt(boat)

    boat_m1_tmp = wt(boat_1[0:2048, 0:2048])
    boat_m1 = cp.copy(boat_1)
    boat_m1[0:2048, 0:2048] = boat_m1_tmp

    boat_0_tmp = wt(boat_m1[0:1024, 0:1024])
    boat_0 = cp.copy(boat_m1)
    boat_0[0:1024, 0:1024] = boat_0_tmp

    boat_11_tmp = wt(boat_0[0:512, 0:512])
    boat_11 = cp.copy(boat_0)
    boat_11[0:512, 0:512] = boat_11_tmp

    boat_2_tmp = wt(boat_11[0:256, 0:256])
    boat_2 = cp.copy(boat_11)
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

    num_clusters = 8

    quantized_boat = vector_quantization(boat_9, num_clusters)

    """ordered_values = np.sort(boat_9.flatten())
    x = np.arange(len(ordered_values))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plot the ordered values in the first subplot
    axs[0].plot(x[150_000:16_000_000], ordered_values[150_000:16_000_000])
    axs[0].set_ylabel('Value')
    axs[0].set_title('Ordered Values of the transformed image')
    axs[0].grid(True)

    # Plot the ordered logarithm values in the second subplot
    axs[1].plot(x[150_000:16_000_000], np.log(np.abs(ordered_values[150_000:16_000_000])))
    axs[1].set_ylabel('log(Value)')
    axs[1].set_title('Log of Absolute Values of the transformed image')
    axs[1].grid(True)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the figure
    plt.show()"""

    aboat_10 = cp.copy(quantized_boat)
    aboat_tmp = aboat_10[0:2, 0:2]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_10[0:2, 0:2] = tmp

    aboat_9 = cp.copy(aboat_10)
    aboat_tmp = aboat_9[0:4, 0:4]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_9[0:4, 0:4] = tmp

    aboat_8 = cp.copy(aboat_9)
    aboat_tmp = aboat_8[0:8, 0:8]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_8[0:8, 0:8] = tmp

    aboat_7 = cp.copy(aboat_8)
    aboat_tmp = aboat_7[0:16, 0:16]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_7[0:16, 0:16] = tmp

    aboat_6 = cp.copy(aboat_7)
    aboat_tmp = aboat_6[0:32, 0:32]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_6[0:32, 0:32] = tmp

    aboat_5 = cp.copy(aboat_6)
    aboat_tmp = aboat_5[0:64, 0:64]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_5[0:64, 0:64] = tmp

    aboat_4 = cp.copy(aboat_5)
    aboat_tmp = aboat_4[0:128, 0:128]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_4[0:128, 0:128] = tmp

    aboat_3 = cp.copy(aboat_4)
    aboat_tmp = aboat_3[0:256, 0:256]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_3[0:256, 0:256] = tmp

    aboat_2 = cp.copy(aboat_3)
    aboat_tmp = aboat_2[0:512, 0:512]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_2[0:512, 0:512] = tmp

    aboat_1 = cp.copy(aboat_2)
    aboat_tmp = aboat_1[0:1024, 0:1024]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_1[0:1024, 0:1024] = tmp

    aboat_0 = cp.copy(aboat_1)
    aboat_tmp = aboat_0[0:2048, 0:2048]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_0[0:2048, 0:2048] = tmp

    aboat_m1 = cp.copy(aboat_0)
    aboat_tmp = aboat_m1[0:4096, 0:4096]
    tmp = pywt.idwt2(get_coeffs(aboat_tmp), wavelet='db8', mode='periodization')
    aboat_m1[0:4096, 0:4096] = tmp

    boat_haar = Image.fromarray(normalizee(aboat_m1[:1731, :2400]).astype(np.uint8))
    boat_haar.save(f'./db8_museum_quant_{num_clusters}.png')

    plt.figure(figsize=(7, 7))
    plt.imshow(normalizee(aboat_m1[:1731, :2400]).astype(np.uint8), cmap='gray')
    plt.title("")
    plt.show()
