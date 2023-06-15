import pywt
import numpy as np
from PIL import Image
import copy as cp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # noinspection PyTypeChecker
    boat = np.asarray(Image.open('images/Boat.png').convert('RGB').convert('L'))
    # noinspection PyTypeChecker
    museum = np.asarray(Image.open('images/Museum.png').convert('L'))

    boat, museum = boat.astype(np.double), museum.astype(np.double)

    cA, (cH, cV, cD) = pywt.dwt2(data=museum, wavelet='db2')

    #cH[:, :] = 0
    cV[:, :] = 0
    cD[:, :] = 0

    boat_haar_1 = pywt.idwt2((cA, (cH, cV, cD)), wavelet='db2').astype(np.uint8)

    boat_haar_1 = Image.fromarray(cA.astype(np.uint8))
    boat_haar_1.save(f'./cA.png')

    plt.figure(figsize=(7, 7))
    plt.imshow(cH, cmap='gray')
    plt.title("db8")
    plt.show()
