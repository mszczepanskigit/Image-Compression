import numpy as np
from PIL import Image
import copy as cp
import matplotlib.pyplot as plt


def first_boat(A):
    cp_A = cp.copy(A)
    return 4 * np.floor(cp_A / 4)


def second_boat(A):
    cp_A = cp.copy(A)
    return 64 * np.floor(cp_A / 64)


def third_boat(A):
    cp_A = cp.copy(A)
    return 32 * (np.floor(cp_A / 64) + np.ceil(cp_A / 64))


def fourth_boat(A):
    cp_A = cp.copy(A)
    cp_A[cp_A < 64] = 32
    cp_A[np.logical_and(63 < cp_A, cp_A < 96)] = 80
    cp_A[np.logical_and(95 < cp_A, cp_A < 112)] = 104
    cp_A[np.logical_and(111 < cp_A, cp_A < 120)] = 116
    cp_A[np.logical_and(119 < cp_A, cp_A < 124)] = 122
    return cp_A


if __name__ == "__main__":
    # noinspection PyTypeChecker
    # boat = np.asarray(Image.open('images/Boat.png').convert('RGB').convert('L'))
    # noinspection PyTypeChecker
    boat = np.asarray(Image.open('images/Museum.png').convert('L'))

    # name = "boat"
    name = "museum"

    boat = boat.astype(np.double)

    boat_quant_1 = first_boat(boat).astype(np.uint8)
    boat_quant_2 = second_boat(boat).astype(np.uint8)
    boat_quant_3 = third_boat(boat).astype(np.uint8)
    boat_quant_4 = fourth_boat(boat).astype(np.uint8)

    boat_quant_1 = Image.fromarray(boat_quant_1)
    boat_quant_1.save(f'images/{name}_quant_1.png')

    boat_quant_2 = Image.fromarray(boat_quant_2)
    boat_quant_2.save(f'images/{name}_quant_2.png')

    boat_quant_3 = Image.fromarray(boat_quant_3)
    boat_quant_3.save(f'images/{name}_quant_3.png')

    boat_quant_4 = Image.fromarray(boat_quant_4)
    boat_quant_4.save(f'images/{name}_quant_4.png')
