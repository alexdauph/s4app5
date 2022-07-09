import numpy as np


def fir_low_pass_filter(fc, fs, order):
    w = (fc * 2 * np.pi) / fs
    k = int(w / np.pi * order + 1)
    low_pass_filter = []

    for n in range(0, order):
        if n == 0:
            low_pass_filter.append(k / order)
        else:
            low_pass_filter.append((np.sin(np.pi * n * k / order)) / (order * np.sin(np.pi * n / order)))

    return low_pass_filter


def fir_band_stop_filter(fc, fs, order, low_pass_filter):
    d = np.zeros(order)
    d[0] = 1.0
    band_stop_filter = []

    for n in range(0, order):
        band_stop_filter.append(d[n] - 2 * low_pass_filter[n] * np.cos(2 * np.pi * 1000/fs * n))

    return band_stop_filter



