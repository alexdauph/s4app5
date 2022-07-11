import numpy as np
import matplotlib.pyplot as plt


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


def fir_1000hz_response(fff):
    f = 1000
    nb_periods = 3

    t = np.arange(0, nb_periods / f, 1 / f / 10000)
    x = np.arange(0, t.size)/t.size * nb_periods * 1/f
    wave = np.sin(2 * np.pi * 1000 * t)

    conv = np.convolve(wave, np.arange(0, 10))

    plt.subplot(3, 1, 1)
    plt.plot(x, wave)
    plt.subplot(3, 1, 2)
    plt.plot(fff)
    plt.subplot(3, 1, 3)
    plt.plot(x, conv[0:x.size])
    plt.show()

    return wave


