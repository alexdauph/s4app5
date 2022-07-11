import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def extract_params(data, fs, dist, prom):
    peaks, properties = find_peaks(data, distance=dist, prominence=prom)
    count = min(peaks.size, 32)
    peaks = peaks[0:count]
    params = [[0 for i in range(3)] for j in range(count)]

    for i in range(0, count):
        params[i][0] = (peaks[i] / data.size) * fs
        params[i][1] = abs(data[peaks[i]])
        params[i][2] = np.angle(data[peaks[i]])

    return params


def synthesize(params, size, fs):
    signal = np.zeros(size)
    t = np.arange(0, size/fs, 1/fs)

    for i in range(0, len(params)):
        signal += params[i][1] * np.cos(2 * np.pi * params[i][0] * t + params[i][2])

    return normalize(signal)


def envelop(data):
    N = 442
    K = 2 * N + 1

    h = np.ones(K) * 1 / N
    return np.convolve(abs(data), h)


def normalize(data):
    max_amp = data[0]

    for i in range(0, data.size):
        if data[i] > max_amp:
            max_amp = data[i]

    data /= max_amp
    return data


def change_frequency(params, value):
    new_params = [[0 for i in range(3)] for j in range(len(params))]

    for i in range(0, len(params)):
        new_params[i][0] = params[i][0] * value
        new_params[i][1] = params[i][1]
        new_params[i][2] = params[i][2]

    return new_params


def freq_response():
    N = 442
    K = 2 * N + 1

    w = np.arange(0, 0.04, 0.00001)
    #w = 2 * np.pi * m / N * 44100

    X = 1/K * np.sin(w * K/2) / np.sin(w / 2)
    X[0] = 1 + 1/K
    plt.plot(w, 20 * np.log10(abs(X)), color="black")
    plt.xlim(0, 0.04)
    plt.xlabel("|H(w)| (rad/éch)")
    plt.ylabel("Gain (dB)")
    plt.title("Réponse en fréquence de la moyenne mobile")

    plt.plot(np.pi/1000, -3, marker='o', color='black')
    plt.text(np.pi/1000+0.0003, -3, "(pi/1000, -3)")

    plt.axhline(y=-13, color='gray', linestyle='--', linewidth=1.0)
    for i in range(1, 6):
        plt.axvline(x=i*(2*np.pi/K), color='gray', linestyle='--', linewidth=1.0)

    plt.show()

