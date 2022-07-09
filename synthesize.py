import numpy as np


def synthesize(fft, peaks, count):
    data = np.zeros(fft.size)
    for i in range(0, min(peaks.size, count)):
        data[peaks[i]] = fft[peaks[i]]
    return np.fft.irfft(data)


def envelop(data):
    N = 442
    K = 885

    h = np.ones(K) * 1 / N
    return np.convolve(data, h)

    # w = np.pi / 1000
    # #h = []
    # n = np.arange(0, 1, 0.0001)
    # h = np.arange(0, 1, 0.0001)
    # # n = np.arange(0, N)
    # # h = np.arange(0, N)
    #
    # h = 1/N * np.sin(n * K/2) / np.sin(n/2) * 1/2
    # h[0] = 1 + 1/K

    # for n in range(0, length, ):
    #     if n == 0:
    #         h.append(0)
    #     else:
    #         h.append(1/N * np.sin(n * K/2) / np.sin(n/2))
