import numpy as np


def synthesize(fft, peaks, count):
    data = np.zeros(fft.size)
    for i in range(0, min(peaks.size, count)):
        data[peaks[i]] = fft[peaks[i]]
    return np.fft.irfft(data)
