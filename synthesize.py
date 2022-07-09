import numpy as np


def synthesize(fft, peaks, count):
    data = np.zeros(fft.size)
    for i in range(0, min(peaks.size, count)):
        data[peaks[i]] = fft[peaks[i]]
    return np.fft.irfft(data)

def synthesize_v2(fft, peaks, fs):
    count = fft.size * 2

    peaks = peaks[0:min(peaks.size, 32)]

    freqs = (peaks / (fft.size)) * fs
    amps = abs(fft[peaks])
    angs = np.angle(fft[peaks])

    sine = np.zeros(count)
    t = np.arange(0, count/fs, 1/fs)
    for i in range(0, peaks.size):
        sine += amps[i] * np.cos(2 * np.pi * freqs[i] * t + angs[i])

    sine = normalize(sine)

    return sine


def envelop(data):
    N = 442
    K = 885

    h = np.ones(K) * 1 / N
    return np.convolve(abs(data), h)


def normalize(data):
    max_amp = data[0]

    for i in range(0, data.size):
        if data[i] > max_amp:
            max_amp = data[i]

    data /= max_amp
    return data
