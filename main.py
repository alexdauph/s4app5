# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from fir_filters import *
from synthesize import *

# --- Import sound data
data, Fs = sf.read('note_basson_plus_sinus_1000_Hz.wav')
print("data2.size = " + str(data.size))

# --- Set filter parameters
N = 1024
length = data.size
Fc = 20

# --- Create band stop filter response
h = fir_low_pass_filter(Fc, Fs, N)
h = fir_band_stop_filter(1000, Fs, N, h)

# --- Filter sound
data_filtered = np.convolve(data, h)

# --- Get FFT
X = np.fft.fft(data_filtered)
X = 2 * X[0:int(X.size/2)]

# --- Find peaks
peaks, properties = find_peaks(X, distance=150, prominence=10)

# --- Synthesize new signal
Z = synthesize(X, peaks, 32)

# --- Graph result
plt.subplot(2, 1, 1)
plt.plot(20 * np.log10(abs(X)))
plt.plot(peaks, 20 * np.log10(X[peaks]), "x")
# Z *= np.hanning(Z.size)
plt.subplot(2, 1, 2)
plt.plot(Z)

plt.show()

sf.write("note_basson_filtre.wav", data=data_filtered.real, samplerate=Fs)
sf.write("test.wav", data=Z.real, samplerate=Fs)


