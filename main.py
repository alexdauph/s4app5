import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from fir_filters import *
from synthesize import *

# # --- Import sound data
# data, Fs = sf.read('note_basson_plus_sinus_1000_Hz.wav')
#
# # --- Set filter parameters
# N = 1024
# length = data.size
# Fc = 20
#
# # --- Create band stop filter response
# h = fir_low_pass_filter(Fc, Fs, N)
# h = fir_band_stop_filter(1000, Fs, N, h)
#
# # --- Filter sound
# data_filtered = np.convolve(data, h)
# data_filtered = data_filtered[2000:134000]
#
# # --- Get envelop
# env = envelop(data_filtered)
#
# # --- Get FFT
# data_filtered_windowed = data_filtered * np.hanning(data_filtered.size)
# X = np.fft.fft(data_filtered_windowed)
# X = 2 * X[0:int(X.size/2)]
#
# # --- Find peaks
# peaks, properties = find_peaks(X, distance=150, prominence=5)
# print("peaks.size = " + str(peaks.size))
#
# # --- Synthesize new signal
# Z = synthesize(X, peaks, 32)
#
# # --- Graph result
# plt.subplot(2, 1, 1)
# plt.plot(20 * np.log10(abs(X)))
# plt.plot(peaks, 20 * np.log10(X[peaks]), "x")
#
# Z = Z * env[0:Z.size]
# plt.subplot(2, 1, 2)
# plt.plot(Z)
#
# plt.show()
#
# sf.write("note_basson_filtre.wav", data=data_filtered.real, samplerate=Fs)
# sf.write("note_basson_synthetise.wav", data=Z.real, samplerate=Fs)






# data, Fs = sf.read('note_guitare_LAd.wav')
# print("data.size = " + str(data.size))
#
# data = abs(data)
# plt.subplot(4, 1, 1)
# plt.plot(data)
#
# # data_fft = np.fft.fft(data)
# # plt.subplot(4, 1, 2)
# # plt.plot(data_fft)
#
# env = envelop(data)
# plt.subplot(4, 1, 2)
# plt.plot(env)
#
# # filtered = np.convolve(data, env)
# # plt.subplot(4, 1, 3)
# # plt.plot(abs(filtered))
#
# # conv = np.convolve(env, data_fft)
# # plt.subplot(4, 1, 4)
# # plt.plot(conv)
#
# plt.show()
#
