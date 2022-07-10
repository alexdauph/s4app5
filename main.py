import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from fir_filters import *
from synthesize import *

# --- Import sound data
note_guitare_LA, fs = sf.read('note_guitare_LAd.wav')
note_basson_and_sine, fs = sf.read('note_basson_plus_sinus_1000_Hz.wav')

# --- Create band stop filter response
low_pass_filter = fir_low_pass_filter(20, fs, 1024)
band_stop_filter = fir_band_stop_filter(1000, fs, 1024, low_pass_filter)

# --- Filter basson
note_basson = np.convolve(note_basson_and_sine, band_stop_filter)
note_basson = note_basson[2000:134000]
# ######################
# plt.subplot(2, 1, 1)
# plt.plot(note_guitare_LA)
# plt.subplot(2, 1, 2)
# plt.plot(note_basson)
# plt.show()
# ######################

# --- Get envelop
env_note_LA = envelop(note_guitare_LA)
env_note_basson = envelop(note_basson)
# ######################
# plt.subplot(2, 1, 1)
# plt.plot(env_note_LA)
# plt.subplot(2, 1, 2)
# plt.plot(env_note_basson)
# plt.show()
# ######################

# --- Apply window
note_guitare_LA = note_guitare_LA * np.hanning(note_guitare_LA.size)
note_basson = note_basson * np.hanning(note_basson.size)

# --- Get fft
fft_note_LA = np.fft.fft(note_guitare_LA)
fft_note_basson = np.fft.fft(note_basson)
# ######################
# plt.subplot(2, 1, 1)
# plt.plot(20 * np.log10(fft_note_LA))
# plt.subplot(2, 1, 2)
# plt.plot(20 * np.log10(fft_note_basson))
# plt.show()
# ######################

# --- Get parameters
params_note_LA = extract_params(fft_note_LA, fs, 1000, 1)
params_note_basson = extract_params(fft_note_basson, fs, 400, 3)
# ######################
# peaks_note_LA, properties_note_LA = find_peaks(fft_note_LA, distance=1000, prominence=1)
# peaks_note_basson, properties_note_basson = find_peaks(fft_note_basson, distance=400, prominence=3)
# plt.subplot(2, 1, 1)
# plt.plot(20 * np.log10(abs(fft_note_LA)))
# plt.plot(peaks_note_LA, 20 * np.log10(fft_note_LA[peaks_note_LA]), "x")
# plt.subplot(2, 1, 2)
# plt.plot(20 * np.log10(abs(fft_note_basson)))
# plt.plot(peaks_note_basson, 20 * np.log10(fft_note_basson[peaks_note_basson]), "x")
# plt.show()
# ######################

# --- Generate other notes
params_note_RE = change_frequency(params_note_LA, 0.667)
params_note_MI = change_frequency(params_note_LA, 0.749)
params_note_FA = change_frequency(params_note_LA, 0.794)
params_note_SOL = change_frequency(params_note_LA, 0.891)

# --- Synthesize signal
synth_note_LA = synthesize(params_note_LA, fft_note_LA.size, fs)
synth_note_RE = synthesize(params_note_RE, fft_note_LA.size, fs)
synth_note_MI = synthesize(params_note_MI, fft_note_LA.size, fs)
synth_note_FA = synthesize(params_note_FA, fft_note_LA.size, fs)
synth_note_SOL = synthesize(params_note_SOL, fft_note_LA.size, fs)
synth_note_basson = synthesize(params_note_basson, fft_note_basson.size, fs)
# ######################
# plt.subplot(2, 1, 1)
# plt.plot(synth_note_LA)
# plt.subplot(2, 1, 2)
# plt.plot(synth_note_basson)
# plt.show()
# ######################

# --- Multiply by envelop
synth_note_LA *= env_note_LA[0:synth_note_LA.size]
synth_note_RE *= env_note_LA[0:synth_note_RE.size]
synth_note_MI *= env_note_LA[0:synth_note_MI.size]
synth_note_FA *= env_note_LA[0:synth_note_FA.size]
synth_note_SOL *= env_note_LA[0:synth_note_SOL.size]
synth_note_basson *= env_note_basson[0:synth_note_basson.size]
######################
plt.subplot(2, 1, 1)
plt.plot(synth_note_LA)
plt.subplot(2, 1, 2)
plt.plot(synth_note_basson)
plt.show()
######################

# --- Generate Beethoven
beg = 6800
div = 10
bet = synth_note_SOL[beg:int(synth_note_SOL.size/div)]
bet = np.concatenate([bet, synth_note_SOL[beg:int(synth_note_SOL.size/div)]])
bet = np.concatenate([bet, synth_note_SOL[beg:int(synth_note_SOL.size/div)]])
bet = np.concatenate([bet, synth_note_MI[beg:int(synth_note_MI.size/1.25)]])
bet = np.concatenate([bet, synth_note_FA[beg:int(synth_note_FA.size/div)]])
bet = np.concatenate([bet, synth_note_FA[beg:int(synth_note_FA.size/div)]])
bet = np.concatenate([bet, synth_note_FA[beg:int(synth_note_FA.size/div)]])
bet = np.concatenate([bet, synth_note_RE[beg:int(synth_note_RE.size/1.25)]])

# --- Create audio files
sf.write("bet.wav", data=bet.real, samplerate=fs)
sf.write("synth_note_LA.wav", data=synth_note_LA.real, samplerate=fs)
sf.write("synth_note_RE.wav", data=synth_note_RE.real, samplerate=fs)
sf.write("synth_note_MI.wav", data=synth_note_MI.real, samplerate=fs)
sf.write("synth_note_FA.wav", data=synth_note_FA.real, samplerate=fs)
sf.write("synth_note_SOL.wav", data=synth_note_SOL.real, samplerate=fs)
sf.write("synth_note_basson.wav", data=synth_note_basson.real, samplerate=fs)

# --- Plot signals
# plt.subplot(2, 1, 1)
# plt.plot(synth_note_LA)
# plt.subplot(2, 1, 2)
# plt.plot(synth_note_basson)
# plt.show()
