import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from fir_filters import *
from synthesize import *


# --- Import sound data
note_guitare_LA, fs = sf.read('note_guitare_LAd.wav')
note_basson_and_sine, fs = sf.read('note_basson_plus_sinus_1000_Hz.wav')
# # ######################
# plt.subplot(2, 1, 1)
# plt.title("Note LA#")
# plt.xlabel("Temps (s)")
# plt.ylabel("Amplitude")
# plt.plot(1/fs * np.arange(0, note_guitare_LA.size), note_guitare_LA)
# plt.subplot(2, 1, 2)
# plt.title("Note basson")
# plt.xlabel("Temps (s)")
# plt.ylabel("Amplitude")
# plt.plot(1/fs * np.arange(0, note_basson_and_sine.size), note_basson_and_sine)
# plt.show()
# # ######################

# --- Create band stop filter response
low_pass_filter = fir_low_pass_filter(20, fs, 1024)
band_stop_filter = fir_band_stop_filter(1000, fs, 1024, low_pass_filter)
######################
resp_x = np.arange(0, 1024) / 1024 * 44100
resp_freq = abs(np.fft.fft(band_stop_filter))
resp_angle = np.angle(np.fft.fft(band_stop_filter))
plt.subplot(2, 1, 1)
plt.title("Amplitude de la réponse en fréquence")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.plot(resp_x, 20*np.log10(resp_freq))

plt.subplot(2, 1, 2)
plt.title("Phase de la réponse en fréquence")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Phase (rad)")
plt.plot(resp_x, resp_angle)
plt.show()
######################

# --- Filter basson
note_basson = np.convolve(note_basson_and_sine, band_stop_filter)
note_basson = np.convolve(note_basson_and_sine, band_stop_filter)
note_basson = note_basson[2000:134000]
sf.write("note_basson_filtre.wav", data=note_basson.real, samplerate=fs)
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
note_guitare_LA *= np.hanning(note_guitare_LA.size)
note_basson *= np.hanning(note_basson.size)

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
params_note_basson = extract_params(fft_note_basson, fs, 400, 2.8)
######################
peaks_note_LA, properties_note_LA = find_peaks(fft_note_LA, distance=1000, prominence=1)
peaks_note_basson, properties_note_basson = find_peaks(fft_note_basson, distance=400, prominence=2.8)
peaks_note_LA = peaks_note_LA[0:32]
peaks_note_basson = peaks_note_basson[0:32]
plt.subplot(2, 1, 1)
plt.title("FFT note LA#")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Gain (dB)")
plt.plot(fs/fft_note_LA.size*np.arange(0, fft_note_LA.size), 20 * np.log10(abs(fft_note_LA)))
plt.plot(fs/fft_note_LA.size*peaks_note_LA, 20 * np.log10(fft_note_LA[peaks_note_LA]), "x", color="orange")
plt.subplot(2, 1, 2)
plt.title("FFT note basson")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Gain (dB)")
plt.plot(fs/fft_note_basson.size*np.arange(0, fft_note_basson.size), 20 * np.log10(abs(fft_note_basson)))
plt.plot(fs/fft_note_basson.size*peaks_note_basson, 20 * np.log10(fft_note_basson[peaks_note_basson]), "x", color="orange")
plt.show()
######################

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
# plt.title("Note LA# synthétisée sans enveloppe")
# plt.xlabel("Temps (s)")
# plt.ylabel("Amplitude")
# plt.plot(1/fs * np.arange(0, synth_note_LA.size), synth_note_LA)
# plt.subplot(2, 1, 2)
# plt.title("Note basson synthétisée sans enveloppe")
# plt.xlabel("Temps (s)")
# plt.ylabel("Amplitude")
# plt.plot(1/fs * np.arange(0, synth_note_basson.size), synth_note_basson)
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
plt.title("Note LA# synthétisée avec enveloppe")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.plot(1/fs * np.arange(0, synth_note_LA.size), synth_note_LA)
plt.subplot(2, 1, 2)
plt.title("Note basson synthétisée avec enveloppe")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.plot(1/fs * np.arange(0, synth_note_basson.size), synth_note_basson)
plt.show()
######################

# --- Generate Beethoven
beg = 8250
div = 9
beethoven = synth_note_SOL[beg:int(synth_note_SOL.size/div)]
beethoven = np.concatenate([beethoven, synth_note_SOL[beg:int(synth_note_SOL.size/div)]])
beethoven = np.concatenate([beethoven, synth_note_SOL[beg:int(synth_note_SOL.size/div)]])
beethoven = np.concatenate([beethoven, synth_note_MI[beg:int(synth_note_MI.size/1.25)]])
beethoven = np.concatenate([beethoven, synth_note_FA[beg:int(synth_note_FA.size/div)]])
beethoven = np.concatenate([beethoven, synth_note_FA[beg:int(synth_note_FA.size/div)]])
beethoven = np.concatenate([beethoven, synth_note_FA[beg:int(synth_note_FA.size/div)]])
beethoven = np.concatenate([beethoven, synth_note_RE[beg:int(synth_note_RE.size/1.25)]])
# ######################
# plt.plot(beethoven)
# plt.show()
# ######################

# --- Create audio files
sf.write("beethoven.wav", data=beethoven.real, samplerate=fs)
sf.write("synth_note_LA.wav", data=synth_note_LA.real, samplerate=fs)
sf.write("synth_note_RE.wav", data=synth_note_RE.real, samplerate=fs)
sf.write("synth_note_MI.wav", data=synth_note_MI.real, samplerate=fs)
sf.write("synth_note_FA.wav", data=synth_note_FA.real, samplerate=fs)
sf.write("synth_note_SOL.wav", data=synth_note_SOL.real, samplerate=fs)
sf.write("synth_note_basson.wav", data=synth_note_basson.real, samplerate=fs)

#freq_response()
