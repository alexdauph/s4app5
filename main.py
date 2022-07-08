# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


x, Fs = sf.read('note_basson_plus_sinus_1000_Hz.wav')

N = 1024
nbEch = x.size
Fc = 20
w = (Fc * 2 * np.pi) / Fs
K = int(w / np.pi * N + 1)
print("w = " + str(w))
print("K = " + str(K))

#K = (Fc / Fs) * N * 2 + 1


n = np.arange(0, N)
print("n = " + str(n))

h = []
for n in range(0, N):
    if n == 0:
        h.append(K/N)
    else:
        h.append((np.sin(np.pi * n * K / N)) / (N * np.sin(np.pi * n / N)))

d = np.zeros(N)
d[0] = 1.0

h2 = []
for n in range(0, N):
    h2.append(d[n] - 2 * h[n] * np.cos(2*np.pi*1000/Fs * n))

plt.plot(h2)
plt.show()

# h2 = []
# d = np.zeros(N)
# d[0] = 1.0

#
conv = np.convolve(x, h2)
plt.plot(conv)
plt.show()


#scipy.find_peak

# print("h = " + str(h))
# print("h2 = " + str(h2))
# plt.subplot(3,1,1)
# plt.plot(h)
# plt.subplot(3,1,2)
# plt.plot(h2)
# plt.subplot(3,1,3)
# plt.plot(conv)
# plt.show()
# #
#
# print("Fs = " + str(Fs))
# print("Fc = " + str(Fc))
# print("w = " + str(w))
# print("K = " + str(K))
#
# print("h = " + str(h))

### Transformer en filtre coupe-bande
# d = np.zeros(nbEch)
# d[0] = 1
#
# for i in range(0, nbEch - 1):
#     h[i] = d[i] - 2 * h[i] * np.cos(1000 * n[i])



# z = np.convolve(x, h)
#
# sf.write("test.wav", data=z.real, samplerate=Fs)



#plt.plot(np.convolve(x, h))
#plt.show()

# print(nbEch)
# print(K)



# def envelop(data):
#
#
#     data = abs(data)
#     return data
# #
# #
# #
# x, Fs = sf.read('note_guitare_LAd.wav')
# env = envelop(x)
#
#
# w = np.hanning(x.size)
# xw = x * w
#
# #plt.plot(xw)
#
# Xw = np.fft.fft(xw)
#
# plt.plot(20*np.log10(abs(Xw)))
# plt.show()


# #FFT =
#
# y = x[8000:8332]
#
# plt.plot(y)
# plt.show()
#
# y = np.pad(y, (0, 3200), 'constant')
#
# y = np.fft.fft(y)
# length = y.size
#
# y = y[0 : int(length/2)]
# y = y*2
#
# print(y)

# z = np.fft.ifft(y)
#
# plt.plot(y)
# plt.show()
# sf.write("test.wav", data = z.real, samplerate = Fs)


'''
def func(x, gain = 1.0):
    y = gain * x
    return y

x = np.asarray([1, 2, 3, 4, 5], dtype = np.float32)
z = func(x[2], 2.0)
print(x)
print(z)

np.save('vector.npy', x)
a = np.load('vector.npy')
print(a)
'''

'''
# Import .wav file
import soundfile as sf
import matplotlib.pyplot as plt

x, Fs = sf.read('note_basson_plus_sinus_1000_Hz.wav')
print(x.shape)
print(Fs)
plt.plot(x)
plt.show()
sf.write("test.wav", data = x, samplerate = Fs)
'''

'''
x = np.asarray([0, 1, 2, 1], dtype = np.float32)
X1 = np.fft.fft(x)
X2 = np.fft.rfft(x) # eviter de generer des valeurs inutiles qui ne serviront pas dans notre analyse

x1 = np.fft.ifft(X1)
x2 = np.fft.irfft(X1) # ce qui nous interesse reellement

print(x)
print("X1 = " + str(X1))
print(X2)
print(x1)
print(x2)
'''


# x = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype = np.float32)
# h = np.asarray([0, 1, -3], dtype = np.float32)
#
# y = np.convolve(x, h)
# plt.plot(y)
# plt.show()
#
#
# print(y)



