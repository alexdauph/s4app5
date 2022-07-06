# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def envelop(data):
    data = abs(data)
    return data

x, Fs = sf.read('note_guitare_LAd.wav')
x = envelop(x)
plt.plot(x)
plt.show()



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

'''
x = np.asarray([0, 1, 2, 1], dtype = np.float32)
h = np.asarray([3, -1, 2], dtype = np.float32)

y = np.convolve(x, h)

print(y)
'''


