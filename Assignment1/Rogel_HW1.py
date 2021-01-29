# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy.io
from scipy.io import wavfile
import matplotlib.pyplot as plt

testa = np.array([1, 2, 3], dtype=float)
testb = np.array([0, 1, 1], dtype=float)


z=[]
def crossCorr(x,y):
    z = np.correlate(x,y, "same")
    return z


def loadSoundFile(filename):
    samplerate , x = scipy.io.wavfile.read(filename)
    x1 = x[:, 0]
    return x1


snare = []
drum_loop = []

snare = loadSoundFile('snare.wav')
drum_loop = loadSoundFile('drum_loop.wav')

correlation = crossCorr(snare,drum_loop)

snare_t=np.arange(snare.size)
drum_loop_t = np.arange(drum_loop.size)
correlation_t = np.arange(correlation.size)
f = plt.figure() 
f.set_figwidth(50) 
f.set_figheight(50) 
fig1 = plt.plot(correlation)
plt.savefig('correlation.png')
plt.show(fig1)

#plt.xticks(np.arange(min(correlation_t), max(correlation_t)+1, 5000.0))
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
print(correlation)

    #look for maxima
    #scipy find peaks
