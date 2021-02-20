"""
Created on Sat Feb 13 20:31:58 2021

@author: Amit Rogel
"""

#Assignment 2
#Amit Rogel
import numpy as np
import scipy.io
from scipy.io import wavfile
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import time

def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    Samplenums = int(sampling_rate_Hz*length_secs)
    t = np.linspace(0.0,length_secs,num = Samplenums)
    x = amplitude*np.sin(frequency_Hz*t+phase_radians)
    return x,t

amplitude = 1.0
sampling_rate_Hz = 44100
frequency_Hz = 400 #Hz
length_secs = 0.5 #seconds
phase_radians = np.pi/2 #radians
(xsin,t) = generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians)


plotsample = int(0.005*sampling_rate_Hz)

xplot = xsin[0:plotsample]
tplot = t[0:plotsample]
xplot = xsin[0:plotsample]
tplot = t[0:plotsample]

fig = plt.figure()
plt.plot(tplot,xplot)
fig.suptitle('Sin Wave 400 Hz')
plt.xlabel('Amplitude')
plt.ylabel('Time (s)')
fig.savefig('Q1.jpg')

def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    Samplenums = int(sampling_rate_Hz*length_secs)
    t = np.linspace(0.0,length_secs,num = Samplenums)
    xsquare = np.array([])
    f = frequency_Hz/(2*np.pi)
    for j in t:
        square = np.array([0])
        for i in range(10):
            k = i+1
            squarecurrent = (4/np.pi)*(np.sin(2*np.pi*(2*k-1)*f*j)/(2*k-1))
            square = np.append(square,squarecurrent)
        xsquare =  np.append(xsquare,np.sum(square))
    return xsquare,t

phase_radians = 0

(x,t) = generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians)


plotsample = int(0.005*sampling_rate_Hz)

xplot = x[0:plotsample]
tplot = t[0:plotsample]
fig = plt.figure()
plt.plot(tplot,xplot)
fig.suptitle('Square Wave 400 Hz')
plt.xlabel('Amplitude')
plt.ylabel('Time (s)')
fig.savefig('Q2.jpg')


def computeSpectrum(x, sample_rate_Hz):
    
    xfft= np.fft.fft(x)
    bins = len(xfft)
    freq = np.linspace(0,(bins-1))
    f = sample_rate_Hz*freq/bins
    xabs = np.abs(xfft)
    xphase = np.angle(xfft)
    xre = xfft.real
    xim = xfft.imag
    return f,xabs,xphase,xre,xim

(f,xabs,xphase,xre,xim) = computeSpectrum(x, sampling_rate_Hz)

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Square Wave 400 Hz FFT')
ax1.plot(xabs)
ax1.set_title('Magnitude and Phase Graph')
ax1.set(ylabel = 'Magnitude')
ax2.plot(xphase)
ax2.set(xlabel = 'Frequency (Hz)', ylabel = 'Phase')
fig.savefig('Q3_square.jpg')

(f,xabs,xphase,xre,xim) = computeSpectrum(xsin, sampling_rate_Hz)

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Sin Wave 400 Hz FFT')
ax1.plot(xabs)
ax1.set_title('Magnitude and Phase Graph')
ax1.set(ylabel = 'Magnitude')
ax2.plot(xphase)
ax2.set(xlabel = 'Frequency (Hz)', ylabel = 'Phase')
fig.savefig('Q3_sin.jpg')


def generateBlocks(x, sample_rate_Hz, block_size, hop_size):
    lx = len(x);
    numHop = int(np.ceil(lx/hop_size))
    tTotal = sample_rate_Hz*lx
    t = np.linspace(0,tTotal,num=numHop)
    X=np.empty([int(block_size),numHop])
    for i in range(numHop):
        block = x[(i*hop_size):(i*hop_size+block_size)]
        if len(block) < block_size:
            block = np.append(block,np.zeros(block_size-len(block)))
        X[:,i] = block
    return t,X



def mySpecgram(x,  block_size, hop_size, sampling_rate_Hz, window_type):
    lx = len(x)
    if window_type == 'hann':
        han = np.hanning(lx)
        x = x*han
    (time_vector,X) = generateBlocks(x, sampling_rate_Hz, block_size, hop_size)
    (freq_vector,magnitude_spectrogram,xphase,xre,xim) = computeSpectrum(X, sampling_rate_Hz)
    fig = plt.figure()
    plt.specgram(x,NFFT=block_size,Fs = sampling_rate_Hz, noverlap = (block_size-hop_size) )
    fig.suptitle('Spectrogram of 400 Hz Square Wave with {} Windowing' .format(window_type))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    fig.savefig('Q4_{}.jpg' .format(window_type))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Magnitude')
    return freq_vector, time_vector, magnitude_spectrogram

block_size = 2048
hop_size = 1024
window_type = 'rect'
(freq_vector, time_vector, magnitude_spectrogram) = mySpecgram(x,  block_size, hop_size, sampling_rate_Hz, window_type)

window_type = 'hann'
(freq_vector, time_vector, magnitude_spectrogram) = mySpecgram(x,  block_size, hop_size, sampling_rate_Hz, window_type)
