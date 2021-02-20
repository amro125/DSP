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
#Creates sin wave function
def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    Samplenums = int(sampling_rate_Hz*length_secs) #finds number of elements based on sample rate and length
    t = np.linspace(0.0,length_secs,num = Samplenums) #creates t array
    x = amplitude*np.sin(frequency_Hz*t+phase_radians) #creates sin wave
    return x,t

#gives inputs to generate sinwave from function
amplitude = 1.0
sampling_rate_Hz = 44100
frequency_Hz = 400 #Hz
length_secs = 0.5 #seconds
phase_radians = np.pi/2 #radians
#runs function
(xsin,t) = generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians)

#figures out how many samples for the first 5 ms 

plotsample = int(0.005*sampling_rate_Hz)

xplot = xsin[0:plotsample]
tplot = t[0:plotsample]
xplot = xsin[0:plotsample]
tplot = t[0:plotsample]

#plots and defines axes
fig = plt.figure()
plt.plot(tplot,xplot)
fig.suptitle('Sin Wave 400 Hz')
plt.xlabel('Amplitude')
plt.ylabel('Time (s)')
fig.savefig('Q1.jpg') #saves figure

#defines square wave function
def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    Samplenums = int(sampling_rate_Hz*length_secs)
    t = np.linspace(0.0,length_secs,num = Samplenums) #similar to sin wave
    xsquare = np.array([])
    f = frequency_Hz/(2*np.pi)
    for j in t:
        square = np.array([0])
        for i in range(10): #10 sine waves used to approximate square wave
            k = i+1
            squarecurrent = (4/np.pi)*(np.sin(2*np.pi*(2*k-1)*f*j)/(2*k-1)) #formula for square wave 
            square = np.append(square,squarecurrent)
        xsquare =  np.append(xsquare,np.sum(square))
    return xsquare,t

#change necessary values for input
phase_radians = 0

#run function
(x,t) = generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians)

#takes number of samples for first 5 ms 
plotsample = int(0.005*sampling_rate_Hz)

#plots and formats
xplot = x[0:plotsample]
tplot = t[0:plotsample]
fig = plt.figure()
plt.plot(tplot,xplot)
fig.suptitle('Square Wave 400 Hz')
plt.xlabel('Amplitude')
plt.ylabel('Time (s)')
fig.savefig('Q2.jpg')


#question 3
#function to run fft and get mag, phase, real and imaginary parts
def computeSpectrum(x, sample_rate_Hz):
    
    xfft= np.fft.fft(x) #takes fft
    bins = len(xfft) 
    freq = np.linspace(0,(bins-1))
    f = sample_rate_Hz*freq/bins #finds number of bins
    xabs = np.abs(xfft) #takes magnintude
    xphase = np.angle(xfft) #takes phase
    xre = xfft.real #real parts of fft
    xim = xfft.imag #imaginary partts of fft
    return f,xabs,xphase,xre,xim

#runs function with sin wave
(f,xabs,xphase,xre,xim) = computeSpectrum(x, sampling_rate_Hz)

#graphs mag and phase
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Square Wave 400 Hz FFT')
ax1.plot(xabs)
ax1.set_title('Magnitude and Phase Graph')
ax1.set(ylabel = 'Magnitude')
ax2.plot(xphase)
ax2.set(xlabel = 'Frequency (Hz)', ylabel = 'Phase')
fig.savefig('Q3_square.jpg')

#runs with square wave
(f,xabs,xphase,xre,xim) = computeSpectrum(xsin, sampling_rate_Hz)

#plots magnitude and phase
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Sin Wave 400 Hz FFT')
ax1.plot(xabs)
ax1.set_title('Magnitude and Phase Graph')
ax1.set(ylabel = 'Magnitude')
ax2.plot(xphase)
ax2.set(xlabel = 'Frequency (Hz)', ylabel = 'Phase')
fig.savefig('Q3_sin.jpg')

#creates spectogram blocks
def generateBlocks(x, sample_rate_Hz, block_size, hop_size):
    lx = len(x);
    numHop = int(np.ceil(lx/hop_size))#find number of hops for size
    tTotal = sample_rate_Hz*lx #total time
    t = np.linspace(0,tTotal,num=numHop) #creats time array
    X=np.empty([int(block_size),numHop])
    for i in range(numHop): #iterates through base on hop and block size to count for overlap
        block = x[(i*hop_size):(i*hop_size+block_size)]#takes current block
        if len(block) < block_size:#checks if we need to pad
            block = np.append(block,np.zeros(block_size-len(block)))
        X[:,i] = block#adds block to matrix
    return t,X


#create spectogram function
def mySpecgram(x,  block_size, hop_size, sampling_rate_Hz, window_type):
    lx = len(x)
    if window_type == 'hann': #checks which window to do and applies correct filter
        han = np.hanning(lx)
        x = x*han
    (time_vector,X) = generateBlocks(x, sampling_rate_Hz, block_size, hop_size) # creates blocks
    (freq_vector,magnitude_spectrogram,xphase,xre,xim) = computeSpectrum(X, sampling_rate_Hz) #takes fft of blocks
    fig = plt.figure()
    plt.specgram(x,NFFT=block_size,Fs = sampling_rate_Hz, noverlap = (block_size-hop_size) )#plots spectrogram
    #formats spectogram
    fig.suptitle('Spectrogram of 400 Hz Square Wave with {} Windowing' .format(window_type))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    fig.savefig('Q4_{}.jpg' .format(window_type))
    cbar = plt.colorbar() #adds colorbar legend
    cbar.ax.set_ylabel('Magnitude')
    return freq_vector, time_vector, magnitude_spectrogram
#runs function for rectangular window
block_size = 2048
hop_size = 1024
window_type = 'rect'
(freq_vector, time_vector, magnitude_spectrogram) = mySpecgram(x,  block_size, hop_size, sampling_rate_Hz, window_type)

#runs function with hanning window
window_type = 'hann'
(freq_vector, time_vector, magnitude_spectrogram) = mySpecgram(x,  block_size, hop_size, sampling_rate_Hz, window_type)
