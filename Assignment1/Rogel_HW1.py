#Assignment 1
#Amit Rogel
import numpy as np
import scipy.io
from scipy.io import wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

testa = np.array([1, 2, 3], dtype=float)
testb = np.array([0, 1, 1], dtype=float)


z=[]
#Question1

def crossCorr(x,y):
    z = np.correlate(x,y)
    return z


def loadSoundFile(filename):
    samplerate , x = scipy.io.wavfile.read(filename)
    x1 = x[:, 0]
    return x1


snare = []
drum_loop = []
#creates correlation function for Q1
def correlation(x,y):
    #Loads files
    snare =loadSoundFile(x)
    drum_loop = loadSoundFile(y)
    #Finds correlations
    correlation_result = crossCorr(drum_loop,snare)
    #Formats plot
    fig1 = plt.plot(correlation_result)
    plt.xlabel('Position')
    plt.ylabel('Correlation')
    plt.title('Correlation Between Snare and Drum Loop')
    #Saves plot and displays it
    plt.savefig('results/01-correlation.png')
    plt.show(fig1)

#Runs function
correlation('Snare.wav','drum_loop.wav')

#Question 2
def findSnarePosition(snareFilename,drumloopFilename):
    #imports files
    snare =loadSoundFile(snareFilename)
    drum_loop = loadSoundFile(drumloopFilename)
    #finds correlation
    correlation_result = crossCorr(drum_loop,snare)
    #Snare is mostly liekly to occur at the highest correlation between the samples, therefore we search for the maximums
    snareLocation = find_peaks(correlation_result,distance=1000) #increased distance to filter results
    snareLocation = snareLocation[0]
    #saves possible list as .txt
    np.savetxt('results/02-snareLocation.txt',snareLocation,fmt = '%-3f',newline='\n')

#runs function
findSnarePosition('Snare.wav','drum_loop.wav')
