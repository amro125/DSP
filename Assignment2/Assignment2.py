#Assignment 2
#Amit Rogel
import numpy as np
import scipy.io
from scipy.io import wavfile
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import time

testa = np.array([1, 2, 3], dtype=float)
testb = np.array([0, 1, 1], dtype=float)

def loadSoundFile(filename):
    [sr, x] = read(filename)
    if x.ndim > 1:
        x = x[:,0]
    return x / abs(np.max(x))


z=[]
#Question1
#if x is 200 length and h is 100 the length of y is 299
def myTimeConv(x,h):
    xlen = len(x)
    hlen = len(h)
    convlen = xlen+hlen-1
    y = np.zeros([convlen,1])
    # if hlen < xlen:
    #     h=np.pad(h,(np.abs(xlen-hlen),0), mode = 'constant')
    #JUST DO IT PIECEWISE 5HEAD
    for i in range(0,hlen):
        y[i] = sum(x[0:i]*h[hlen-i:hlen])
    for i in range(0,(xlen-hlen)):
        y[i+hlen]=sum(x[i:i+hlen]*h[0:hlen])
    for i in range(0, (hlen-1)):
        y[i+xlen] = sum(x[xlen-hlen+i:xlen]*h[0:hlen-i])
    return y

    # return y

x = np.ones(200)
h_start = np.linspace(0,1,num=26)
h_end = np.linspace(1,0,num=26)
h = np.append(h_start,h_end[1:])
y = myTimeConv(x,h)

fig1 = plt.plot(y)
plt.xlabel('Time')
plt.ylabel('Convolution')
plt.title('Convolution Between Steady Signal and Impulse')
plt.show(fig1)      
plt.savefig('results/02-convolution.png')

x = loadSoundFile('piano.wav')
h = loadSoundFile('impulse-response.wav')
 

print('Ready for function2!')
def CompareConv(x,h):
    #Does the convolution in SCipy
    start = time.time() #starts timere to figure time takes to calculate
    scipyconv = myTimeConv(x,h)
    end = time.time()#stops timer
    scipyt = end - start
    
    #Does the custom written convolution
    start = time.time()
    myConv = scipy.convolve(x,h, mode = 'full')
    myend = time.time()
    myt = myend - start    
    
    #Compares restults
    scipyconv = np.transpose(scipyconv)
    myConv = np.transpose(myConv)
    diff = scipyconv-myConv
    
    m = np.mean(diff)
    mabs = abs(m)
    stdev = np.std(diff)
    t = np.append(scipyt,myt)
    return m,mabs,stdev,t
       

#Runs function and saves result
(m,mabs,stdev,t) = CompareConv(x,h)
np.savetxt('results/02-m',m,fmt = '%-3f')
np.savetxt('results/02-mabs',mabs,fmt = '%-3f') 
np.savetxt('results/02-stdev',stdev,fmt = '%-3f') 
np.savetxt('results/02-time',t,fmt = '%-3f') 
print('done!')
