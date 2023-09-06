# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 20:06:40 2023

@author: Martin
"""

# Importo librerías
import numpy as np
import matplotlib.pyplot as plt
import time

# funcion Seno
def mi_funcion_sen( vmax, dc, ff, ph, nn, fs, snr):
    
    Ts = 1/fs
    tt = np.arange(0, nn*Ts, Ts)
    ww = 2 * np.pi * ff
    vnorm = np.sqrt(2) 
    
    clean_signal = vnorm * np.sin(ww * tt + ph) + dc
    
    pow_noise = pow(base=10, exp= -snr/10)
    vnoise = np.sqrt(pow_noise*12)
    
    noise = np.random.uniform(-vnoise/2, vnoise/2, size = nn)
    
    
    
    xx = (clean_signal+noise)*vmax/np.sqrt(2)
    
    return tt, xx

def mi_funcion_DFT_v1( xx ):
    
    NN = len(xx)
    
    XX = np.zeros(NN, dtype=np.complex64)
    
    for i in range(NN):
        for j in range (NN):
            XX[i] += xx[j] * np.exp(-2j * np.pi * i * j / NN) #/ NN
    return XX

def mi_funcion_DFT_v2( xx ):
    
    NN = len(xx)
    
    XX = np.zeros(NN, dtype=np.complex64)
    indexes = np.arange(NN)
    
    for i in range(NN):
        XX[i] += sum(xx * np.exp(-2j * np.pi * i * indexes / NN)) #/ NN
    return XX


vmax = 20
dc = 0
ff = 10
ph = 0
nn = 1000
fs = nn
snr = 50
tt, xx = mi_funcion_sen( vmax, dc, ff, ph, nn, fs, snr)

plt.plot(tt, xx)

N = nn
df = fs/nn
ff = np.linspace(0, (N-1)*df, N)

print("Tiempos de ejecucion:")
start = time.time()
fft_ft_XX = 1/N*np.fft.fft( xx, axis = 0 )
end = time.time()
print("  np.fft.fft:        "+str(round(end - start,4))+"s")

start = time.time()
my_ft_XX = mi_funcion_DFT_v1(xx)
end = time.time()
print("  mi_funcion_DFT_v1: "+str(round(end - start,4))+"s")

start = time.time()
my_ft_XX_2 = mi_funcion_DFT_v2(xx)
end = time.time()
print("  mi_funcion_DFT_v2: "+str(round(end - start,4))+"s")

plt.figure(1)
plt.subplot(2,1,1)
plt.plot( tt, 10 * xx)
plt.subplot(2,1,2)
bfrec = ff <= fs/2
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(fft_ft_XX[bfrec])**2))
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(my_ft_XX[bfrec])**2))
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(my_ft_XX_2[bfrec])**2),'-.')

