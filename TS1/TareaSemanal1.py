# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:31:46 2023

@author: Martin
"""
# Importo librerías
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# funcion Seno
def mi_funcion_sen( vmax, dc, ff, ph, nn, fs):
    
    Ts = 1/fs
    tt = np.arange(0, nn*Ts, Ts)
    ww = 2 * np.pi * ff
    
    xx = vmax * np.sin(ww * tt + ph) + dc
    
    return tt, xx

def mi_funcion_cuad( vmax, dc, ff, ph, nn, fs, ton=0.5):
    
    #Tiempo
    Ts = 1/fs                       #Periodo de muestreo
    T_expl = Ts*nn                  #Tiempo de exploracion
    ww = 2 * np.pi * ff             #Frecuencia angular
    tt = np.arange(0, T_expl, Ts)   #base de tiempo
    
    #ceros = int(tf*(1-ton))         #

    #xx = np.ones(nn)
    #xx[0:ceros] = 0
    
    #xx = xx*vmax+dc
    #xx = vmax * np.square()
    
    xx = vmax * signal.square(ww * tt + ph, duty=ton) + dc
    
    return tt, xx



tt, xx = mi_funcion_sen(vmax = 2, dc = 0, ff = 100, ph = 0, nn = 10000, fs = 10000)
tt, aa = mi_funcion_sen(vmax = 4, dc = 0, ff = 2, ph = 0, nn = 10000, fs = 10000)

st=(1+aa)*xx

plt.plot(tt, st)


