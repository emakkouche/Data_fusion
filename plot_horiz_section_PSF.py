#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:56:09 2022

@author: e.akkouche
"""

import numpy as np
# import tools
from time import time


from grad import grad
from div import div
from produce_HS_MS import add_noise_nocorr

import matplotlib.pyplot as plt
from astropy.io import fits
from tools import compute_symmpad_3d, _centered, compute_symmpad
from skimage.transform import resize
from scipy.signal import convolve2d
from CONSTANTS import *

import warnings
warnings.filterwarnings('ignore')

def get_spa_bandpsf_hs(band, sigma=0):
    # Get a spatial Point Spread Function at wavelength number 'band'. It has to be calculated in the Fourier domain and saved in advance with webbpsf.
    g_ = fits.getdata(PSF+'M_fft.fits')[:, band]
    k, m, n = g_.shape
    g = g_[0]+g_[1]*1.j
    return np.reshape(g, (m, n))


Band = 10

w = 10

H = get_spa_bandpsf_hs(Band, sigma=0)

Hifft = np.fft.ifft2(H)
        
Hifft = np.fft.ifftshift(Hifft)
    
ind = np.unravel_index(np.argmax(Hifft, axis=None), Hifft.shape)

f = plt.figure()
plt.imshow(np.log(1e-7 + np.abs(Hifft)));plt.colorbar()
plt.axis('tight')
plt.xlabel('')
plt.ylabel('')
plt.title('PSF NIRSpec Band = 1000')
f.show()

f = plt.figure()
plt.plot(np.abs(Hifft)[ind[0]-w:ind[0]+w,ind[1]-w:ind[1]+w])
plt.axis('tight')
plt.xlabel('')
plt.ylabel('Amplitude')
plt.title('Coupe horizontale PSF NIRSpec Band = 1000')
f.show()

