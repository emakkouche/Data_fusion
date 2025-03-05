#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:50:05 2022

@author: e.akkouche
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


from CONSTANTS import *

import tools

import warnings
warnings.filterwarnings('ignore')

def get_spa_bandpsf_hs(band, sigma=0):
    # Get a spatial Point Spread Function at wavelength number 'band'. It has to be calculated in the Fourier domain and saved in advance with webbpsf.
    g_ = fits.getdata(PSF+'H_fft.fits')[:, band]
    k, m, n = g_.shape
    g = g_[0]+g_[1]*1.j
    return np.reshape(g, (m, n))


def process_PSF(Y,sigma = 0):
    
    print('\n****** Processing PSF ******\n')
    
    #VZ c'est Y
    Nband,Lin,Col = Y.shape
    
    PSF = np.zeros((Nband,Lin,Col),dtype = complex)
    
    Hreshaped = np.zeros((Lin,Col),dtype = complex)
    
    # H = get_spa_bandpsf_hs(1, sigma=0)
    
    # L,C = H.shape
    
    # Hamming = np.tile(np.hamming(C),(Lin,1))
    
    # for k in range(Nband):
    for k in np.arange(0,Nband,500):
        
        H = get_spa_bandpsf_hs(k, sigma)
        
        Hifft = np.fft.ifft2(H,norm='ortho')
        
        Hifft = np.fft.ifftshift(Hifft)
    
        ind = np.unravel_index(np.argmax(Hifft, axis=None), Hifft.shape)
        
        RLbnd = ind[0] - Lin//2
        RUbnd = ind[0] + Lin//2
        # CLbnd = ind[1] - H.shape[1]//2
        # CUbnd = ind[1] + H.shape[1]//2
        Nshift = (Col - H.shape[1])//2
        
        # Hreshaped[:,Nshift:(Nshift+C)] = Hshift[RLbnd:RUbnd,:] * Hamming
        
        Hreshaped[:,Nshift:(Nshift+H.shape[1])] = Hifft[RLbnd:RUbnd,:] 
        
        # Hreshaped = np.fft.fftshift(Hreshaped)
        
        # PSF[k] = np.fft.fftshift(Hreshaped)
        
        PSF[k] = np.fft.fftshift(Hreshaped)
        
        Hreshaped *= 0
        
        print('k = '+ str(k))
    
    print('\n****** Processing PSF Done ******\n')
    
    return np.fft.fft2(PSF,axes=(1, 2),norm='ortho')


Yh = fits.getdata(HS_IM)

Yfft = np.fft.fft2(tools.compute_symmpad_3d(Yh,fact_pad),axes=(1, 2),norm='ortho')

PSF_resize = process_PSF(np.reshape(Yfft,(4974,150,414)))

M = fits.getdata(DATA+'M_1.fits').T
A = fits.getdata(DATA+'A.fits')

A = A[:,:,Start:Start+NbcolMA]

n, p, q = A.shape

k = 1000

X = tools.compute_symmpad(np.reshape(np.dot(M[k], np.reshape(A, (n, p*q))), (p, q)), fact_pad)

Xinit = np.reshape(np.dot(M[k], np.reshape(A, (n, p*q))), (p, q))

Yblur = np.real(tools._centered(np.fft.ifft2(PSF_resize[k]*np.fft.fft2(X, norm='ortho'), norm='ortho')[:-2, :-2], (p, q)))
