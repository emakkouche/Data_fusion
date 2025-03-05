#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:28:13 2022

@author: e.akkouche
"""

import matplotlib.pyplot as plt
import numpy as np

from tools import compute_symmpad_3d, _centered, compute_symmpad
from astropy.io import fits
from tools import *

from CONSTANTS import *

import warnings
warnings.filterwarnings('ignore')

def produce_HS_nir_bis(M, A, tabwave, fname, dname, snr=50, sigma=0, d=0.31, SUBSAMPLIG=False):
    """
    Produce the HS image (each band at a time)
    """
    SUBSAMPLIG = True
    
    #Get spectral throughput
    Lh = fits.getdata(DATA+'Lh.fits')

    # Get spectral PSF
    L = get_spec_psf()
    
    # Shapes
    lh, m = M.shape

    n, p, q = A.shape
    
    # Initialize HS image
    if SUBSAMPLIG:
        #Shapes if subsampling
        p_ = int(p//(1/d)+1)
        q_ = int(q//(1/d)+1)
        Y = np.zeros((lh, p_, q_)) 
    else:
        Y = np.zeros((lh,p,q))

    for i in range(lh):
        # Compute the 3D scene at a wavelength 
        X = compute_symmpad(np.reshape(np.dot(M[i], np.reshape(A, (n, p*q))), (p, q)), fact_pad)
        # Get spatial PSF at that band
        H = get_spa_bandpsf_hs(i, sigma)
        # Adjust the shape of PSF
        Hcrop = reshape_psf(H, X)
        
        # Convolve with PSF without subsample
        Y[i] = np.real(_centered(np.fft.ifft2(Hcrop*np.fft.fft2(X, norm='ortho'), norm='ortho')[:-2, :-2], (p, q)))
        
        # print(f"i = {i}")  
        '''---------------------------------------'''
        
    print('\n********** Simulation Done  !!**********\n')

    return Y

"""---------------------------------------------------------"""
def get_spec_psf():
    # Get spectral PSF (1-D gaussian blur)
    return fits.getdata(DATA+'PSF_spec.fits')

"""---------------------------------------------------------"""
def get_spa_bandpsf_hs(band, sigma=0):
    # Get a spatial Point Spread Function at wavelength number 'band'. It has to be calculated in the Fourier domain and saved in advance with webbpsf.
    g_ = fits.getdata(PSF+'M_fft.fits')[:, band]
    k, m, n = g_.shape
    g = g_[0]+g_[1]*1.j
    return np.reshape(g, (m, n))
"""---------------------------------------------------------"""
def reshape_psf(H,X):
    
    #Crop PSF sur les dimensions de X pour effectuer produit terme à terme
    """
    Lin(X) = Lin(A) + 2 * (fact_pad + 1)
    Col(X) = Col(A) + 2 * (fact_pad + 1)
    
    Lin(PSF) > Lin(X)
    Col(PSF) < Col(X)
    """
    
    Hifft = np.fft.ifft2(H)
        
    Hifft = np.fft.ifftshift(Hifft)
        
    ind = np.unravel_index(np.argmax(Hifft, axis=None), Hifft.shape)
        
    RLbnd = ind[0] - NR//2
    RUbnd = ind[0] + NR//2
    CLbnd = ind[1] - NC//2
    CUbnd = ind[1] + NC//2
      
    SHIFT = (NC - H.shape[1])//2
    
    Hcrop = np.zeros(X.shape,dtype = complex)    
        
    Hcrop[:,SHIFT:(SHIFT+H.shape[1])] = Hifft[RLbnd:RUbnd,:]
    
    Hcrop /= np.sum(Hcrop)
    
    Hcrop = np.fft.fft2(np.fft.fftshift(Hcrop))
    
    return Hcrop

"""---------------------------------------------------------"""
def add_noise_nocorr(Y, sigma2):
    # Compute additive noise with no spatial correlation
    L, M, N = Y.shape
    noise = np.sqrt(sigma2)*np.random.randn(L, M, N)
    
    print('Apply nocorr noise done') #Asupprimer
    return noise

"""---------------------------------------------------------"""
def main(file_data, file_save, sigma2,snr=50, sigma=0): 
    ##### Get the 3D scene with full resolution
    M = fits.getdata(file_data+'M_1.fits').T
    A = fits.getdata(file_data+'A.fits')
    
    A = A[:,:,START:START+NBCOL_X]
    #####

    ##### Get specification : wavelength table, instrument spec.
    tabwave = fits.getdata(file_data+'tabwave.fits')[:, 0]
    channel = 'short'
    fname = 'na'
    dname = 'na'
    #####

    ##### Compute images
    print('Simulating HS and MS images ...')
    Y = produce_HS_nir_bis(M, A, tabwave, fname, dname, snr, sigma)
    Yh_highsnr = Y + add_noise_nocorr(Y,sigma2)
    
    #Ym_highsnr = apply_noise(produce_MS_nir_bis(M, A, tabwave, channel, snr, sigma), 'nircam') #Modifié
    #####

    ##### Save images
    print('Saving HS and MS images ...')
    # Save high snr images
    hdu = fits.PrimaryHDU(Yh_highsnr)
    hdu.writeto(HS_IM, overwrite=True)
    #hdu = fits.PrimaryHDU(Ym_highsnr)
    #hdu.writeto(MS_IM, overwrite=True)
    ######
    
    return Yh_highsnr