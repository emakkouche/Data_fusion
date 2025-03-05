#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:04:15 2022

@author: e.akkouche
"""
from time import time

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.transform import resize

import acp_v2 as pca
from CONSTANTS import *
from errors import snr
from grad import grad
from div import div
from produce_HS_MS import get_spa_bandpsf_hs

"""---------------------Sans padding bordures--------------------"""
"""REMARQUE!!! dim(H) > dim(Y)"""
BAND = 3541
START = 10
NB_COL_MA = 500
ERROR_RATIO = 1e-10
EPSILON = 0.4*1e-2
NB_ITER = 2000

Yh = fits.getdata(HS_IM)

M = fits.getdata(DATA+'M_1.fits').T
A = fits.getdata(DATA+'A.fits')

A = A[:,:,START:START+NB_COL_MA]

n, p, q = A.shape

Xinit = np.reshape(np.dot(M[BAND], np.reshape(A, (n, p*q))), (p, q))

Nband,Lin,Col = Yh.shape

H = get_spa_bandpsf_hs(BAND, sigma=0)

Hifft = np.fft.ifft2(H)
        
Hifft = np.fft.ifftshift(Hifft)
    
ind = np.unravel_index(np.argmax(Hifft, axis=None), Hifft.shape)
        
RLbnd = ind[0] - Lin//2
RUbnd = ind[0] + Lin//2
CLbnd = ind[1] - Col//2
CUbnd = ind[1] + Col//2

Nshift = (Col - H.shape[1])//2

Hreshape = np.fft.fftshift(Hifft[RLbnd:RUbnd,CLbnd:CUbnd])

"""-----------------------------------------------------"""

Phi = lambda x,h: np.real(np.fft.ifft2(np.fft.fft2(x,norm = 'ortho') * np.fft.fft2(h,norm = 'ortho')))

repeat3 = lambda x,k: resize( np.repeat( x, k, axis=1), [Yh.shape[1],Yh.shape[1], k])

epsilon = EPSILON

lambda_list = np.linspace(0,2,20)

tau = 1.9 / ( 1 + max(lambda_list) * 8 / epsilon)

niter = NB_ITER

E = np.zeros((niter,1))

err = np.zeros((len(lambda_list),1))

fBest = Yh[BAND,:,:]

for k, Lambda in enumerate(lambda_list):
    
    fTV = Yh[BAND,:,:]

    fTVprec = fTV*100
    
    i = 0
    
    while i < niter and np.linalg.norm(fTV - fTVprec)/np.linalg.norm(fTVprec) > ERROR_RATIO :
    
        # Compute the gradient of the smoothed TV functional.
        Gr = grad(fTV)
        
        d = np.sqrt(epsilon**2 + np.sum(Gr**2, axis=2))
        
        G = -div(Gr[:,:,0] / d,Gr[:,:,1] / d )
        # step
        e = Phi(fTV,Hreshape) - Yh[BAND,:,:]
    
        fTVprec = fTV
    
        fTV = fTV - tau*( Phi(e,Hreshape) + Lambda*G)
        # energy
        E[i] = 1/2*np.linalg.norm(e.flatten())**2 + Lambda*np.sum(d.flatten())
    
        i +=1
    
    err[k] = snr(Xinit,fTV)
    
    if err[k] > snr(Xinit,fBest):
        fBest = fTV
        
    
# while i < niter and np.linalg.norm(fTV - fTVprec)/np.linalg.norm(fTVprec) > 1e-10 :
    
#     # Compute the gradient of the smoothed TV functional.
#     Gr = grad(fTV)
#     d = np.sqrt(epsilon**2 + np.sum(Gr**2, axis=2))
#     G = -div(Gr[:,:,0] / d,Gr[:,:,1] / d )
#     # step
#     e = Phi(fTV,Hreshape) - Yh[124,:,:]
    
#     fTVprec = fTV
    
#     fTV = fTV - tau*( Phi(e,Hreshape) + Lambda*G)
#     # energy
#     E[i] = 1/2*np.linalg.norm(e.flatten())**2 + Lambda*np.sum(d.flatten())
    
#     i = i+1
# display energy

plt.clf;
plt.plot(E)
plt.axis('tight')
plt.xlabel('Iteration #')
plt.ylabel('Energy')

plt.imshow(Xinit)
plt.title('Originale')

plt.imshow(Yh[BAND,:,:])
plt.title('Observée')

plt.imshow(fTV)
plt.title('Deconvolution TV')


plt.plot(lambda_list,err)
plt.axis('tight')
plt.xlabel('\lambda') 
plt.ylabel('SNR')

plt.imshow(fBest)
plt.title('Best SNR TV Deconvolution')