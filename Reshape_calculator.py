#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:10:53 2022

@author: e.akkouche
"""

import numpy as np
# import tools
from time import time


from grad import grad
from div import div
from produce_HS_MS import add_noise_nocorr
from optimize import save_Zoptim

import matplotlib.pyplot as plt
from astropy.io import fits
from tools import compute_symmpad_3d, _centered, compute_symmpad
from skimage.transform import resize
from scipy.signal import convolve2d
from skimage.restoration import denoise_tv_chambolle
# from pandeia.engine.instrument_factory import InstrumentFactory
from CONSTANTS import *
from errors import *
import warnings
warnings.filterwarnings('ignore')


def get_spa_bandpsf_hs(band, sigma=0):
    # Get a spatial Point Spread Function at wavelength number 'band'. It has to be calculated in the Fourier domain and saved in advance with webbpsf.
    g_ = fits.getdata(PSF+'M_fft.fits')[:, band]
    k, m, n = g_.shape
    g = g_[0]+g_[1]*1.j
    return np.reshape(g, (m, n))


#####################Version qui fonctionne####################

# Hreshaped = np.zeros((90,354),dtype = complex)
# H = get_spa_bandpsf_hs(4000, sigma=0)
# Hamming = np.tile(np.hamming(H.shape[1]),(Yh.shape[0],1))
# Hshift = (np.fft.fftshift(H))             #A Supprimer
# ind = np.unravel_index(np.argmax(Hshift, axis=None), Hshift.shape)


# RLbnd = ind[0] - 90//2
# RUbnd = ind[0] + 90//2
# CLbnd = ind[1] - 354//2
# CUbnd = ind[1] + 354//2
        
#         #Reshape PSF
        
# # Hreshaped[:,Nshift:(Nshift+NbpixC)] = Hshift[RLbnd:RUbnd,CLbnd:CUbnd] #Hreshape ==> 174x984 Hshift 174x384
# Hreshaped = Hshift[RLbnd:RUbnd,CLbnd:CUbnd]
# Hreshaped[:,Nshift:(Nshift+NbpixC)] *= Hamming
# Hreshaped = np.fft.fftshift(Hreshaped)


# Phi = lambda x,h: np.real(np.fft.ifft2(np.fft.fft2(x) * h))
# repeat3 = lambda x,k: resize( np.repeat( x, k, axis=1), [90, 354, k])
# epsilon = 0.4*1e-2
# Lambda = 0.2
# y = Yh[4000,:,:]
# tau = 1.9 / ( 1 + Lambda * 8 / epsilon)
# Xestim = y
# niter = 2000
# E = np.zeros((niter,1))
# for i in np.arange(0, niter):
#     # Compute the gradient of the smoothed TV functional.
#     Gr = grad(Xestim)
#     d = np.sqrt(epsilon**2 + np.sum(Gr**2, axis=2))
#     G = -div(Gr[:,:,0] / d,Gr[:,:,1] / d )
#     # step
#     e = Phi(Xestim,Hreshaped)-y
#     Xestim = Xestim - tau*( Phi(e,Hreshaped) + Lambda*G)
#     # energy
#     E[i] = 1/2*np.linalg.norm(e.flatten())**2 + Lambda*sum(d.flatten())
# # display energy
# clf;
# plt.plot(E)
# axis('tight')
# xlabel('Iteration #')
# ylabel('Energy')
############################################################

# Band = 1000

# Yh = fits.getdata(HS_IM)

# M = fits.getdata(DATA+'M_1.fits').T
# A = fits.getdata(DATA+'A.fits')

# A = A[:,:,Start:Start+NbcolMA]

# n, p, q = A.shape

# Xinit = np.reshape(np.dot(M[Band], np.reshape(A, (n, p*q))), (p, q))

# Nband,Lin,Col = Yh.shape

# H = get_spa_bandpsf_hs(Band, sigma=0)

# Hifft = np.fft.ifft2(H,norm = 'ortho')
# Hifft = np.fft.ifft2(H)

        
# Hifft = np.fft.ifftshift(Hifft)
    
# ind = np.unravel_index(np.argmax(Hifft, axis=None), Hifft.shape)
        
# RLbnd = ind[0] - Lin//2
# RUbnd = ind[0] + Lin//2
# CLbnd = ind[1] - Col//2
# CUbnd = ind[1] + Col//2

# Nshift = (Col - H.shape[1])//2

# Hreshaped[:,Nshift:(Nshift+H.shape[1])] = Hifft[RLbnd:RUbnd,:] 

# Hreshape = np.fft.fftshift(Hifft[RLbnd:RUbnd,CLbnd:CUbnd])

# Hreshape /= np.sum(Hreshape)

# Yblur = np.real(np.fft.ifft2(np.fft.fft2(Hreshape)*np.fft.fft2(Xinit,norm = 'ortho'), norm='ortho'))

# Yblur +=  np.sqrt(0.1)*np.random.randn(Lin,Col) 

# Yh[Band,:,:] = Yblur

# Yblur = Yh[Band,:,:]

# PSF[k] = np.fft.fftshift(Hreshaped)
"""-----------------------------------------------------"""

# Phi = lambda x,h: np.real(np.fft.ifft2(np.fft.fft2(x,norm = 'ortho') * np.fft.fft2(h,norm = 'ortho'),norm = 'ortho'))

# repeat3 = lambda x,k: resize( np.repeat( x, k, axis=1), [90, 354, k])

# dotp = lambda x,y: np.sum(x.flatten()*y.flatten())
# a = np.random.randn(90,354)
# b = np.random.randn(90,354,2)
# dotp(grad(a),b) + dotp(a,div(b[:,:,0],b[:,:,1]))

# epsilon = 0.2*1e-1
# # lambda_list = np.linspace(0,2,20)
# lambda_list = np.log(np.linspace(1,3,20))

# tau = 1.9 / ( 1 + max(lambda_list) * 8 / epsilon)

# niter = 4000

# E = np.zeros((niter,1))

# err = np.zeros((len(lambda_list),1))

# Xbest = Yh[Band,:,:]
# Xbest = Yblur
# Xbest = np.zeros(Xinit.shape)

# for k in np.arange(0,len(lambda_list)):
    
    # Xestim = Yh[Band,:,:]
    # Xestim = Yh[Band,:,:] * np.random.rand(Lin,Col) 
    # Xestim = np.zeros(Xinit.shape)
    # Xestim = np.random.rand(Lin,Col) 
    
    # Xestim_prev = Xestim*100
    
    # i = 0
    
    # Lambda = lambda_list[k]
    
    # while i < niter and np.linalg.norm(Xestim - Xestim_prev)/np.linalg.norm(Xestim_prev) > 1e-5 :
    
    #     # Compute the gradient of the smoothed TV functional.
    #     Gr = grad(Xestim)
    #     d = np.sqrt(epsilon**2 + np.sum(Gr**2, axis=2))
    #     # d = np.sqrt(epsilon**2 + np.linalg.norm(Gr))
    #     G = -div(Gr[:,:,0] / d,Gr[:,:,1] / d )
        
    #     # step
    #     e = Phi(Xestim,Hreshape) - Yh[Band,:,:]
    
    #     Xestim_prev = Xestim
    
    #     Xestim = Xestim - tau*( Phi(e,Hreshape) + Lambda*G)
    #     # energy
    #     E[i] = 1/2*np.linalg.norm(e.flatten())**2 + Lambda*np.sum(d.flatten())
    
    #     i = i+1
    
    # err[k] = snr(Xinit,Xestim)
    
    # if err[k] > snr(Xinit,Xbest):
    #     Xbest = Xestim
        
    
# while i < niter and np.linalg.norm(Xestim - Xestim_prev)/np.linalg.norm(Xestim_prev) > 1e-10 :
    
#     # Compute the gradient of the smoothed TV functional.
#     Gr = grad(Xestim)
#     d = np.sqrt(epsilon**2 + np.sum(Gr**2, axis=2))
#     G = -div(Gr[:,:,0] / d,Gr[:,:,1] / d )
#     # step
#     e = Phi(Xestim,Hreshape) - Yh[124,:,:]
    
#     Xestim_prev = Xestim
    
#     Xestim = Xestim - tau*( Phi(e,Hreshape) + Lambda*G)
#     # energy
#     E[i] = 1/2*np.linalg.norm(e.flatten())**2 + Lambda*np.sum(d.flatten())
    
#     i = i+1
# display energy

# plt.plot(E)
# plt.axis('tight')
# plt.xlabel('Iteration #')
# plt.ylabel('Energie')

# plt.imshow(Xinit)
# plt.title('Référence')

# plt.imshow(Yblur)
# plt.title('Observée')

# plt.imshow(Yh[Band,:,:])
# plt.title('Observée')

# plt.imshow(Xestim)
# plt.title('Deconvolution - Variation totale')


# plt.plot(lambda_list,err)
# plt.axis('tight')
# plt.xlabel('\lambda (échelle log)') 
# plt.ylabel('SNR')

# plt.imshow(Xbest)
# plt.title('Variation totale SNR = '+str(round(np.max(err),2))+'dB')    


# plt.plot(Xinit[63,:], 'r',label = 'Référence') # plotting t, a separately 
# plt.plot(Xbest[63,:], 'b',label = 'Variation totale') # plotting t, b separately 
# plt.legend(loc="upper right")


# plt.plot(Yblur[63,:],'g')
# plt.show()
    
# print('Finish')

Band = 4000

Yh = fits.getdata(HS_IM)

M = fits.getdata(DATA+'M_1.fits').T
A = fits.getdata(DATA+'A.fits')

A = A[:,:,START:START+NBCOL_X]

n, p, q = A.shape

Xinit = np.reshape(np.dot(M[Band], np.reshape(A, (n, p*q))), (p, q))

Nband,Lin,Col = Yh.shape

H = get_spa_bandpsf_hs(Band, sigma=0)

Hifft = np.fft.ifft2(H)
        
Hifft = np.fft.ifftshift(Hifft)
    
ind = np.unravel_index(np.argmax(Hifft, axis=None), Hifft.shape)
        
RLbnd = ind[0] - Lin//2
RUbnd = ind[0] + Lin//2
CLbnd = ind[1] - Col//2
CUbnd = ind[1] + Col//2

Nshift = (Col - H.shape[1])//2

# Hreshaped[:,Nshift:(Nshift+H.shape[1])] = Hifft[RLbnd:RUbnd,:] 

Hreshape = np.fft.fftshift(Hifft[RLbnd:RUbnd,CLbnd:CUbnd])

Hreshape /= np.sum(Hreshape)

Yblur = np.real(np.fft.ifft2(np.fft.fft2(Hreshape)*np.fft.fft2(Xinit,norm = 'ortho'), norm='ortho'))

Yblur +=  np.sqrt(0.1)*np.random.randn(Lin,Col) 

Yh[Band,:,:] = Yblur

# Yblur = Yh[Band,:,:]

# PSF[k] = np.fft.fftshift(Hreshaped)

"""----------"""
#Phi = lambda x,h: np.real(np.fft.ifft2(np.fft.fft2(x,norm = 'ortho') * np.fft.fft2(h,norm = 'ortho'),norm = 'ortho'))
Phi = lambda x,h: np.real(np.fft.ifft2(np.fft.fft2(x) * np.fft.fft2(h)))

repeat3 = lambda x,k: resize( np.repeat( x, k, axis=1), [90, 354, k])

"""-------"""
epsilon = 0.2
# lambda_list = np.linspace(0,2,20)
lambda_list = np.linspace(0,3,10)

tau = 1.9 / ( 1 + max(lambda_list) * 8 / epsilon)

niter = 4000

E = np.zeros((niter,1))
min_phi= np.zeros((niter,1))
max_phi= np.zeros((niter,1))

err = np.zeros((len(lambda_list),1))

# Xbest = Yh[Band,:,:]
#Xbest = Yblur
Xbest = np.zeros(Xinit.shape)

for k in np.arange(0,len(lambda_list)):
    
    #Xestim = Yh[Band,:,:]
    # Xestim = Yh[Band,:,:] * np.random.rand(Lin,Col) 
    Xestim = np.zeros(Xinit.shape)
    #Xestim = np.random.rand(Lin,Col) 
    
    Xestim_prev = Xestim+1
    
    i = 0
     
    Lambda = lambda_list[k]
    
    while i < niter and np.linalg.norm(Xestim - Xestim_prev)/np.linalg.norm(Xestim_prev) > 1e-5 :
    
        # Compute the gradient of the smoothed TV functional.
        Gr = grad(Xestim)
        d = np.sqrt(epsilon**2 + np.sum(Gr**2, axis=2))
        # d = np.sqrt(epsilon**2 + np.linalg.norm(Gr))
        G = -div(Gr[:,:,0] / d,Gr[:,:,1] / d )
        
        tempo = Phi(Xestim,Hreshape)
        # step
        
        max_phi[i] = np.max(tempo)
        min_phi[i] = np.min(tempo)
        
        e = tempo - Yh[Band,:,:]
    
        Xestim_prev = Xestim
    
        Xestim = Xestim - tau*( Phi(e,Hreshape) + Lambda*G)
        # energy
        E[i] = 1/2*np.linalg.norm(e.flatten())**2 + Lambda*np.sum(d.flatten())
    
        i = i+1
    
    err[k] = snr(Xinit,Xestim)
    
    if err[k] > snr(Xinit,Xbest):
        Xbest = Xestim

    fname = SAVE2+'XoptimTV_mu_'+str(k)+'.fits'
    save_Zoptim(Xestim, fname)
        
    fname = SAVE2+'JTV_mu_'+str(k)
    np.save(fname,E)


fname = SAVE2+'SNR_TV'
np.save(fname,err)
        
plt.plot(E)
plt.axis('tight')
plt.xlabel('Iteration #')
plt.ylabel('Energie')

plt.imshow(Xinit);plt.colorbar()
plt.title('Référence')

plt.imshow(Yblur);plt.colorbar()
plt.title('Observée')

# plt.imshow(Yh[Band,:,:])
# plt.title('Observée')

plt.imshow(Xestim);plt.colorbar()
plt.title('Deconvolution - Variation totale')

plt.plot(lambda_list,err)
plt.axis('tight')
plt.xlabel('\lambda (échelle log)') 
plt.ylabel('SNR')

plt.imshow(Xbest);plt.colorbar()
plt.title('Variation totale SNR = '+str(round(np.max(err),2))+'dB')        

# Band = 3541

# Yh = fits.getdata(HS_IM)

# M = fits.getdata(DATA+'M_1.fits').T
# A = fits.getdata(DATA+'A.fits')

# A = A[:,:,Start:Start+NbcolMA]

# n, p, q = A.shape

# Xinit = np.reshape(np.dot(M[Band], np.reshape(A, (n, p*q))), (p, q))

# Nband,Lin,Col = Yh.shape

# H = get_spa_bandpsf_hs(Band, sigma=0)

# Hifft = np.fft.ifft2(H)
        
# Hifft = np.fft.ifftshift(Hifft)
    
# ind = np.unravel_index(np.argmax(Hifft, axis=None), Hifft.shape)
        
# RLbnd = ind[0] - Lin//2
# RUbnd = ind[0] + Lin//2
# CLbnd = ind[1] - Col//2
# CUbnd = ind[1] + Col//2

# Nshift = (Col - H.shape[1])//2

# # Hreshaped[:,Nshift:(Nshift+H.shape[1])] = Hifft[RLbnd:RUbnd,:] 

# Hreshape = np.fft.fftshift(Hifft[RLbnd:RUbnd,CLbnd:CUbnd])

# # PSF[k] = np.fft.fftshift(Hreshaped)
# """-----------------------------------------------------"""

# Phi = lambda x,h: np.real(np.fft.ifft2(np.fft.fft2(x,norm = 'ortho') * np.fft.fft2(h,norm = 'ortho')))

# repeat3 = lambda x,k: resize( np.repeat( x, k, axis=1), [90, 354, k])

# # dotp = lambda x,y: np.sum(x.flatten()*y.flatten())
# # a = np.random.randn(90,354)
# # b = np.random.randn(90,354,2)
# # dotp(grad(a),b) + dotp(a,div(b[:,:,0],b[:,:,1]))


# epsilon = 0.4*1e-2

# tau = 1.9 / ( 1 + Lambda * 8 / epsilon)

# niter = 2000
# E = np.zeros((niter,1))

# lambda_list = np.linspace(0,1,20)
# err = np.zeros((len(lambda_list),1))

# Xbest = Yh[Band,:,:]

# for k in np.arange(0,len(lambda_list)):
    
#     Xestim = Yh[Band,:,:]

#     Xestim_prev = Xestim*100
    
#     i = 0
    
#     Lambda = lambda_list[k]
    
#     while i < niter and np.linalg.norm(Xestim - Xestim_prev)/np.linalg.norm(Xestim_prev) > 1e-10 :
    
#         # Compute the gradient of the smoothed TV functional.
#         Gr = grad(Xestim)
#         d = np.sqrt(epsilon**2 + np.sum(Gr**2, axis=2))
#         G = -div(Gr[:,:,0] / d,Gr[:,:,1] / d )
#         # step
#         e = Phi(Xestim,Hreshape) - Yh[Band,:,:]
    
#         Xestim_prev = Xestim
    
#         Xestim = Xestim - tau*( Phi(e,Hreshape) + Lambda*G)
#         # energy
#         E[i] = 1/2*np.linalg.norm(e.flatten())**2 + Lambda*np.sum(d.flatten())
    
#         i = i+1
    
#     err[k] = snr(Xinit,Xestim)
    
#     if err[k] > snr(Xinit,Xbest):
#         Xbest = Xestim


