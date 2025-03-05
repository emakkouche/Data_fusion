#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings
# import scipy.signal as sg
import numpy as np
from astropy.io import fits
from skimage.transform import resize
from time import time
from produce_HS_MS import get_pce
import acp_v2 as pca
import tools
import cmath
# from scipy.ndimage import gaussian_filter
from CONSTANTS import *
warnings.filterwarnings('ignore')


##################### GET DATASET ############################################

def get_hsms(MS_IM, HS_IM):
    # Get HS and MS images from path in CONSTANTS.py
    Ync = fits.getdata(MS_IM)
    Yns = fits.getdata(HS_IM)
    l, m, n = Ync.shape
    # Reshape HS image in order to set the ratio between MS and HS spatial pixel sizes to an integer
    Yns = resize(Yns, (Yns.shape[0], m//d, n//d), order=3, mode='symmetric')*(d**2*FLUXCONV_NC)
    # Compute the observed noise variance in each image.
    sig2_hs = np.mean(Yns)
    sig2_ms = np.mean(Ync)
    # Get wavelengths
    tabwave = fits.getdata(DATA+'tabwave.fits')[:, 0]
    return Ync, Yns, tabwave, [sig2_ms, sig2_hs]


########################## INITIALISATION #####################################

def initialisation(Ync, Yns, Lh, n_comp, sigma, filename=SAVE2):
    # Perform PCA on the HS image
    print(' PCA on the HS image : ')
    t1 = time()
    # Reshapes
    l, m, n = Yns.shape
    Yns_ = np.reshape(np.dot(np.diag(Lh**-1), np.reshape(Yns, (l, m*n))), (l, m, n))
    # PCA
    V, Z, mean = pca.pca_nirspec(Yns_, n_comp)
    # Save PCA
    save_V_mean(V, mean, sigma, filename)
    # Upsampling with bi-cubic interpolation
    Z = resize(Z, (n_comp, Ync.shape[1], Ync.shape[2]), order=3, mode='symmetric')
    t2 = time()
    print(str(t2-t1)+'s.')
    print('------------------------------------------------------------------')
    # Symmetric boundaries conditions
    Z = np.fft.fft2(tools.compute_symmpad_3d(Z, fact_pad), norm='ortho')
    # Save PCA
    save_Z(Z, filename)
    # Mean spectrum in the Fourier domain
    N = Z.shape[1]*Z.shape[2]
    mean_ = np.zeros((mean.shape[0], N))
    mean_[:, 0] = mean*np.sqrt(N)
    return V, Z, mean_


def save_V_mean(V, mean, sigma, filename=SAVE2):
    # Save PCA
    hdu = fits.PrimaryHDU(V)
    hdu.writeto(filename+'V_1_mjy.fits', overwrite=True)
    hdu = fits.PrimaryHDU(mean)
    hdu.writeto(filename+'mean_1_mjy.fits', overwrite=True)


def save_Z(Z, filename):
    # Save PCA
    hdu = fits.PrimaryHDU(np.real(np.fft.ifft2(Z, axes=(1, 2), norm='ortho')))
    hdu.writeto(filename+'Z_0.fits', overwrite=True)


################################### PREPROCESSING #############################

def preprocess(Ync, Yns, Lm, Lh, mean):
    # Pre-processing of the HS and MS images and operators
    print(' Operators and data preprocessing : ')
    t1 = time()

    # Symmetric boundaries condition and FFT on HS and MS images
    Ync = tools.compute_symmpad_3d(Ync, fact_pad//d+1)
    Ync = np.fft.fft2(Ync, axes=(1, 2), norm='ortho')
    Yns = np.fft.fft2(tools.compute_symmpad_3d(Yns, fact_pad//d+1)[:, :-2, :-2], axes=(1, 2), norm='ortho')
    # Shapes
    L, M, N = Ync.shape
    l, m, n = Yns.shape
    # Substract the mean image : Lm M (mean) and Lh H (mean) S to each image
    mean_ = mean.copy()
    mean_[:, 0] = mean[:, 0]*tools.get_h_mean()
    Ync = np.reshape(Ync, (L, M*N))-np.dot(Lm, mean_)
    mean_ = mean.copy()
    mean_[:, 0] = mean[:, 0]*tools.get_g_mean()
    Yns = np.reshape(Yns, (l, m*n))-np.dot(np.diag(Lh), tools.aliasing(mean_, (l, M, N)))
    t2 = time()
    print(str(t2-t1)+'s.')
    print('------------------------------------------------------------------')
    # return Ync,Yns,W
    return Ync, Yns

################################ Pre-process Regul. ####################################


def preprocess_D(sh):
    m, n = sh
    Dx = np.zeros(sh)
    Dy = np.zeros(sh)

    Dx[0, 0] = 1
    Dx[0, 1] = -1
    Dy[0, 0] = 1
    Dy[1, 0] = -1

    Dx = np.fft.fft2(Dx)
    Dy = np.fft.fft2(Dy)

    return (Dx, Dy)


def get_weigth(Ync, V, Lm, Lh, D, nr, nc):
    #### Computes weights to be associated to the spatial regularization.

    # Compute Ym D
    Ymdx = np.fft.ifft2(Ync.reshape(Ync.shape[0], nr, nc)*D[0], axes=(1, 2))
    Ymdy = np.fft.ifft2(Ync.reshape(Ync.shape[0], nr, nc)*D[1], axes=(1, 2))

    # Trace Ym
    trymx = np.linalg.norm(Ymdx, ord=2, axis=0)
    trymy = np.linalg.norm(Ymdy, ord=2, axis=0)

    # Trace Lm
    trlm = np.trace(np.dot(Lm, Lm.T))
    sigchap2x = trymx/trlm
    sigchap2y = trymy/trlm

    vtv = np.diag(np.dot(V.T, V))
    Sigmazx = np.zeros((vtv.shape[0], sigchap2x.shape[0], sigchap2x.shape[1]))
    Sigmazy = np.zeros((vtv.shape[0], sigchap2y.shape[0], sigchap2y.shape[1]))
    for i in range(vtv.shape[0]):
        Sigmazx[i] = sigchap2x*vtv[i]
        Sigmazy[i] = sigchap2y*vtv[i]

    epsilon = 1e-2
    return (0.5*(1/(Sigmazx+epsilon)+1/(Sigmazy+epsilon)), 0.5*(1/(Sigmazx+epsilon)+1/(Sigmazy+epsilon)))


################################ Main pre-processing function ####################################

def set_inputs(lacp, nr, nc, MS_IM, HS_IM, sigma, filename=SAVE2):
    #### Preprocessing of HS and MS images, operators and regularization

    # Get images
    Ym, Yh, tabwave, sig2 = get_hsms(MS_IM, HS_IM)
    # print('NANs in Ym and Yh :'str(np.sum(np.isnan(Ym)))+' ; '+str(np.sum(np.isnan(Yh))))

    # Get spectral operators
    Lm = fits.getdata(DATA+'Lm.fits')
    # print('NANs in Lm :'+str(np.sum(np.isnan(Lm))))
    Lh = fits.getdata(DATA+'Lh.fits')
    # print('NANs in Lh :'+str(np.sum(np.isnan(Lh))))

    # PCA on HS image
    V, Z, mean = initialisation(Ym, Yh, Lh, lacp, sigma, filename)
    # print('NANs in V,Z and mean :'+str(np.sum(np.isnan(V)))+' ; '+str(np.sum(np.isnan(Z)))+' ; '+str(np.sum(np.isnan(mean))))

    # Pre-process images
    Ym, Yh = preprocess(Ym, Yh, Lm, Lh, mean)

    # Pre-process regularization and get weights
    D = preprocess_D((nr, nc))
    Wd = get_weigth(Ym, V, Lm, Lh, D, nr, nc)
    Wd = (Wd-np.min(Wd))/np.max(Wd-np.min(Wd))
    # Wd = np.ones(Wd.shape)

    return np.reshape(Ym, np.prod(Ym.shape)), np.reshape(Yh, np.prod(Yh.shape)), Lm, Lh, V, np.reshape(Z, np.prod(Z.shape)), D, Wd, sig2
