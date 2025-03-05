#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:12:10 2022

@author: e.akkouche
"""

"""
This code implents forward models of the NIRCam imager and the NIRSpec IFU embedded in the JWST as described in the references above.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from tools import compute_symmpad_3d, _centered, compute_symmpad
from skimage.transform import resize
from scipy.signal import convolve2d
# from pandeia.engine.instrument_factory import InstrumentFactory
from CONSTANTS import *
from tools import reshape_psf

import warnings
warnings.filterwarnings('ignore')


############ Produce Multispectral Image with NIRCam Forward Model ############


##### If necessary, use pandeia engine to construct the MS spectral degradation operator.
def get_pce(instrument, mode, config, wave):
    # Get photon conversion efficiency of the instrument, from pandeia engine
    obsmode = {
               'instrument': instrument,
               'mode': mode,
               'filter': config['filter'],
               'aperture': config['aperture'],
               'disperser': config['disperser']
               }

    conf = {'instrument': obsmode}

    i = InstrumentFactory(config=conf)
    pce = i.get_total_eff(wave)
    return pce


def get_filters(tabwave, channel, exp_time=EXPTIME_NC):
    # Computes spectral degradation operator Lm of the instrument with unit convestion (mjy/arcsec^2 to e-)

    # Instrument config.
    instrument = 'nircam'
    # Short wavelengths
    fname_list = ['f115w', 'f140m', 'f150w', 'f150w2', 'f162m', 'f164n', 'f182m', 'f187n', 'f200w', 'f210m', 'f212n']
    mode = 'sw_imaging'
    aper = 'sw'
    # Long wavelengths
    # fname_list=['f250m','f277w','f300m','f322w2','f323n','f335m','f356w','f360m','f405n','f410m','f430m','f444w','f460m','f466n','f470n','f480m']
    # mode='lw_imaging'
    # aper='lw'

    # Conversion constant
    C = 205000
    # Calculate Lm
    i = 0
    Lm = np.zeros((len(fname_list), len(tabwave)))
    for f in fname_list:
        config = {'filter': f, 'aperture': aper, 'disperser': 'none'}
        Lm[i] = get_pce(instrument, mode, config, tabwave)
        i += 1
    delt_lamb = np.linspace(tabwave[1]-tabwave[0], tabwave[-1]-tabwave[-2], num=len(tabwave))
    Lm = C*Lm*tabwave**(-1)*1.5091905*delt_lamb*exp_time*FLUXCONV_NC
    return Lm
##### End pandeia engine


##### In Ref 2, section V : Robustness with respect to model mismatch.
# Add white gaussian noise to the spectral degradation operator Lm
def calc_sigma(Lm, snr=50):
    return np.linalg.norm(Lm)**2*10**(-0.1*snr)*(1/np.prod(Lm.shape))


def addnoise_filt(Lm, snr=50):
    sigma = calc_sigma(Lm, snr)
    l, m = Lm.shape
    return Lm + np.sqrt(sigma)*np.random.randn(l, m)
#####


def get_spa_psf_ms(sigma=0):
    # Get the whole spatial Point Spread Function matrix. It has to be calculated in the Fourier domain and saved in advance with webbpsf.
    h_ = fits.getdata(PSF+'G_fft.fits')
    k, l, m, n = h_.shape
    h = h_[0]+h_[1]*1.j
    return np.reshape(h, (l, m, n))


def get_spa_bandpsf_ms(band, sigma=0):
    # Get a spatial Point Spread Function at wavelength number 'band'. It has to be calculated in the Fourier domain and saved in advance with webbpsf.
    h_ = fits.getdata(PSF+'G_fft.fits')[:, band]
    k, m, n = h_.shape
    h = h_[0]+h_[1]*1.j
    return np.reshape(h, (m, n))


def produce_MS_nir(M, A, tabwave, channel, snr=50, sigma=0):
    """
    Use this function only if the product MA, i.e. the whole scene with full spectral and spatial resolutions, can be stored in memory.
    """
    # Unit conversion of M : mJy/arcsec^2 to mJy/pix_nc

    # Spectral degradation operator with pandeia engine
    # Lm = addnoise_filt(get_filters(tabwave, channel), snr)
    # Lm = get_filters(tabwave, channel)

    # Spectral degradation operator using the fits file
    Lm = fits.getdata(DATA+'Lm.fits')

    # Number of MS bands
    lm = Lm.shape[0]

    # Get spatial PSFs
    H = get_spa_psf_ms(sigma)

    # Shapes
    lh, m = M.shape
    n, p, q = A.shape

    # Compute the whole scene with full spectral and spatial resolutions
    X = compute_symmpad_3d(np.reshape(np.dot(M, np.reshape(A, (n, p*q))), (lh, p, q)), fact_pad)

    # Convolve X with spectrally variant PSFs and degrade spectral resolution with Lm
    Y = np.dot(Lm, np.reshape(_centered(np.fft.ifft2(H*np.fft.fft2(X, norm='ortho'), norm='ortho')[:, :-2, :-2], (lh, p, q)), (lh, p*q)))

    # Return MS image

    return np.reshape(np.real(Y), (lm, p, q))


def produce_MS_nir_bis(M, A, tabwave, channel, snr=50, sigma=0):
    """
    Use this function if the product MA the whole scene with full spectral and spatial resolutions, cannot be stored in memory.
    """
    # Unit conversion of M : mJy/arcsec^2 to mJy/pix_nc
    
    # Spectral degradation operator with pandeia engine
    # Lm = addnoise_filt(get_filters(tabwave, channel), snr)
    # Lm = get_filters(tabwave, channel)

    # Spectral degradation operator using the fits file
    Lm = fits.getdata(DATA+'Lm.fits')

    # Number of MS bands
    lm = Lm.shape[0]

    # Shapes
    lh, m = M.shape
    n, p, q = A.shape

    # Initialize MS image
    Y = np.zeros((lm, p, q))
    # Compute MS bands
    for i in range(lh):
        # Compute the scene at a wavelength
        X = compute_symmpad(np.reshape(np.dot(M[i], np.reshape(A, (n, p*q))), (p, q)), fact_pad)
        # Get PSF at this band
        H = get_spa_bandpsf_ms(i, sigma)
        # Get spectral degradation at this wavelength
        filt = np.reshape(Lm[:, i], (lm, 1))
        # Convolve and spectrally degrade
        temp = np.reshape(_centered(np.fft.ifft2(H*np.fft.fft2(X, norm='ortho'), norm='ortho')[:-2, :-2], (p, q)), (p*q))
        # Update MS image
        Y += np.real(np.reshape(filt*temp, (lm, p, q)))
    #return Y



########### Produce Hyperspectral Image with NIRSpec Forward Model ############


##### In Ref 2, section V : Robustness with respect to model mismatch.
# Add white gaussian noise to the spectral degradation operator Lh
def addnoise_filt_hs(Lh, snr=50):
    sigma = calc_sigma(Lh, snr)
    return Lh + np.sqrt(sigma)*np.random.randn(Lh.shape[0])
########


def get_spec_psf():
    # Get spectral PSF (1-D gaussian blur)
    return fits.getdata(DATA+'PSF_spec.fits')


def get_spa_psf_hs(sigma=0):
    # Get the whole spatial Point Spread Function matrix. It has to be calculated in the Fourier domain and saved in advance with webbpsf.
    g_ = fits.getdata(PSF+'M_fft.fits')
    k, l, m, n = g_.shape
    g = g_[0]+g_[1]*1.j
    return np.reshape(g, (l, m, n))


def get_spa_bandpsf_hs(band, sigma=0):
    # Get a spatial Point Spread Function at wavelength number 'band'. It has to be calculated in the Fourier domain and saved in advance with webbpsf.
    g_ = fits.getdata(PSF+'M_fft.fits')[:, band]
    k, m, n = g_.shape
    g = g_[0]+g_[1]*1.j
    return np.reshape(g, (m, n))


def subsample(X, d=0.31):
    # 3D Subsampling function with Nircam/Nirspec pixel size ratio.
    l, m, n = X.shape
    return resize(X, np.round((l, m*d, n*d)))


def subsample2d(X, d=0.31):
    # 2D Subsampling function with Nircam/Nirspec pixel size ratio.
    m, n = X.shape
    return resize(X, np.round((m*d, n*d)))


def produce_HS_nir(M, A, tabwave, fname, dname, snr=50, sigma=0):
    """
    Use this function only if the product MA, i.e. the whole scene with full spectral and spatial resolutions, can be stored in memory.
    M in mjy/arcsec**2
    !!!!! TABWAVE IN MICRONS !!!!!
    """
    # -- Ref 2, section V : Robustness
    # Lh = addnoise_filt_hs(fits.getdata(DATA+'Lh.fits'), snr)
    Lh = fits.getdata(DATA+'Lh.fits')

    # Get spectral PSF
    L = get_spec_psf()
    
    # Get spatial PSFs
    H = get_spa_psf_hs(sigma)

    # Shapes
    lh, m = M.shape
    n, p, q = A.shape

    # Compute the whole scene with full spectral and spatial resolutions
    X = compute_symmpad_3d(np.reshape(np.dot(M, np.reshape(A, (n, p*q))), (lh, p, q)), fact_pad)
    # Spatial degradation by convolving with PSFs (in the Fourier domain)
    Y = np.real(np.reshape(_centered(np.fft.ifft2(H*np.fft.fft2(X, norm='ortho'), norm='ortho')[:, :-2, :-2], (lh, p, q)), (lh, p*q)))
    # Convolve spectra with spectral PSF, multiply with spectral throughput and subsample.
    Y = subsample(np.reshape((Lh*np.apply_along_axis(lambda x: np.convolve(x, L, mode='same'), axis=0, arr=Y).T).T, (lh, p, q)))
    return Y


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
        # Shapes for subsampled image
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
        
    print('\n********** Simulation Done  !!**********\n')

    return Y

############################## Noise Model ####################################

def mult_noise(Y):
    # Compute multiplicative noise ~ N(0,sqrt(|Y|))
    L, M, N = Y.shape
    noise = np.sqrt(abs(Y))*np.random.randn(L, M, N)
    
    print('Apply multiplicative noise done') 
    return noise


def add_noise_nocorr(Y, sigma2):
    # Compute additive noise with no spatial correlation
    L, M, N = Y.shape
    noise = np.sqrt(sigma2)*np.random.randn(L, M, N)
    
    print('Apply nocorr noise done') 
    return noise


def add_noise_corr_ns(Y, sigma2=SIG2READ_NS, nframes=NFRAMES_NS):
    # Compute additive noise with spatial correlation for HS image from NIRSpec
    L, M, N = Y.shape
    # Additive Gaussian Noise
    noise = add_noise_nocorr(Y, sigma2)
    # Get correlation matrix
    if nframes > 18:
        nframes = 18
    cormat = fits.getdata(CORR)[nframes]
    cormat = cormat/np.sum(cormat)
    noisecorr = np.zeros(Y.shape)
    # Apply additive correlated noise as described in [1]
    for i in range(N//30):
        temp = compute_symmpad(np.reshape(noise[:, :, 30*i:30*(i+1)], (L, M*30)), 10)
        noisecorr[:, :, 30*i:30*(i+1)] = np.reshape(_centered(convolve2d(temp, cormat, mode='same'), (L, M*30)), (L, M, 30))
    n = N-30*(i+1)
    temp = compute_symmpad(np.reshape(noise[:, :, 30*(i+1):], (L, M*n)), min(10, n))
    noisecorr[:, :, 30*(i+1):] = np.reshape(_centered(convolve2d(temp, cormat, mode='same'), (L, M*n)), (L, M, n))
    
    print('Apply correlated ns noise done') 
    return noisecorr


def add_noise_corr_nc(Y, sigma2=SIG2READ_NC, nframes=NFRAMES_NC):
    # Compute additive noise with spatial correlation for MS image from NIRCam
    L, M, N = Y.shape
    # Additive Gaussian Noise
    noise = add_noise_nocorr(Y, sigma2)
    # Get correlation matrix
    if nframes > 18:
        nframes = 18
    cormat = fits.getdata(CORR)[nframes]
    cormat = cormat/np.sum(cormat)
    noisecorr = np.zeros(Y.shape)
    # Apply additive correlated noise as described in [1]
    for i in range(L):
        temp = compute_symmpad(noise[i, :, :], 10)[:-2, :-2]
        noisecorr[i, :, :] = _centered(convolve2d(temp, cormat, mode='same'), (M, N))
        
    print('Apply correlated nc noise done') 
    return noisecorr


def apply_noise(Y, instrument):
    # Apply noise to either HS or MS image.
    #Y_noise = Y+mult_noise(Y)
    if 'nirspec' in instrument:
        Y_noise = Y + add_noise_corr_ns(Y)
    elif 'nircam' in instrument:
        Y_noise += add_noise_corr_nc(Y)
    return Y_noise


############################## Simulation and saving ####################################

def main(file_data, file_save, snr=50, sigma=0): #Modifié
    ##### Get the 3D scene with full resolution
    M = fits.getdata(file_data+'M_1.fits').T
    A = fits.getdata(file_data+'A.fits')
    #####

    ##### Get specification : wavelength table, instrument spec.
    tabwave = fits.getdata(file_data+'tabwave.fits')[:, 0]
    channel = 'short'
    fname = 'na'
    dname = 'na'

    ##### Compute images
    print('Simulating HS and MS images ...')
    Y = produce_HS_nir_bis(M, A, tabwave, fname, dname, snr, sigma)
    Yh_highsnr = Y + add_noise_nocorr(Y,sigma2=SIG2READ_NS)
    Ym_highsnr = apply_noise(produce_MS_nir_bis(M, A, tabwave, channel, snr, sigma), 'nircam') 
    

    ##### Save images
    print('Saving HS and MS images ...')
    # Save high snr images
    hdu = fits.PrimaryHDU(Yh_highsnr)
    hdu.writeto(HS_IM, overwrite=True)
    hdu = fits.PrimaryHDU(Ym_highsnr)
    hdu.writeto(MS_IM, overwrite=True)
    
    return Yh_highsnr, Ym_highsnr

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_data', dest='param_1', type=str)
    parser.add_argument('-file_save', dest='param_2', type=str)
    parser.add_argument('-snr', dest='param_3', type=float)
    parser.add_argument('-sigma', dest='param_4', type=float)
    args = parser.parse_args()
    
    main(args.param_1, args.param_2, args.param_3, args.param_4)
    
if __name__ == "__main__":
    
    args_parser()