#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code computes and saves fusion performance measures.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import fusion as fuse
from CONSTANTS import *
from astropy.io import fits
import matplotlib.image as mplmg

def compute_errors(v_true, v, z_true, z, mean, filename):
    # Cube dimensions
    lh, lint = v.shape
    px, py = z.shape[1:]
    # Compute sam
    print('Compute SAM ...')
    sam = np.zeros((px, py))
    for i in range(px):
        for j in range(py):
            spec_true = np.dot(v_true, z_true[:, i, j])
            spec = np.dot(v, z[:, i, j]) + mean
            sam[i, j] = compute_sam(spec_true, spec)
    print('Mean SAM = '+str(np.mean(sam)))
    print('Max SAM = '+str(np.max(sam)))

    # Compute global psnr
    print('Compute PSNR ...')
    max_true = search_max(v_true, z_true)
    norm_diff = compute_norm(v_true, v, z_true, z, mean)
    psnr = 10*np.log10(max_true**2*(norm_diff)**-1*lh*px*py)
    print('Global PSNR = '+str(psnr))

    print('Compute SNR ...')
    norm_true = compute_norm_true(v_true, z_true)
    norm_diff = compute_norm(v_true, v, z_true, z, mean)
    snr = 10*np.log10(norm_true*(norm_diff)**-1)
    print('Global SNR = '+str(snr))

    # Compute uiqi
    print('Compute SSIM ...')
    ssim = np.zeros((lh))
    # Reshape
    z = np.reshape(z, (lint, px*py))
    z_true = np.reshape(z_true, (4, px*py))
    for l in range(lh):
        band = np.dot(v[l], z)+mean[l]
        band_true = np.dot(v_true[l], z_true)
        c1 = (0.01*np.max(band_true))**2
        c2 = (0.03*np.max(band_true))**2
        x_bar, y_bar, sig_x, sig_y, sig_xy = compute_uiqi(band_true, band, px*py)
        ssim[l] = (2*sig_xy+c1)*(2*x_bar*y_bar+c2)*(np.dot((sig_x**2+sig_y**2+c1),(x_bar**2+y_bar**2+c2).T))**-1
    print('Mean SSIM = '+str(np.mean(1-ssim)))

    # Save in an hdf5 file
    print('Saving errors ...')
    f = h5py.File(filename, 'w')
    f.create_dataset('sam', data=sam)
    f.create_dataset('psnr', data=psnr)
    f.create_dataset('uiqi', data=uiqi)
    f.close()


def compute_sam(s, s_hat):
    return np.arccos(np.dot(s, s_hat.T)*(np.linalg.norm(s)*np.linalg.norm(s_hat))**-1)


def search_max(v, z):
    max_ = 0
    lh, lint = v.shape
    px, py = z.shape[1:]
    z = np.reshape(z, (lint, px*py))
    for l in range(lh):
        band = np.dot(v[l], z)
        max_band = np.max(band)
        if max_band > max_:
            max_ = max_band
    return max_


def compute_norm(v_true, v, z_true, z, mean):
    lh, lint = v.shape
    lh, lint_ = v_true.shape
    px, py = z.shape[1:]
    z = np.reshape(z, (lint, px*py))
    z_true = np.reshape(z_true, (lint_, px*py))
    norm = 0
    for l in range(lh):
        band = np.dot(v[l], z) + mean[l]
        band_true = np.dot(v_true[l], z_true)
        norm += np.sum((band-band_true)**2)
    return norm


def compute_norm_true(v_true, z_true):
    lh, lint = v_true.shape
    px, py = z_true.shape[1:]
    z_true = np.reshape(z_true, (lint, px*py))
    norm = 0
    for l in range(lh):
        band_true = np.dot(v_true[l], z_true)
        norm += np.sum(band_true**2)
    return norm


def compute_uiqi(band_true, band, N):
    x_bar = N**-1*np.sum(band_true)
    y_bar = N**-1*np.sum(band)
    sig_x = np.sqrt(N**-1*np.sum((band_true-x_bar)**2))
    sig_y = np.sqrt(N**-1*np.sum((band-y_bar)**2))
    sig_xy = N**-1*np.sum((band_true-x_bar)*(band-y_bar))
    return x_bar, y_bar, sig_x, sig_y, sig_xy


def snr(x, y):
    """
    snr - signal to noise ratio

     snr = 20*log10( norm(x) / norm(x-y) )

       x is the original clean signal (reference).
       y is the denoised signal.
    """

    return 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(x - y))

############# Plotting routines #############


def plot_errors(mu, filename, shape):
    N = len(mu)
    sam = np.zeros((N, shape[1], shape[2]))
    psnr = np.zeros((N))
    uiqi = np.zeros((N, shape[0]))
    for i in range(N):
        filename_ = filename+'_full_'+str(mu[i])+'_lowsnr'
        # Load errors
        f = h5py.File(filename_, 'r')
        sam[i] = np.array(f['sam'])
        psnr[i] = np.array(f['psnr'])
        uiqi[i] = np.array(f['uiqi'])
        print ('** MU = '+str(mu[i]))
        print ('** SAM = '+str(np.mean(sam[i])))
        print ('** PSNR = '+str(psnr[i]))
        print ('** UIQI = '+str(np.mean(uiqi[i])))
        f.close()

    plt.figure('Mean SAM vs mu')
    plt.subplot(211)
    plt.plot(mu[10:], np.mean(sam[10:], axis=(1, 2)), 's-')
    plt.subplot(212)
    plt.semilogx(mu[10:], np.mean(sam[10:], axis=(1, 2)), 's-')
    plt.savefig(SAVE+'sam.png')
    plt.figure('Mean UIQI vs mu')
    plt.subplot(211)
    plt.plot(mu[10:], np.mean(uiqi[10:], axis=1), 's-')
    plt.subplot(212)
    plt.semilogx(mu[10:], np.mean(uiqi[10:], axis=1), 's-')
    plt.savefig(SAVE+'uiqi.png')
    plt.figure('Global PSNR vs mu')
    plt.subplot(211)
    plt.plot(mu[10:], psnr[10:], 's-')
    plt.subplot(212)
    plt.semilogx(mu[10:], psnr[10:], 's-')
    plt.savefig(SAVE+'psnr.png')
    plt.show()


def plot_uiqi_norm(mu, filename, v_true, z_true):
    tabwave = fits.getdata(DATA+'tabwave.fits')[:, 0] 

    filename_1 = filename+'_full_'+str(mu)+'_'
    filename_2 = filename+'_brovey_'
    filename_3 = filename+'_rfuse_'
    filename_4 = filename+'_ms_'+str(1e-5)+'_'
    filename_5 = filename+'_hs_'+str(2e-5)+'_'
    filename_6 = filename+'_baseline_'
    
    f = h5py.File(filename_1, 'r')
    uiqi1 = np.array(f['uiqi'])
    f.close()
    f = h5py.File(filename_2, 'r')
    uiqi2 = np.array(f['uiqi'])
    f.close()
    f = h5py.File(filename_3, 'r')
    uiqi3 = np.array(f['uiqi'])
    f.close()
    f = h5py.File(filename_4, 'r')
    uiqi4 = np.array(f['uiqi'])
    f.close()
    f = h5py.File(filename_5, 'r')
    uiqi5 = np.array(f['uiqi'])
    f.close()
    f = h5py.File(filename_6, 'r')
    uiqi6 = np.array(f['uiqi'])
    f.close()

    spec = np.dot(v_true[4200:], z_true[:,50,685])

    # font = {'family': 'serif',
    #     'weight': 'normal',
    #     'size': 10,
    #     }

    # fig, ax = plt.subplots(nrows=6, ncols=1)
    # im = ax[0].semilogy(tabwave[:1000], uiqi1[:1000])
    # ax[0].set_ylim(0.9, 1.0)
    # ax[0].set_title('Symmetric fusion; mean = '+str(np.mean(uiqi1)), font)
    # im = ax[1].semilogy(tabwave[:1000], uiqi2[:1000])
    # ax[1].set_ylim(0.9, 1.0)
    # ax[1].set_title('Brovey; mean = '+str(np.mean(uiqi2)), font)
    # im = ax[2].semilogy(tabwave[:1000], uiqi3[:1000])
    # ax[2].set_ylim(0.9, 1.0)
    # ax[2].set_title('R-FUSE; mean = '+str(np.mean(uiqi3)), font)
    # im = ax[3].semilogy(tabwave[:1000], uiqi4[:1000])
    # ax[3].set_ylim(0.9, 1.0)
    # ax[3].set_title('PCA + MS reconstruction; mean = '+str(np.mean(uiqi4)), font)
    # im = ax[4].semilogy(tabwave[:1000], uiqi5[:1000])
    # ax[4].set_ylim(0.9, 1.0)
    # ax[4].set_title('PCA + HS super-resolution; mean = '+str(np.mean(uiqi5)), font)
    # im = ax[5].semilogy(tabwave[:1000], uiqi6[:1000])
    # ax[5].set_ylim(0.9, 1.0)
    # ax[5].set_title('Baseline; mean = '+str(np.mean(uiqi6)), font)

    # fig, ax = plt.subplots(nrows=6, ncols=1)
    # im = ax[0].semilogy(tabwave[4000:], uiqi1[4000:])
    # ax[0].set_ylim(0.9, 1.0)
    # ax[0].set_yticklabels([])
    # ax[0].set_title('Symmetric fusion; mean = '+str(np.mean(uiqi1)), font)
    # im = ax[1].semilogy(tabwave[4000:], uiqi2[4000:])
    # ax[1].set_ylim(0.9, 1.0)
    # ax[1].set_title('Brovey; mean = '+str(np.mean(uiqi2)), font)
    # im = ax[2].semilogy(tabwave[4000:], uiqi3[4000:])
    # ax[2].set_ylim(0.9, 1.0)
    # ax[2].set_title('R-FUSE; mean = '+str(np.mean(uiqi3)), font)
    # im = ax[3].semilogy(tabwave[4000:], uiqi4[4000:])
    # ax[3].set_ylim(0.9, 1.0)
    # ax[3].set_title('PCA + MS reconstruction; mean = '+str(np.mean(uiqi4)), font)
    # im = ax[4].semilogy(tabwave[4000:], uiqi5[4000:])
    # ax[4].set_ylim(0.9, 1.0)
    # ax[4].set_title('PCA + HS super-resolution; mean = '+str(np.mean(uiqi5)), font)
    # im = ax[5].semilogy(tabwave[4000:], uiqi6[4000:])
    # ax[5].set_ylim(0.9, 1.0)
    # ax[5].set_title('Baseline; mean = '+str(np.mean(uiqi6)), font)

    # fig, ax = plt.subplots(nrows=6, ncols=1)
    # im = ax[0].semilogy(tabwave, uiqi1)
    # ax[0].set_ylim(0.9, 1.0)
    # ax[0].set_yticklabels([])
    # ax[0].set_title('Symmetric fusion; mean = '+str(np.mean(uiqi1)), font)
    # im = ax[1].semilogy(tabwave, uiqi2)
    # ax[1].set_ylim(0.9, 1.0)
    # ax[1].set_title('Brovey; mean = '+str(np.mean(uiqi2)), font)
    # im = ax[2].semilogy(tabwave, uiqi3)
    # ax[2].set_ylim(0.9, 1.0)
    # ax[2].set_title('R-FUSE; mean = '+str(np.mean(uiqi3)), font)
    # im = ax[3].semilogy(tabwave, uiqi4)
    # ax[3].set_ylim(0.9, 1.0)
    # ax[3].set_title('PCA + MS reconstruction; mean = '+str(np.mean(uiqi4)), font)
    # im = ax[4].semilogy(tabwave, uiqi5)
    # ax[4].set_ylim(0.9, 1.0)
    # ax[4].set_title('PCA + HS super-resolution; mean = '+str(np.mean(uiqi5)), font)
    # im = ax[5].semilogy(tabwave, uiqi6)
    # ax[5].set_ylim(0.9, 1.0)
    # ax[5].set_title('Baseline; mean = '+str(np.mean(uiqi6)), font)
    # # plt.subplots_adjust(wspace=0, hspace=0)

    # plt.figure()
    # plt.legend()

    # spikes = np.array([2.06798, 2.0745, 2.07625, 2.08249, 2.08666, 2.09468, 2.09642, 2.09992, 2.10062, 2.11361,
    #      2.11221, 2.12173, 2.12775, 2.13343, 2.15416, 2.16135, 2.16459, 2.16604, 2.17254, 2.18162, 2.18416, 2.20134,
    #      2.20501, 2.21053, 2.22309, 2.22532, 2.24431, 2.24768, 2.25368, 2.26668, 2.28698, 2.29998, 2.30727,
    #      2.30842, 2.30996, 2.32232, 2.33707, 2.33941, 2.34448, 2.34526])

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    # ax1.set_xlabel('Wavelength (microns)', fontsize=15)
    # ax1.set_ylabel('CSSIM', color=color, fontsize=15)
    ax1.semilogy(tabwave[4200:-1],1-uiqi5[4200:-1], label='Référence', linewidth=1)
    ax1.semilogy(tabwave[4200:-1],1-uiqi2[4200:-1], label='Brovey', linewidth=1)
    ax1.semilogy(tabwave[4200:-1],1-uiqi1[4200:-1], label='Sobolev', linewidth=1)
    ax1.semilogy(tabwave[4200:-1],1-uiqi3[4200:-1], label='Sobolev pondéré', linewidth=1)
    ax1.semilogy(tabwave[4200:-1],1-uiqi4[4200:-1], label='Représentation par patchs', linewidth=1)
    # ax1.semilogy(tabwave[4200:-1],1-uiqi1[4200:-1], label='Proposed', linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.set_ylim(5e-5, 2e-1)
    

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:grey'
    # ax2.set_ylabel('Intensité (mJy.arcsec$^{-2}$)', color=color, fontsize=15)  # we already handled the x-label with ax1
    ax2.semilogy(tabwave[4200:-1], spec[:-1], color=color, linewidth=0.5)
    ax2.fill_between(tabwave[4200:-1], spec[:-1], color=color, alpha=0.15)
    # for i in range(spikes.shape[0]):
        # ax2.axvline(x=spikes[i], linestyle='dashed', linewidth=0.25, color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax2.set_ylim(7.5, 1e5)

    # ax1.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.3), fontsize=10)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.xlim(2.068, 2.35)
    plt.xticks(fontsize=10)
    plt.savefig('cssimvswl.pdf')

    # plt.figure()
    # plt.semilogy(tabwave[:500],uiqi1[:500], label='Symmetric fusion')
    # plt.semilogy(tabwave[:500],uiqi2[:500], label='Brovey')
    # plt.semilogy(tabwave[:500],uiqi3[:500], label='R-Fuse')
    # plt.semilogy(tabwave[:500],uiqi4[:500], label='MS-only')
    # plt.semilogy(tabwave[:500],uiqi5[:500], label='HS-only')
    # plt.semilogy(tabwave[:500],uiqi6[:500], label='Baseline')
    # plt.legend()

    bins = np.linspace(0,1,1000)

    plt.figure()
    hist5 = plt.hist((1-uiqi5).flatten(), bins=bins, alpha=0.5, label='Baseline')
    hist2 = plt.hist((1-uiqi2).flatten(), bins=bins, alpha=0.5, label='Brovey')
    hist1 = plt.hist((1-uiqi1).flatten(), bins=bins, alpha=0.5, label='R-Fuse')
    hist3 = plt.hist((1-uiqi3).flatten(), bins=bins, alpha=0.5, label='MS-only')
    hist4 = plt.hist((1-uiqi4).flatten(), bins=bins, alpha=0.5, label='HS-only')
    # hist1 = plt.hist(sam1.flatten(), bins=bins, alpha=0.5, label='Proposed')
    # plt.legend()
    plt.close()

    plt.figure()
    plt.semilogy(np.cumsum(hist5[0]), hist5[1][:-1], label='Référence', linewidth=1)
    plt.semilogy(np.cumsum(hist2[0]), hist2[1][:-1], label='Brovey', linewidth=1)
    plt.semilogy(np.cumsum(hist1[0]), hist1[1][:-1], label='Sobolev', linewidth=1)
    plt.semilogy(np.cumsum(hist3[0]), hist2[1][:-1], label='Sobolev pondéré', linewidth=1)
    plt.semilogy(np.cumsum(hist4[0]), hist4[1][:-1], label='Représentation par patchs', linewidth=1)
    # plt.semilogy(np.cumsum(hist1[0]), hist1[1][:-1], label='Proposed')
    plt.legend()

    plt.savefig('cssim_histcumul.pdf')



def plot_error_maps(mu, filename):
    # filename_1 = filename+'_full_lowsnr'
    # filename_2 = filename+'brovey_lowsnr'
    # filename_3 = filename+'rfuse_lowsnr'
    # filename_4 = filename+'_ms_lowsnr'
    # filename_5 = filename+'_hs_lowsnr'
    # filename_6 = filename+'_baseline_lowsnr'
    filename_1 = filename+'_full_'+str(mu)+'_'
    filename_2 = filename+'_brovey_'
    filename_3 = filename+'_rfuse_'
    filename_4 = filename+'_ms_'+str(1e-5)+'_'
    filename_5 = filename+'_hs_'+str(3e-5)+'_'
    filename_6 = filename+'_baseline_'
    
    f = h5py.File(filename_1, 'r')
    sam1 = np.array(f['sam'])
    f.close()
    

    f = h5py.File(filename_2, 'r')
    sam2 = np.array(f['sam'])
    f.close()
    

    f = h5py.File(filename_3, 'r')
    sam3 = np.array(f['sam'])
    f.close()
    

    f = h5py.File(filename_4, 'r')
    sam4 = np.array(f['sam'])
    f.close()
    

    f = h5py.File(filename_5, 'r')
    sam5 = np.array(f['sam'])
    f.close()
    
    
    f = h5py.File(filename_6, 'r')
    sam6 = np.array(f['sam'])
    f.close()

    vmin = np.min(sam2)
    vmax = np.max(sam3)
    
    mplmg.imsave(filename_1+'_.pdf', sam1, vmin=vmin, vmax=vmax)
    mplmg.imsave(filename_2+'_.pdf', sam2, vmin=vmin, vmax=vmax)
    mplmg.imsave(filename_3+'_.pdf', sam3, vmin=vmin, vmax=vmax)
    mplmg.imsave(filename_4+'_.pdf', sam4, vmin=vmin, vmax=vmax)
    mplmg.imsave(filename_5+'_.pdf', sam5, vmin=vmin, vmax=vmax)
    mplmg.imsave(filename_6+'_.pdf', sam6, vmin=vmin, vmax=vmax)

    # font = {'family': 'serif',
    #     'weight': 'normal',
    #     'size': 10,
    #     }

    # plt.figure('SAM error map')
    # fig, ax = plt.subplots(nrows=6, ncols=1)
    # im = ax[0].imshow(sam1, vmin=vmin, vmax=vmax)
    # ax[0].set_axis_off()
    # ax[0].set_title('Symmetric fusion', font)
    # im = ax[1].imshow(sam2, vmin=vmin, vmax=vmax)
    # ax[1].set_axis_off()
    # ax[1].set_title('Brovey', font)
    # im = ax[2].imshow(sam3, vmin=vmin, vmax=vmax)
    # ax[2].set_axis_off()
    # ax[2].set_title('R-FUSE', font)
    # im = ax[3].imshow(sam4, vmin=vmin, vmax=vmax)
    # ax[3].set_axis_off()
    # ax[3].set_title('PCA + MS reconstruction', font)
    # im = ax[4].imshow(sam5, vmin=vmin, vmax=vmax)
    # ax[4].set_axis_off()
    # ax[4].set_title('PCA + HS super-resolution', font)
    # im = ax[5].imshow(sam6, vmin=vmin, vmax=vmax)
    # ax[5].set_axis_off()
    # ax[5].set_title('Baseline', font)

    # # fig.tight_layout()
    # fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
    # plt.savefig(filename_1+'sam_lowsnr.eps')
    # plt.close()

    # fig, ax = plt.figure()
    # im = ax.imshow(sam1, vmin=vmin, vmax=vmax)

    bins = np.linspace(0,1,1000)

    plt.figure()
    hist5 = plt.hist(sam5.flatten(), bins=bins, alpha=0.5, label='Baseline')
    hist2 = plt.hist(sam2.flatten(), bins=bins, alpha=0.5, label='Brovey')
    hist1 = plt.hist(sam1.flatten(), bins=bins, alpha=0.5, label='R-Fuse')
    hist3 = plt.hist(sam3.flatten(), bins=bins, alpha=0.5, label='MS-only')
    hist4 = plt.hist(sam4.flatten(), bins=bins, alpha=0.5, label='HS-only')
    # hist1 = plt.hist(sam1.flatten(), bins=bins, alpha=0.5, label='Proposed')
    # plt.legend()
    plt.close()

    plt.figure()
    plt.semilogy(np.cumsum(hist5[0]), hist5[1][:-1], label='Référence', linewidth=1)
    plt.semilogy(np.cumsum(hist2[0]), hist2[1][:-1], label='Brovey', linewidth=1)
    plt.semilogy(np.cumsum(hist1[0]), hist1[1][:-1], label='Sobolev', linewidth=1)
    plt.semilogy(np.cumsum(hist3[0]), hist2[1][:-1], label='Sobolev pondéré', linewidth=1)
    plt.semilogy(np.cumsum(hist4[0]), hist4[1][:-1], label='Représentation par patchs', linewidth=1)
    # plt.semilogy(np.cumsum(hist1[0]), hist1[1][:-1], label='Proposed')
    plt.legend()

    plt.savefig(SAVE+'sam_histcumul.pdf')
