#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:26:35 2022

@author: e.akkouche
"""

import argparse
import matplotlib.pyplot as plt
from astropy.io import fits

import pca
import produce_HS_MS
from optimize import *
from tools import *
from CONSTANTS import *

import warnings
warnings.filterwarnings('ignore')

def main(simulate=False):

    #--------------------Generate image X and Yh--------------------
    if simulate:

        Yh, Ym = produce_HS_MS.main(DATA, DATA)
        # Yh = simulate_HS_MS.main(DATA, DATA,sigma2 = 0.1)

    #--------------------Choose subspace dimension------------------
    S, cumul_variance = pca.choose_subspace(HS_IM)

    # Affichage complet
    plt.semilogy(S);
    plt.ylabel('Valeurs propres')
    plt.xlabel('Indice de la composante principale')
    plt.show()
        
    #Affichage tronqué
    plt.semilogy(S[:Lacp]);
    plt.xticks(np.arange(len(S[:Lacp])), np.arange(1, len(S[:Lacp])+1))
    plt.ylabel('Valeurs propres')
    plt.xlabel('Indice de la composante principale')

    plt.savefig(SAVE+'eigenvalues_pca.png')

    #Afficher variance cumulee
    plt.plot(cumul_variance)
    plt.title('Cumulative Explained Variance -- PCA')
    plt.xlabel('Dimensions du sous-espace')

    #--------------------Total Variance PCA--------------------
    Yh = fits.getdata(HS_IM)
    Lacp = 10

    V, Z, mean = pca.pca_nirspec(Yh,Lacp)

    # # print('\n********** Saving PCA **********\n')

    # hdu = fits.PrimaryHDU(Z)
    # hdu.writeto(DATA+'Z.fits', overwrite=True)

    # hdu = fits.PrimaryHDU(V)
    # hdu.writeto(DATA+'V.fits', overwrite=True)

    # hdu = fits.PrimaryHDU(mean)
    # hdu.writeto(DATA+'mean.fits', overwrite=True)

    # print('\n********** Saving Done !! **********\n')

    #--------------------Check PCA--------------------
    # V = np.load(DATA+'V.npy')
    # Z = np.load(DATA+'Z.npy')
    Yh = fits.getdata(HS_IM)
    V = fits.getdata(V_acp)
    Z = fits.getdata(Z_acp)

    Xback, error = pca.check_pca(V, Z, mean, Yh)

    # hdu = fits.PrimaryHDU(np.real(np.fft.ifft2(error, axes=(1, 2), norm='ortho')))
    # hdu.writeto(DATA+'Difference.fits', overwrite=True)


    #-------------------- Fusion --------------------

    #Yh = fits.getdata(HS_IM)
    #V = fits.getdata(V_acp)
    #Z = fits.getdata(Z_acp)

    B,L,C, *_ = Z.shape
    img = np.random.rand(B, L, C)

    #--------------------Preprocessing--------------------
    Yfft, Zfft, H, T1, T2, maxH2, D = preprocessing(Yh, V, img, Lacp)

    Lband,Lin,Col = Yfft.shape

    Yfft = np.reshape(Yfft, (Lband, Lin*Col))

    H = np.reshape(H,(Lband, Lin*Col))

    Lband,Lin,Col = Zfft.shape
    Zfft = np.reshape(Zfft,(Lband, Lin*Col))

    #--------------------Gradient Descent--------------------
    Jmu = []
    Regmu = []
    J2mu = []
    loss = []

    mu_range = 10**(np.linspace(0,6,10))

    X_true = get_Xtrue(DATA)

    X_best = np.random.rand(*X_true.shape)

    # Band = 1000

    # X_true = Yh

    print("---------------- BEGIN --------------")

    t1 = time()

    for k, mu in enumerate(mu_range):
            
        print(f"----------------- mu : {mu} -----------------")

        Zoptim, J_Zoptim = GD(Yfft, V, Zfft, H, Lacp, T2, T1, D, mu, maxH2)

        Zifft = postprocess(Zoptim, Lacp)

        Jmu.append(np.linalg.norm(Yfft-np.dot(V,Zoptim)*H)**2)

        # J2mu.append(J_Zoptim)

        Regmu.append(np.linalg.norm((D[0]+D[1])*np.reshape(Zoptim,(Lacp,NR,NC)))**2)

        X_estim = recover_X(V, Zifft)

        loss.append(snr(X_true,X_estim))

        if loss[k] > snr(X_true,Xbest):

            Xbest = X_estim

            fname = SAVE2 + 'Zoptim_mu_' + str(k) + '.fits'

            save_Zoptim(Zifft, fname)

            fname = SAVE2 + 'J_Zoptim' + str(k)

            np.save(fname,J_Zoptim)

    fname = SAVE2+'SNR_sobolev'
    np.save(fname,loss)

    t2 = time()

    print('******************************************')
    print('******* TOTAL COMPUTATION TIME : '+str(np.round((t2-t1)/60))+'min '+str(np.round((t2-t1)%60))+'s.')


    plt.plot(mu_range,loss)
    plt.axis('tight')
    plt.xlabel('\lambda (échelle log)') 
    plt.ylabel('SNR')
    plt.show()

    plt.imshow(Xbest)
    plt.title('Variation totale SNR = '+str(round(np.max(loss),2))+'dB') 

    # for mu in mus:
        
    #     Zmu = fits.getdata(SAVE2+'_full_'+str(mu)+'z_opti.fits')
        
    #     Zmu = np.fft.fft2(tools.compute_symmpad_3d(Zmu,fact_pad),axes=(1, 2),norm='ortho')
        
    #     Zmu = np.reshape(Zmu,(Zmu.shape[0],Zmu.shape[1]*Zmu.shape[2]))
        
    #     Jmu.append(np.linalg.norm(Yfft-np.dot(V,Zmu)*H)**2)
        
    #     Regmu.append(np.linalg.norm((D[0]+D[1])*np.reshape(Zmu,(Lacp,NR,NC))))
        
    """--------------------Postprocess & saving--------------------"""
    # save_Zoptim(Zifft, SAVE)

    # Zoptim = fits.getdata(SAVE_PATH)

    # Band = 4900

    # Xestim = recover_X(V, Zifft)

    # Xtrue = get_Xtrue(DATA,Band)

    """--------------------Affichage--------------------"""
    # plt.imshow(Xestim[Band,:,:])
    # plt.show()
    # plt.imshow(Yh[Band,:,:])
    # plt.show()
    # plt.imshow(Xtrue)
    # plt.show()

    # np.save(SAVE+'J_Zoptim',J_Zoptim)
    # plt.semilogy(J)


    # plt.plot(J_Zoptim)
    # plt.yscale('log')
    # plt.title('Evolution de la fonction coût '+'(pas = ' +str(step)+')')
    # plt.xlabel('Itération')
    # plt.savefig(SAVE+'Evolution_fonction_coût_'+'pas_' +str(step))
    # plt.show()

    # JZe_7 = np.load(SAVE+'J_Zoptim_1e7.npy')
    # plt.plot(JZe_7,"-b", label="pas = 1e-7")
    # plt.plot(J_Zoptim, "-r", label="pas = 1e-6")
    # plt.legend(loc="upper right")
    # plt.title('Evolution de la fonction coût au cours des itérations')
    # plt.xlabel('Itération')
    # plt.savefig(SAVE+'Evolution_fonction_coût_2_pas')

    # plt.plot(Regmu,Jmu)
    # plt.title('Courbe en L')
    # plt.xlabel('Regul Sobolev(mu)')
    # plt.ylabel('Attache aux données(mu)')

    # plt.imshow(Xestim[Band,:,:])
    # plt.title('Sobolev SNR = '+str(round(snr(Xtrue,Xestim[Band,:,]),2))+'dB')  


    #Zmu = fits.getdata(SAVE2+'_full_4.641588833612778z_opti.fits')
    #Xmu = recover_X(V, Zmu)

    #plt.imshow(Xmu[Band,:,:])
    #plt.plot(Xmu[Band,63,:])

    #plt.plot(Xtrue[63,:], 'r',label = 'Référence') # plotting t, a separately 
    #plt.plot(Xmu[Band,63,:], 'b',label = 'Sobolev') # plotting t, b separately 
    #plt.legend(loc="upper right")


    #plt.plot()

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-simulate', dest='param_1', type=str)
    
    main(args.param_1)
    
if __name__ == "__main__":
    args_parser()
