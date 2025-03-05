#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:10:50 2022

@author: e.akkouche
"""
def process_PSF(VZ,sigma = 0):
    
    print('\n****** Processing PSF ******\n')
    
    Lband,Lin,Col = VZ.shape
    
    PSF = np.zeros((Lband,Lin,Col),dtype = complex)
    
    Hreshaped = np.zeros((Lin,Col),dtype = complex)
    
    H = get_spa_bandpsf_hs(1, sigma)
    
    L,C = H.shape
    
    Hamming = np.tile(np.hamming(C),(Lin,1))
    
    for k in range(Lband):
        
        H = get_spa_bandpsf_hs(k, sigma)
        
        Hshift = (np.fft.fftshift(H))
    
        ind = np.unravel_index(np.argmax(Hshift, axis=None), Hshift.shape)
        
        RLbnd = ind[0] - Lin//2
        RUbnd = ind[0] + Lin//2
        # CLbnd = ind[1] - C//2
        # CUbnd = ind[1] + C//2
        Nshift = (Col - C)//2
        
        Hreshaped[:,Nshift:(Nshift+C)] = Hshift[RLbnd:RUbnd,:] * Hamming

        PSF[k] = np.fft.fftshift(Hreshaped)
        
        Hreshaped *= 0
        
        print('k = '+ str(k))
    
    print('\n****** Processing PSF Done ******\n')
    
    return PSF



##################Le gradient ne converge pas dans cette version ########
# def process_PSF(Y,sigma = 0):
    
#     print('\n****** Processing PSF ******\n')
    
#     #VZ c'est Y
#     Nband,Lin,Col = Y.shape
    
#     PSF = np.zeros((Nband,Lin,Col),dtype = complex)
    
#     Hreshaped = np.zeros((Lin,Col),dtype = complex)

#     for k in range(Nband):
        
#         H = get_spa_bandpsf_hs(k, sigma)
        
#         Hifft = np.fft.ifft2(H)
        
#         Hifft = np.fft.ifftshift(Hifft)
    
#         ind = np.unravel_index(np.argmax(Hifft, axis=None), Hifft.shape)
        
#         RLbnd = ind[0] - Lin//2
#         RUbnd = ind[0] + Lin//2
#         Nshift = (Col - H.shape[1])//2
        
#         Hreshaped[:,Nshift:(Nshift+H.shape[1])] = Hifft[RLbnd:RUbnd,:] 
        
#         PSF[k] = np.fft.fftshift(Hreshaped)
        
#         Hreshaped *= 0
        
#         print('k = '+ str(k))
    
#     print('\n****** Processing PSF Done ******\n')
    
#     return np.fft.fft2(PSF,axes=(1, 2),norm='ortho')