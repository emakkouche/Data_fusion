#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 09:41:13 2022

@author: e.akkouche
"""

from CONSTANTS import *

Dv = Z[:,1:,:] - Z[:,:-1,:]
Dh = Z[:,:,1:] - Z[:,:,:-1]

lin_pad = np.zeros((Lacp,1,NC))
col_pad = np.zeros((Lacp,NR,1))

Dv = np.concatenate((Dv,lin_pad),axis=1)
Dh = np.concatenate((Dh,col_pad),axis=2)

DvT = Z[:,:-2,:] - Z[:,1:-1,:]
DhT = Z[:,:,:-2] - Z[:,:,1:-1]

lin_pad = -Z[:,0,:]
col_pad = -Z[:,:,0]

DvT = np.insert(DvT,0,lin_pad,axis = 1)
DhT = np.insert(DhT,0,col_pad,axis = 2)

B,Col = Z[:,-2,:].shape
lin_pad = np.reshape(Z[:,-2,:],(B,1,Col))

B,Lin = Z[:,:,-2].shape 
col_pad = np.reshape(Z[:,:,-2],(B,Lin,1))

DvT = np.concatenate((DvT,lin_pad),axis = 1)
DhT = np.concatenate((DhT,col_pad),axis = 2)
