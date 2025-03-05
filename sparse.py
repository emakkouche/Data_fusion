# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numpy.matlib
import scipy.sparse as sp
# from astropy.io import fits
# from skimage.transform import resize
from time import time
# import acp_v2 as pca
import tools
from sparse_preprocess import set_inputs, preprocess_D
from CONSTANTS import *

import warnings
warnings.filterwarnings('ignore')

"""
__________________________MS PART -- MATRIX A_____________________________
"""


def M(i, j, phv, nr, nc):
    res = np.zeros(nr*nc, dtype=np.complex)
    nf = phv.shape[0]
    for m in range(nf):
        res += np.conj(phv[m, j])*phv[m, i]
    return res


def PHV(lacp, P, V, nr, nc):
    nf = P.shape[0]
    lh = V.shape[0]
    res = np.zeros((nf, lacp, nr*nc), dtype=np.complex)
#    print(' *** PHV computation ***')
    for m in range(nf):
        for i in range(lacp):
            sum_h = np.zeros(nr*nc, dtype=np.complex)
            for l in range(lh):
                sum_h += tools.get_h_band(l)*P[m, l]*V[l, i]
            res[m, i] = sum_h
    return res


def Anc(lacp, nr, nc, P, V):
    t1 = time()
    row = np.arange(nr*nc)
    row = np.matlib.repmat(row, 1, lacp**2)[0]
    for i in range(lacp):
        row[i*nr*nc*lacp:(i+1)*nr*nc*lacp] += i*nr*nc
    col = np.arange(nr*nc*lacp)
    col = np.matlib.repmat(col, 1, lacp)[0]

    phv = PHV(lacp, P, V, nr, nc)
    mat = np.reshape(np.reshape(np.arange(lacp**2), (lacp, lacp)).T, lacp**2)
    data = np.zeros((lacp**2, nr*nc), dtype=np.complex)
    for i in range(lacp):
        # print('i='+str(i))
        for j in range(lacp-i):
            # print('j='+str(j+i))
            temp = M(i, j+i, phv, nr, nc)
            if j == 0:
                data[i*(lacp+1)] = temp
            else:
                index = j+i*(lacp+1)
                data[mat[index]] = np.conj(temp)
                data[index] = temp
    data = np.reshape(data, (lacp**2*nr*nc))
    t2 = time()
    print('Anc computation time : '+str(t2-t1)+'s.')
    return sp.coo_matrix((data, (row, col)), shape=(lacp*nr*nc, lacp*nr*nc), dtype=np.complex), phv


def Anc_bis(lacp, nr, nc, P, V):
    """
    NO PSF
    """
    t1 = time()
    row = np.arange(nr*nc)
    row = np.matlib.repmat(row, 1, lacp**2)[0]
    for i in range(lacp):
        row[i*nr*nc*lacp:(i+1)*nr*nc*lacp] += i*nr*nc
    col = np.arange(nr*nc*lacp)
    col = np.matlib.repmat(col, 1, lacp)[0]

    # data = np.zeros((lacp**2, nr*nc), dtype=np.complex)
    LmV = np.dot(np.dot(V.T, P.T), np.dot(P, V))
    data = np.array([])
    for i in range(4):
        for j in range(4):
            data = np.concatenate((data, numpy.matlib.repmat(LmV[i, j], 5, 1)[:, 0]))
    t2 = time()
    print('Anc computation time : '+str(t2-t1)+'s.')
    return sp.coo_matrix((data, (row, col)), shape=(lacp*nr*nc, lacp*nr*nc), dtype=np.complex)


###############################################################################
"""
__________________________HS PART -- MATRIX A____________________________
"""


def nTn_sparse(nr, nc, d):
    n1 = sp.identity(nc//d)
    n2_ = n1.copy()
    for i in range(d-1):
        n2_ = sp.hstack((n2_, n1))
    n2 = n2_.copy()
    for i in range(d-1):
        n2 = sp.vstack((n2, n2_))
    n3 = n2.copy()
    for i in range(nr//d-1):
        n3 = sp.block_diag((n3, n2))
    n4_ = n3.copy()
    for i in range(d-1):
        n4_ = sp.hstack((n4_, n3))
    n4 = n4_.copy()
    for i in range(d-1):
        n4 = sp.vstack((n4, n4_))
    return n4


def C(i, j, V, row, col, nr, nc, d):
    res = np.zeros(nr*nc*d**2, dtype=np.complex)
    lh = len(V)
    for m in range(lh):
        g = tools.get_g_band(m)
        gntng = d**(-2)*np.conj(g[row])*g[col]
        res += (V[m, i]*gntng*V[m, j])
    return res


def Ans(lacp, nr, nc, d, V, Q):
    t1 = time()
    ntn = nTn_sparse(nr, nc, d)
    V = np.dot(np.diag(Q), V)
    row = np.matlib.repmat(ntn.row, 1, lacp**2)[0]
    for i in range(lacp):
        row[i*nr*nc*lacp*d**2:(i+1)*nr*nc*lacp*d**2] += i*nr*nc
    col = np.zeros(nr*nc*d**2*lacp)
    for i in range(lacp):
        col[i*nr*nc*d**2:(i+1)*nr*nc*d**2] = ntn.col+i*nr*nc
    col = np.matlib.repmat(col, 1, lacp)[0]
    mat = np.reshape(np.reshape(np.arange(lacp**2), (lacp, lacp)).T, lacp**2)
    data = np.zeros((lacp**2, nr*nc*d**2), dtype=np.complex)
    for i in range(lacp):
        # print('i='+str(i))
        for j in range(lacp-i):
            # print('j='+str(j+i))
            temp = C(i, j+i, V, ntn.row, ntn.col, nr, nc, d)
            if j == 0:
                data[i*(lacp+1)] = temp
            else:
                index = j+i*(lacp+1)
                data[mat[index]] = np.conj(temp)
                data[index] = temp
    data = np.reshape(data, np.prod(data.shape))
    t2 = time()
    print('Ans computation time : '+str(t2-t1)+'s.')
    return sp.coo_matrix((data, (row, col)), shape=(lacp*nr*nc, lacp*nr*nc), dtype=np.complex)


###############################################################################
"""
__________________________Regulatization PART FOR MATRIX A_____________________
"""

def Areg(lacp, nr, nc, D):
    row = np.arange(lacp*nc*nr)
    col = row.copy()
    data = np.reshape(np.matlib.repmat(D, lacp, 1), (lacp*nr*nc))
    return sp.coo_matrix((data, (row, col)), shape=(lacp*nr*nc, lacp*nr*nc), dtype=np.complex)


###############################################################################


def build_b(Ym, Yh, P, Q, V, nr, nc, lacp, sig2):
    Ym = np.reshape(Ym, (len(Ym)//(nr*nc), nr*nc))
    Yh = np.reshape(Yh, (len(Yh)//(nr*nc//d**2), nr*nc//d**2))
    lh = Yh.shape[0]
    lm = Ym.shape[0]
#    bnc=np.zeros((lacp,nr*nc))
    bnc = np.dot(P.T, Ym)
    # ############# Control procedure  #############
    # if (np.sum(np.isnan(bnc))>=1) :
    #     print('NANs in LmT Ym :'+str(np.sum(np.isnan(bnc))))
    # ###############################################
    for l in range(lh):
        bnc[l] = tools.get_h_band(l, mode='adj')*bnc[l]
    # ############# Control procedure  #############
    # if (np.sum(np.isnan(bnc))>=1) :
    #     print('NANs in MT LmT Ym :'+str(np.sum(np.isnan(bnc))))
    # ###############################################
    bnc = np.dot(V.T, bnc)
    # ############# Control procedure  #############
    # if (np.sum(np.isnan(bnc))>=1) :
    #     print('NANs in VT MT LmT Ym :'+str(np.sum(np.isnan(bnc))))
    # ###############################################

    Yh_ = tools.aliasing_adj(Yh, (Yh.shape[0], nr, nc))
    # ############# Control procedure  #############
    # if (np.sum(np.isnan(Yh_))>=1) :
    #     print('NANs in ST Yh :'+str(np.sum(np.isnan(Yh_))))
    # ###############################################
    for l in range(lh):
        Yh_[l] *= tools.get_g_band(l, mode='adj')
    # ############# Control procedure  #############
    # if (np.sum(np.isnan(Yh_))>=1) :
    #     print('NANs in  HT ST Yh :'+str(np.sum(np.isnan(Yh_))))
    # ###############################################
    bns = np.dot(np.dot(np.diag(Q), V).T, Yh_)
    # ############# Control procedure  #############
    # if (np.sum(np.isnan(bns))>=1) :
    #     print('NANs in VT LhT HT ST Ym :'+str(np.sum(np.isnan(bns))))
    # ###############################################

    # print('-(sig2[0]*nr*nc*lm)^(-1) : '+str(-(sig2[0]*nr*nc*lm)**(-1)))
    # print('-(sig2[1]*(nr//d)*(nc//d)*lh)^(-1) : '+str(-(sig2[1]*nr//d*nc//d*lh)**(-1)))

    sbnc = -1/(sig2[0]*nr*nc*lm)
    sbns = -1/(sig2[1]*(nr//d)*(nc//d)*lh)

    return np.reshape(snbc*bnc, np.prod(bnc.shape)), np.reshape(sbns*bns, np.prod(bns.shape))


###############################################################################

def build_c(Ym, Yh, sig2, lm, lh, nr, nc):
    sbnc = 1/(sig2[0]*nr*nc*lm)
    sbns = 1/(sig2[1]*(nr//d)*(nc//d)*lh)
    return 0.5*sbnc*np.dot(np.conj(Ym).T, Ym), 0.5*sbns*np.dot(np.conj(Yh).T, Yh)

###############################################################################


def get_linearsyst_reginf(lacp, MS_IM, HS_IM, filename=SAVE2):
    Ym, Yh, P, Q, V, Z, D, Wd, sig2 = set_inputs(lacp, nr, nc, MS_IM, HS_IM, -1)
    lm, lh = P.shape
    anc, phv = Anc(lacp, nr, nc, P, V)
    sbnc = 1/(sig2[0]*nr*nc*lm)
    sbns = 1/(sig2[1]*(nr//d)*(nc//d)*lh)
    Am = sbnc*anc
    Ah = sbns*Ans(lacp, nr, nc, d, V, Q)
    b1, b2 = build_b(Ym, Yh, P, Q, V, nr, nc, lacp, sig2)
    # ############# Control procedure  #############
    # print('NANs in bm :'+str(np.sum(np.isnan(b1))))
    # print('NANs in bh :'+str(np.sum(np.isnan(b2))))
    # ###############################################
    c1, c2 = build_c(Ym, Yh, sig2, lm, lh, nr, nc)
    return Am, Ah, b1, b2, c1, c2, Z, D, Wd
    
    return 0;