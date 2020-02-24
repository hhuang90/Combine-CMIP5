#cython: boundscheck=False, wraparound=False, language_level=3
from scipy import linalg as spl

import random
import numpy as np
#cimport numpy as cnp

from tools import *

from cython.parallel cimport prange
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log

from scipy.linalg.cython_lapack cimport dpotrf, dpotrs, dpotri

import state


def estimate_gamma_m(bGammaHm=0.05,bGammaFm=0.05):

    cdef double[:,:] dist = state.dist 
    cdef int[:] Rm = state.RHm;
    
    cdef double[:] XHmrDiff = state.XHmrDiff.flatten()
    cdef double[:] XHmrDiff_original = state.XHmrDiff.flatten()
    
    cdef double[:] XFmrDiff = state.XFmrDiff.flatten()
    cdef double[:] XFmrDiff_original = state.XFmrDiff.flatten()
    
    cdef double[:] logGammaHmProbPart1 = state.logGammaHmProbPart1
    cdef double[:,:,:] covMatHm = state.covMatHm
    cdef double[:,:,:] invCovMatHm = state.invCovMatHm
    cdef double[:] acceptGammaHm = state.acceptGammaHm
    cdef double[:] rateHm = state.rateHm
    cdef double[:] phiHm = state.phiHm
    cdef double[:] gammaHm = state.gammaHm;
    
    cdef double[:] logGammaFmProbPart1 = state.logGammaFmProbPart1
    cdef double[:,:,:] covMatFm = state.covMatFm
    cdef double[:,:,:] invCovMatFm = state.invCovMatFm
    cdef double[:] acceptGammaFm = state.acceptGammaFm
    cdef double[:] rateFm = state.rateFm
    cdef double[:] phiFm = state.phiFm
    cdef double[:] gammaFm = state.gammaFm;
    
    
    cdef int M = state.M, n = state.n, m,i,j, info;
    cdef int nsq = n*n;
    cdef int offset = max(Rm) * n;
    
    cdef double[:] covMat_m_New = np.zeros(M*nsq);
    cdef double[:] covMat_m_New_save = np.zeros(M*nsq);

    cdef double[:] gamma_m_New = np.zeros(M);
    cdef double[:] logProbPart1 = np.zeros(M), logProb = np.zeros(M), logProbOld = np.zeros(M);
    cdef double[:] dice = np.zeros(M);
    cdef double cov
    
    # Update gammaHm
    for m in range(M):
        gamma_m_New_log=random.normalvariate(c_log(state.gammaHm[m]),bGammaHm)
        gamma_m_New[m]=c_exp(gamma_m_New_log)
        dice[m] = c_log(random.uniform(0,1))

    with nogil:
        for m in prange(M):
            if gamma_m_New[m] < 1e6 :
                for i in range(n):
                    for j in range(n):
                        cov = c_exp(-dist[i,j]/gamma_m_New[m]);
                        covMat_m_New[m*nsq+i*n+j] = cov
                        covMat_m_New_save[m*nsq+i*n+j] = cov
            
            
                dpotrf('L',&n,&covMat_m_New[m*nsq],&n,&info);

                for i in range(n):
                    logProbPart1[m] += c_log(covMat_m_New[m*nsq+i*n+i]);
                logProbPart1[m] *= -1 * Rm[m];

                dpotrs('L',&n,&Rm[m],&covMat_m_New[m*nsq],&n,&XHmrDiff[m*offset],&n,&info);

                for i in range(Rm[m]*n):
                    logProb[m] += XHmrDiff[m*offset+i]*XHmrDiff_original[m*offset+i]

                logProb[m] = logProbPart1[m] + logProb[m]*phiHm[m]*(-0.5)

                logProbOld[m] = logGammaHmProbPart1[m] + rateHm[m] * phiHm[m] + c_log(gammaHm[m]);

                if dice[m] < (logProb[m]-logProbOld[m]):
                    gammaHm[m] = gamma_m_New[m]

                    for i in range(n):
                        for j in range(n):
                            covMatHm[m,i,j] = covMat_m_New_save[m*nsq+i*n+j]

                    dpotri('L',&n,&covMat_m_New[m*nsq],&n,&info)
                    for i in range(n):
                        for j in range(i,n):
                            invCovMatHm[m,i,j] = covMat_m_New[m*nsq+i*n+j]
                        for j in range(i):
                            invCovMatHm[m,i,j] = invCovMatHm[m,j,i]

                    logGammaHmProbPart1[m] = logProbPart1[m]
                    acceptGammaHm[m] = acceptGammaHm[m] + 1
                
    # Update gammaFm
    for m in range(M):
        gamma_m_New_log=random.normalvariate(c_log(state.gammaFm[m]),bGammaFm)
        gamma_m_New[m]=c_exp(gamma_m_New_log)
        dice[m] = c_log(random.uniform(0,1))
    
    Rm = state.RFm;
    logProbPart1 = np.zeros(M)
    logProb = np.zeros(M)
    
    with nogil:
        for m in prange(M):
            if gamma_m_New[m] < 1e6 :
                for i in range(n):
                    for j in range(n):
                        cov = c_exp(-dist[i,j]/gamma_m_New[m]);
                        covMat_m_New[m*nsq+i*n+j] = cov
                        covMat_m_New_save[m*nsq+i*n+j] = cov


                dpotrf('L',&n,&covMat_m_New[m*nsq],&n,&info);

                for i in range(n):
                    logProbPart1[m] += c_log(covMat_m_New[m*nsq+i*n+i]);
                logProbPart1[m] *= -1 * Rm[m];

                dpotrs('L',&n,&Rm[m],&covMat_m_New[m*nsq],&n,&XFmrDiff[m*offset],&n,&info);

                for i in range(Rm[m]*n):
                    logProb[m] += XFmrDiff[m*offset+i]*XFmrDiff_original[m*offset+i]

                logProb[m] = logProbPart1[m] + logProb[m]*phiFm[m]*(-0.5)

                logProbOld[m] = logGammaFmProbPart1[m] + rateFm[m] * phiFm[m] + c_log(gammaFm[m]);

                if dice[m] < (logProb[m]-logProbOld[m]):
                    gammaFm[m] = gamma_m_New[m]

                    for i in range(n):
                        for j in range(n):
                            covMatFm[m,i,j] = covMat_m_New_save[m*nsq+i*n+j]

                    dpotri('L',&n,&covMat_m_New[m*nsq],&n,&info)
                    for i in range(n):
                        for j in range(i,n):
                            invCovMatFm[m,i,j] = covMat_m_New[m*nsq+i*n+j]
                        for j in range(i):
                            invCovMatFm[m,i,j] = invCovMatFm[m,j,i]

                    logGammaFmProbPart1[m] = logProbPart1[m]
                    acceptGammaFm[m] = acceptGammaFm[m] + 1
    
    return()