import numpy as np
from scipy.special import gammaln

import state

def posteriorNu(nu,psi,psiA,psiM,M):
    
    val1 = np.log(nu/psi/2)*(nu*M/2)-M*gammaln(nu/2)
    
    val2 = np.sum((nu/2-1)*np.log(psiM)-nu*psiM/psi/2)
    
    val3 = val1 / M
    
    val4 = (nu/2-1)*np.log(psiA)-nu*psiA/psi/2;

    val5 = (1e-3-1)*np.log(nu)-1e-3*nu
    
    return(val1+val2+val3+val4+val5)


def estimate_nu():
    
    # Update nuH
    
    bNuH = 0.5;
    nuHNew = np.random.normal(np.log(state.nuH),bNuH,1)
    nuHNew = np.exp(nuHNew)
    
    posteriorNew = posteriorNu(nuHNew,state.phiH,state.phiHa,state.phiHm,state.M) + np.log(nuHNew)
    posteriorOld = posteriorNu(state.nuH,state.phiH,state.phiHa,state.phiHm,state.M) + np.log(state.nuH)
    
    if np.log(np.random.uniform()) < posteriorNew-posteriorOld :
        state.nuH = nuHNew;
        state.posteriorNuH = posteriorNew;
        state.acceptNuH += 1;

    # Update nuF
    
    bNuF = 0.5;
    nuFNew = np.random.normal(np.log(state.nuF),bNuF,1)
    nuFNew = np.exp(nuFNew)
    
    posteriorNew = posteriorNu(nuFNew,state.phiF,state.phiFa,state.phiFm,state.M) + np.log(nuFNew)
    posteriorOld = posteriorNu(state.nuF,state.phiF,state.phiFa,state.phiFm,state.M) + np.log(state.nuF)
    
    if np.log(np.random.uniform()) < posteriorNew-posteriorOld :
        state.nuF = nuFNew;
        state.posteriorNuF = posteriorNew;
        state.acceptNuF += 1;
    
    return()
