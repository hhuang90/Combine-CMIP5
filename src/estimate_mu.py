import numpy as np
from tools import *

import state

def estimate_mu():
    
    sumSumInvV=np.sum(state.invV)
    sumInvV=np.sum(state.invV,axis=0)
    
    # Update muF
    B1 = state.invCovMatF * state.tauF
    A1B1= B1.dot(state.YF - state.beta*(state.YH-state.muH))
    
    B2 = B1*sumSumInvV
    A2B2=B2.dot(state.beta*state.muH+sumInvV.dot(state.XFm-state.XHm*state.beta)/sumSumInvV)
    
    
    B = B1+B2+np.identity(state.n)*1e-6;    
    AB=A1B1+A2B2
  
    state.muF=genereateNormal(AB,B);
    
    # Update muH
    B4 = B2*state.beta*state.beta
    A4B4 = state.beta*B2.dot(state.muF-sumInvV.dot(state.XFm-state.XHm*state.beta)/sumSumInvV)
    
    B1 = state.invCovMatH*state.tauH
    A1B1 = B1.dot(state.YH)
    
    B2 = state.invCovMatF*state.beta*state.beta*state.tauF
    A2B2 = state.invCovMatF.dot(state.muF+state.beta*state.YH-state.YF)*state.beta*state.tauF
    
    B3 = B1*sumSumInvV
    A3B3 = B1.dot(sumInvV.dot(state.XHm))
    
    AB=A1B1+A2B2+A3B3+A4B4
    
    B=B1+B2+B3+B4+np.identity(state.n)*1e-6;
    
    state.muH=genereateNormal(AB,B)
    
    state.epsHm=state.XHm-state.muH;
    state.epsFm_=state.XFm-state.muF;
    state.epsFm=state.epsFm_-state.beta*state.epsHm;

    state.epsHm=state.epsHm.transpose()
    state.epsFm_=state.epsFm_.transpose()
    state.epsFm=state.epsFm.transpose()

    state.epsYH=state.YH-state.muH;
    state.epsYF_=state.YF-state.muF;
    state.epsYF=state.epsYF_-state.beta*state.epsYH;
    
    return()