import numpy as np

import state

def estimate_beta():
    
    # Update beta
    
    v1 = state.epsYH.dot(state.invCovMatF.dot(state.epsYF_))
    v2 = np.sum(state.invCovMatF.dot(state.epsFm_).dot(state.invV)*state.epsHm)
    
    v = (v1+v2)*state.tauF
    
    Q1 = state.epsYH.dot(state.invCovMatF.dot(state.epsYH))
    Q2 = np.sum(state.invCovMatF.dot(state.epsHm).dot(state.invV)*state.epsHm)

    Q = (Q1+Q2)*state.tauF+1e-6;

    state.beta=np.random.normal(v/Q,np.sqrt(1/Q),1);

    state.epsFm=state.epsFm_-state.beta*state.epsHm;
    state.epsYF=state.epsYF_-state.beta*state.epsYH;

    return()