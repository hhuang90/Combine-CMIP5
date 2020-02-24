import numpy as np

import state

def estimate_tau():
    
    # Update tauH
    
    shape = (state.M*state.n+state.n)/2+1e-3;
    
    rate = state.epsYH.dot(state.invCovMatH.dot(state.epsYH))
    rate += np.sum(state.invCovMatH.dot(state.epsHm).dot(state.invV)*state.epsHm)
    rate /= 2
    state.rateH = -rate
    rate += 1e-3;

    state.tauH=np.random.gamma(shape,1/rate,1)

    # Update tauF
    
    rate = state.epsYF.dot(state.invCovMatF.dot(state.epsYF))
    rate += np.sum(state.invCovMatF.dot(state.epsFm).dot(state.invV)*state.epsFm);
    rate /= 2
    state.rateF = -rate
    rate +=1e-3;

    state.tauF=np.random.gamma(shape,1/rate,1)
    
    return()