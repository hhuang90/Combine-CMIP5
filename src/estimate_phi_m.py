import numpy as np
from scipy import linalg as spl

import state

def estimate_phi_m():
    
    for m in range(state.M):

        # Update phiHm
        shape = state.n * state.RHm[m] + state.nuH
        shape /= 2
        
        rate = np.sum(state.XHmrDiff[m,0:state.RHm[m]].dot(state.invCovMatHm[m])*state.XHmrDiff[m,0:state.RHm[m]])
        state.rateHm[m] = -rate/2
        rate += state.nuH/state.phiH
        rate /= 2
        
        state.phiHm[m]=np.random.gamma(shape,1/rate,1)
        
        # Update phiFm
        shape = state.n*state.RFm[m]+state.nuF
        shape /= 2
        
        rate = np.sum(state.XFmrDiff[m,0:state.RFm[m]].dot(state.invCovMatFm[m])*state.XFmrDiff[m,0:state.RFm[m]])
        state.rateFm[m] = -rate/2
        rate += state.nuF/state.phiF
        rate /= 2

        state.phiFm[m]=np.random.gamma(shape,1/rate,1)
        
    return()
        
    
        