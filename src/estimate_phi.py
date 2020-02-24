import numpy as np

import state

def estimate_phi():

    # Update phiH
    
    shape = (state.nuH+state.nuH*state.M)/2 + 1e-3
    
    rate = np.sum(state.phiHm)+state.phiHa
    rate *= state.nuH/2
    rate += 1e-3
    
    state.phiH = np.random.gamma(shape,1/rate,1)
    state.phiH = 1/state.phiH
    
    # Update phiF
    
    shape = (state.nuF+state.nuF*state.M)/2 + 1e-3
    
    rate = np.sum(state.phiFm)+state.phiFa
    rate *= state.nuF/2
    rate += 1e-3
    
    state.phiF = np.random.gamma(shape,1/rate,1)
    state.phiF = 1/state.phiF
    
    return()
