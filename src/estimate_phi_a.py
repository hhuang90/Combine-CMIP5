import numpy as np

import state

def estimate_phi_a():

    # Update phiHa
    
    shape = (state.n+state.nuH)/2
    
    rate = np.sum((state.YHa-state.YH)**2) + state.nuH/state.phiH
    rate /= 2
    
    state.phiHa=np.random.gamma(shape,1/rate,1)

    # Update phiFa
    
    shape = (state.n+state.nuF)/2
    
    rate = np.sum((state.YFa-state.YF)**2) + state.nuF/state.phiF
    rate /= 2
    
    state.phiFa=np.random.gamma(shape,1/rate,1)
    
    return()
