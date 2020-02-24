import numpy as np

import state

def estimate_tauW():
    # Update tauW
    shape=state.N * state.n/2+1e-3
    rate=np.sum((state.W-state.YHa)*(state.W-state.YHa))/2+1e-3
    
    state.tauW=np.random.gamma(shape,1/rate,1)

    return()
