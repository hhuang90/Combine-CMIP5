import scipy.stats as stats
import numpy as np
from tools import *

import state

def estimate_V():
    
    # Update V
    
    degreeFreedom=2*state.n+state.M+state.df+1
    
    Q1 = state.df*state.VPrior
    Q2 = state.epsHm.transpose().dot(state.invCovMatH).dot(state.epsHm)*state.tauH
    Q3 = state.epsFm.transpose().dot(state.invCovMatF).dot(state.epsFm)*state.tauF

    Q = Q1 + Q2 + Q3
    
    state.V = stats.invwishart.rvs(degreeFreedom,Q)
    
    state.V /= state.V[0,0]

    state.invV = my_inv(state.V);
    
    return()
