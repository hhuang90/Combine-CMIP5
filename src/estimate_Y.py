import numpy as np
from tools import *

import state

def estimate_Y():
    # Update YH
    A1B1 = state.YHa * state.phiHa;
    A2B2 = state.invCovMatH.dot(state.muH) * state.tauH;
    A3B3 = state.beta * state.invCovMatF.dot(state.YF + state.beta*state.muH - state.muF) * state.tauF

    B1 = np.identity(state.n) * state.phiHa;
    B2 = state.tauH * state.invCovMatH;
    B3 = state.tauF * state.beta * state.invCovMatF * state.beta;

    AB = A1B1+A2B2+A3B3;
    B = B1+B2+B3;

    state.YH = genereateNormal(AB,B)

    # Update YHa
    A1B1 = state.YH * state.phiHa;
    A2B2 = np.sum(state.W,0) * state.tauW;

    B1 = state.phiHa;
    B2 = state.W.shape[0] * state.tauW;

    var = 1/(B1+B2)
    mean = var*(A1B1+A2B2)
    state.YHa = np.random.normal(mean,np.sqrt(var))

    # Update YF
    A1B1 = state.YFa * state.phiFa;
    A2B2 = state.invCovMatF.dot(state.beta*(state.YH-state.muH)+state.muF) * state.tauF;

    B1 = np.identity(state.n) * state.phiFa;
    B2 = state.tauF * state.invCovMatF;

    AB = A1B1+A2B2;
    B = B1+B2;
    state.YF = genereateNormal(AB,B)

    # Update YFa
    state.YFa=np.random.normal(state.YF,np.sqrt(1/state.phiFa))

    return()