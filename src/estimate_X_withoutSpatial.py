import numpy as np
from tools import *

import state

def estimate_X_withoutSpatial():
    for m in range(state.M):
        if m==0 :
            tmpV=1/state.V[m,m] 
        else :
            tmpVector=spl.solve(state.V[0:m,0:m],state.V[0:m,m],assume_a="pos");
            tmpV=state.V[m,m]-state.V[m,0:m].dot(tmpVector);
            tmpV=1/tmpV;
            sumTmpVector=np.sum(tmpVector)
            
        # Update XFm
        B1 = state.phiFm[m] * state.RFm[m] * np.identity(state.n)
        A1B1 = state.phiFm[m] * state.sumXFmr[m]
        
        B2 = state.tauF * tmpV * np.identity(state.n)
        A2 = state.muF + state.beta * state.epsHm[:,m]
        
        if m>0 :
            tmpVectorXHm = tmpVector.dot(state.XHm[0:m,:])
            tmpUpdate = tmpVector.dot(state.XFm[0:m,:]) - state.beta * tmpVectorXHm + sumTmpVector * (state.beta * state.muH - state.muF)
            A2+=tmpUpdate
 
        AB = A1B1 + B2.dot(A2);
        B = B1 + B2;

        state.XFm[m,:]=genereateNormal(AB,B)
        
        # Update XHm 
        B3 = state.beta * state.beta * B2;
        A3 = state.beta*(state.XFm[m,:] - state.muF + state.beta * state.muH);
        if m>0 :
            A3 -= state.beta * tmpUpdate
        
        A3B3=B2.dot(A3)
        
        B1 = state.phiHm[m] * state.RHm[m] * np.identity(state.n)
        A1B1 = state.phiHm[m] * state.sumXHmr[m]
        
        B2 = state.tauH * tmpV * np.identity(state.n)
        A2 = state.muH.copy()
        
        if m>0 :
            A2 += tmpVectorXHm - sumTmpVector * state.muH

        AB = A1B1+B2.dot(A2)+A3B3
        B = B1+B2+B3;

        state.XHm[m,:]=genereateNormal(AB,B)

        state.epsHm[:,m] = state.XHm[m,:] - state.muH;
        state.epsFm[:,m] = state.XFm[m,:] - state.muF - state.beta * state.epsHm[:,m]
        
        state.XHmrDiff[m,:,:]=state.XHmr[m,:,:]-state.XHm[m,:]
        state.XFmrDiff[m,:,:]=state.XFmr[m,:,:]-state.XFm[m,:]

    return()