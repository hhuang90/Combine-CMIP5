from scipy import linalg as spl
import numpy as np
from tools import *

import state

def posteriorGammaNew(covMat,epsX,epsY,tau,M,invV):
    logDSign, logD = np.linalg.slogdet(covMat)
    logProbPart1 = logDSign*logD*(M+1)*(-0.5);

    tmpYnX = spl.solve(covMat,np.column_stack((epsX,epsY)),assume_a="pos");
    tmpY = tmpYnX[:,M];
    tmpX = tmpYnX[:,0:M]

    logProbPart2 = tmpY.dot(epsY);
    
    logProbPart3 = tmpX.dot(invV)
    logProbPart3 = np.sum(logProbPart3*epsX)

    logProb = logProbPart1+(logProbPart2+logProbPart3)*tau*(-0.5)
    
    return(logProb, logProbPart1)


def estimate_gamma(bGammaH=0.05,bGammaF=0.05):
    # Update gammaH
    
    gammaHNew=np.random.normal(np.log(state.gammaH),bGammaH,1)
    gammaHNew=np.exp(gammaHNew)

    if gammaHNew < 1e6 :
        
        covMatHNew = np.exp(-state.dist/gammaHNew)

        logProbNew, logProbNewPart1 = posteriorGammaNew(covMatHNew,state.epsHm,state.epsYH,state.tauH,state.M,state.invV)
        logProbNew += np.log(gammaHNew)

        logProbOld = state.logGammaHProbPart1 + state.rateH*state.tauH + np.log(state.gammaH);

        if np.log(np.random.uniform()) < logProbNew-logProbOld:
            state.gammaH = gammaHNew
            state.covMatH = covMatHNew
            state.invCovMatH = my_inv(state.covMatH);
            state.logGammaHProbPart1 = logProbNewPart1 
            state.acceptGammaH += 1

    
    # Update gammaF
    
    gammaFNew=np.random.normal(np.log(state.gammaF),bGammaF,1)
    gammaFNew=np.exp(gammaFNew)

    if gammaFNew < 1e6 :
        
        covMatFNew = np.exp(-state.dist/gammaFNew)

        logProbNew, logProbNewPart1 = posteriorGammaNew(covMatFNew,state.epsFm,state.epsYF,state.tauF,state.M,state.invV)
        logProbNew += np.log(gammaFNew)

        logProbOld = state.logGammaFProbPart1 + state.rateF*state.tauF + np.log(state.gammaF);

        if np.log(np.random.uniform()) < logProbNew-logProbOld:
            state.gammaF = gammaFNew
            state.covMatF = covMatFNew
            state.invCovMatF = my_inv(state.covMatF);
            state.logGammaFProbPart1 = logProbNewPart1 
            state.acceptGammaF += 1
    
    
    return()