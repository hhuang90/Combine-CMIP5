from scipy import linalg as spl
import numpy as np
import dbm.dumb
import shelve

import state
import result

def genereateNormal(AB,B):
    #generate N(B^{-1}*AB,B^{-1})
    U=spl.cholesky(B)
    tmp=np.random.normal(size=AB.shape[0])
    
    mean=spl.solve_triangular(U,AB,trans="T")
    mean=spl.solve_triangular(U,mean)
    
    return(spl.solve_triangular(U,tmp)+mean)

def my_inv(x):
    tmp, _  = spl.lapack.dpotrf(x,False,False)
    inv, _ = spl.lapack.dpotri(tmp)
    inv = np.triu(inv) + np.triu(inv, k=1).T
    return(inv)

def read_application_data(dataFile):
    db = dbm.dumb.open(dataFile,'r')
    my_shelf = shelve.Shelf(db)
    
    state.M = my_shelf['M']
    state.n = my_shelf['n']

    state.W = my_shelf['W']
    state.RHm = my_shelf['RHm']
    state.RFm = my_shelf['RFm']
    
    state.XHmr = my_shelf['XHmr']
    state.XFmr = my_shelf['XFmr']
    
    state.dist = my_shelf['dist']
    
    db.close()
    

def read_data(dataFile):
    db = dbm.dumb.open(dataFile,'r')
    my_shelf = shelve.Shelf(db)
    
    state.M = my_shelf['M']
    state.n = my_shelf['n']
    
    state.W = my_shelf['W']
    
    state.dist = my_shelf['dist']
    
    state.XHmr = my_shelf['XHmr']
    state.XFmr = my_shelf['XFmr']

    state.RHm = my_shelf['RHm']
    state.RFm = my_shelf['RFm']
    
    state.V = my_shelf['V']
    
    state.muH = my_shelf['muH']
    state.muF = my_shelf['muF']

    state.XHm = my_shelf['XHm']
    state.XFm = my_shelf['XFm']
    
    state.YH = my_shelf['YH']
    state.YHa = my_shelf['YHa']

    state.YF = my_shelf['YF']
    state.YFa = my_shelf['YFa']
    
    state.tauH = my_shelf['tauH']
    state.tauF = my_shelf['tauF']
    state.tauW = my_shelf['tauW']
    
    state.gammaH = my_shelf['gammaH']
    state.gammaF = my_shelf['gammaF']
    
    state.gammaHm = my_shelf['gammaHm']
    state.gammaFm = my_shelf['gammaFm']
    
    state.phiHm = my_shelf['phiHm']
    state.phiFm = my_shelf['phiFm']
    
    state.phiHa = my_shelf['phiHa']
    state.phiFa = my_shelf['phiFa']
    
    state.beta = 2
    state.phiH = 10
    state.phiF = 10
    state.nuH = 100
    state.nuF = 100
    
    db.close()

    return()

def save_data(file):
    db = dbm.dumb.open(file,'n')
    my_shelf = shelve.Shelf(db)
    
    for name in dir(state):
        if name.startswith('__'): continue;
            
        my_shelf[name] = getattr(state,name)
    
    for name in dir(result):
        if name.startswith('__'): continue;
    
        key = "result_{}".format(name)
        my_shelf[key] = getattr(result,name)
    
    db.close()
    
    return()

def assign_parameters():
    state.df=1

    state.nuH = 1
    state.nuF = 1

    state.phiH = 1
    state.phiF = 1

    state.phiHa = state.phiH
    state.phiHm = np.ones(state.M)*state.phiH

    state.phiFa = state.phiF
    state.phiFm = np.ones(state.M)*state.phiF

    state.gammaH = 0.5
    state.gammaF = 0.5

    state.gammaHm = 0.5 * np.ones(state.M)
    state.gammaFm = 0.5 * np.ones(state.M)

    state.tauH = 1
    state.tauF = 1
    state.tauW = 1

    state.beta = 1

    state.acceptGammaH = 0;
    state.acceptGammaF = 0;
    state.acceptNuH = 0;
    state.acceptNuF = 0;

    state.V = np.identity(state.M)

    state.XHm = np.zeros([state.M,state.n])
    state.XFm = np.zeros([state.M,state.n])
    for m in range(state.M):
        state.XHm[m,:]=np.mean(state.XHmr[m,0:state.RHm[m],:],axis=0)
        state.XFm[m,:]=np.mean(state.XFmr[m,0:state.RFm[m],:],axis=0)

    state.muH = np.mean(state.XHm,axis=0)
    state.muF = np.mean(state.XFm,axis=0)-np.mean((state.XHm-state.muH),axis=0)*state.beta
    state.YH = state.muH.copy()
    state.YF = state.muF.copy()
    state.YHa = np.mean(state.W,axis=0)
    state.YFa = state.YF.copy()
    state.invV = my_inv(state.V)
    state.VPrior = state.V.copy()

    state.N = state.W.shape[0]
    
    state.covMatH = np.exp(-state.dist/state.gammaH)
    state.covMatF = np.exp(-state.dist/state.gammaF)

    result.phiHa=np.zeros(state.nChain)
    result.phiFa=np.zeros(state.nChain)

    result.nuH=np.zeros(state.nChain)
    result.nuF=np.zeros(state.nChain)

    result.gammaH=np.zeros(state.nChain)
    result.gammaF=np.zeros(state.nChain)

    result.gammaHm = np.zeros([state.nChain,state.M])
    result.gammaFm = np.zeros([state.nChain,state.M])
    
    result.phiH=np.zeros(state.nChain)
    result.phiF=np.zeros(state.nChain)

    result.tauH=np.zeros(state.nChain)
    result.tauF=np.zeros(state.nChain)
    result.tauW=np.zeros(state.nChain)
    
    result.beta=np.zeros(state.nChain)
    
    result.phiHm = np.zeros([state.nChain,state.M])
    result.phiFm = np.zeros([state.nChain,state.M])
    
    result.muH=np.zeros([state.nChain,state.n])
    result.muF=np.zeros([state.nChain,state.n])

    result.YH=np.zeros([state.nChain,state.n])
    result.YHa_mean=np.zeros(state.n)
    result.YHa_one=np.zeros(state.nChain)
    
    result.YF=np.zeros([state.nChain,state.n])
    result.YFa_mean=np.zeros(state.n)
    result.YFa_one=np.zeros(state.nChain)
    
    result.XHm_mean=np.zeros(state.n)
    result.XFm_mean=np.zeros(state.n)
    
    result.XHm_one=np.zeros(state.nChain)
    result.XFm_one=np.zeros(state.nChain)

    result.V=np.zeros([state.nChain,state.M,state.M])
    result.V_mean=np.zeros([state.M,state.M])
    result.V_one=np.zeros(state.nChain)

    
    state.invCovMatH = my_inv(state.covMatH);
    state.invCovMatF = my_inv(state.covMatF);
    state.invV = my_inv(state.V);

    state.sumXHmr = np.zeros([state.M,state.n])
    state.sumXFmr = np.zeros([state.M,state.n])
    for m in range(state.M):
        state.sumXHmr[m,:]=np.sum(state.XHmr[m,0:state.RHm[m],:],axis=0)
        state.sumXFmr[m,:]=np.sum(state.XFmr[m,0:state.RFm[m],:],axis=0)

    state.epsHm = state.XHm-state.muH
    state.epsFm = state.XFm-state.muF-state.beta*state.epsHm
    state.epsHm = state.epsHm.transpose()
    state.epsFm = state.epsFm.transpose()

    state.epsYH = state.YH-state.muH
    state.epsYF = state.YF-state.muF-state.beta*state.epsYH

    state.XHmrDiff = np.zeros([state.M,int(max(state.RHm)),state.n])
    state.XFmrDiff = np.zeros([state.M,int(max(state.RHm)),state.n])
    
    state.covMatHm = np.zeros([state.M,state.n,state.n])
    state.covMatFm = np.zeros([state.M,state.n,state.n])

    state.invCovMatHm = np.zeros([state.M,state.n,state.n])
    state.invCovMatFm = np.zeros([state.M,state.n,state.n])
    
    state.rateHm = np.zeros(state.M)
    state.rateFm = np.zeros(state.M)
    
    state.acceptGammaHm = np.zeros(state.M)
    state.acceptGammaFm = np.zeros(state.M)
        
    state.logGammaHmProbPart1 = np.zeros(state.M)
    state.logGammaFmProbPart1 = np.zeros(state.M)
        
    for m in range(state.M):
        state.XHmrDiff[m,0:state.RHm[m],:]=state.XHmr[m,0:state.RHm[m],:]-state.XHm[m,:]
        state.XFmrDiff[m,0:state.RFm[m],:]=state.XFmr[m,0:state.RFm[m],:]-state.XFm[m,:]
    
        state.covMatHm[m] = np.exp(-state.dist/state.gammaHm[m])
        state.covMatFm[m] = np.exp(-state.dist/state.gammaFm[m])
        
        state.invCovMatHm[m] = my_inv(state.covMatHm[m])
        state.invCovMatFm[m] = my_inv(state.covMatFm[m])
        
        logDSign, logD = np.linalg.slogdet(state.covMatHm[m])
        state.logGammaHmProbPart1[m] = logDSign*logD*state.RHm[m]*(-0.5);
            
        logDSign, logD = np.linalg.slogdet(state.covMatFm[m])
        state.logGammaFmProbPart1[m] = logDSign*logD*state.RFm[m]*(-0.5);
    
    logDSign, logD = np.linalg.slogdet(state.covMatH)
    state.logGammaHProbPart1 = logDSign*logD*(state.M+1)*(-0.5);

    logDSign, logD = np.linalg.slogdet(state.covMatF)
    state.logGammaFProbPart1 = logDSign*logD*(state.M+1)*(-0.5);
    
    return()