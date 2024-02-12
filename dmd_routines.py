import numpy as np
import knn_mi_comp as kmc
from copy import copy
import scipy.linalg as sci
from sympy import *
from sympy.matrices import Matrix, eye, zeros, ones, diag
from numpy.polynomial import Polynomial

def dmd_cmp(gm, gp, thrshhld):
    
    u, s, vh = np.linalg.svd(gm, full_matrices=False)
    sm = np.max(s)
    indskp = np.log10(s/sm) > -thrshhld
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]
    kmat = gp @ vr @ np.diag(1. / sr) @ np.conj(ur.T)

    return kmat

def lagged_model_cmp(ydata, kmats, lags, mxlag):
    pdim, nsteps = ydata.shape    
    model = np.zeros((pdim, nsteps), dtype=np.float64)
    model[:, :mxlag] = ydata[:, :mxlag]
    for jj in range(nsteps-mxlag):
        tstep = jj + mxlag
        for cnt, lag in enumerate(lags[::-1]):
            if pdim > 1:
                model[:, tstep] += kmats[:, pdim*cnt:pdim*(cnt+1)] @ model[:, tstep - lag]
            else:
                model[:, tstep] += kmats[:, cnt] * model[:, tstep - lag]
    return model

def lagged_dmd_cmp(ydata,lags):
    pdim, nsteps = ydata.shape
    mxlag = lags[-1]
    ypos = ydata[:, mxlag:]
    yneg = np.zeros((len(lags)*pdim, nsteps-mxlag), dtype=np.float64)
    for cnt, lag in enumerate(lags[::-1]):
        yneg[cnt*pdim:(cnt+1)*pdim, :] = ydata[:, mxlag-lag:-lag]
    kmatstmp = sci.lstsq(yneg.T, ypos.T, cond=1e-15, lapack_driver='gelsd')
    kmats = kmatstmp[0].T     
    return kmats

def evals_computer(kmats, lags):
    z = symbols('z')
    
    pdim, ntots = np.shape(kmats)
    ktot = Matrix(np.zeros((pdim, pdim)))
    mxlag = lags[-1]
    norms = np.zeros(len(lags))
    for cnt, lag in enumerate(lags[::-1]):
        norms[cnt] = np.linalg.norm(kmats[:, pdim*cnt:pdim*(cnt+1)])
        ktot += Matrix(kmats[:, pdim*cnt:pdim*(cnt+1)]) * z**(mxlag-lag)        
    ktot -= (z**mxlag)*eye(pdim)
    print(norms)
    print("Matrix Built")
    charfun = Poly(ktot.det(), z)    
    print("Determinant computed")
    evals = np.roots(charfun.all_coeffs()[::-1])
    return evals

def it_dmd(max_lag, ydata):
    # ground expectation
    thrshhld = 15
    knghbr = 3
    kmatl = dmd_cmp(ydata[:, :-1], ydata[:, 1:], thrshhld)

    kmats = kmatl
    chosen_lags = [1]    

    # forward build
    significant = True
    remaining_lags = list(np.arange(2, max_lag))
    xvals = ydata[:, max_lag:]
    while significant:
        cmi_max = 0.01
        triggered = False
        for lcnt, lag in enumerate(remaining_lags):            
            #xvals = ydata[:, lag:]                                    
            zvals = lagged_model_cmp(ydata, kmats, chosen_lags, max_lag)                
            prop_lags = sorted(chosen_lags + [lag])
            kmats_prop = lagged_dmd_cmp(ydata, prop_lags)
            
            yvals = lagged_model_cmp(ydata, kmats_prop, prop_lags, max_lag)
            cmi = kmc.cmiknn(xvals.T, yvals[:, max_lag:].T, zvals[:, max_lag:].T, knghbr)
            if cmi > cmi_max:
                cmi_max = cmi
                clag = lag
                ccnt = lcnt
                chosenkmats = copy(kmats_prop)
                xvalsmax = copy(xvals)
                yvalsmax = copy(yvals[:, max_lag:])                
                zvalsmax = copy(zvals[:, max_lag:])
                triggered = True

        if triggered:        
            dist = kmc.shuffle_test(xvalsmax.T, yvalsmax.T, zvalsmax.T, knghbr)
            if dist.cdf.evaluate(cmi_max) < .99:
                significant = False                
            else:
                chosen_lags.append(clag)
                chosen_lags.sort()
                kmats = copy(chosenkmats)
                if clag == max_lag:
                    significant = False
                else:
                    poppedcnt = remaining_lags.pop(ccnt)
        else:
            significant = False

    print("Current choices for lags are:")
    print(chosen_lags)
    
    # backward prune
    if len(chosen_lags) > 2:
     
        yvalsmin = lagged_model_cmp(ydata, kmats, chosen_lags, chosen_lags[-1])
        while not(significant):
            xvalsmin = ydata[:, chosen_lags[-1]:]        
            cmi_min = 1e6
            for cnt, lag in enumerate(chosen_lags[1:]):
                
                prop_lags = chosen_lags[:cnt+1] + chosen_lags[cnt+2:]
                kmats_prop = lagged_dmd_cmp(ydata, prop_lags)
                zvals = lagged_model_cmp(ydata, kmats_prop, prop_lags, chosen_lags[-1])
                cmi = kmc.cmiknn(xvalsmin.T, yvalsmin[:, chosen_lags[-1]:].T, zvals[:, chosen_lags[-1]:].T, knghbr)                
                if cmi < cmi_min:
                    cmi_min = cmi
                    ccnt = cnt + 1
                    zvalsmin = copy(zvals)
                    chosenkmats = copy(kmats_prop)
                    
            dist = kmc.shuffle_test(xvalsmin.T, yvalsmin[:, chosen_lags[-1]:].T, zvalsmin[:, chosen_lags[-1]:].T, knghbr)
            pval = dist.cdf.evaluate(np.abs(cmi_min))
            
            if  pval < .99 and len(chosen_lags)>2:
                poppedcnt = chosen_lags.pop(ccnt)
                kmats = copy(chosenkmats)
                yvalsmin = copy(zvalsmin)
            else:
                significant = True
        
    return kmats, chosen_lags

def ed_dmd_iter(kmats, lags, nsteps, tints):
    pdim = kmats[0].shape[0]
    mxlag = lags[-1]
    forecasts = np.zeros((pdim, nsteps+mxlag), dtype=np.float64)
    forecasts[:, :mxlag] = tints
    
    for jj in range(nsteps):
        for cnt, lag in enumerate(lags[::-1]):
            forecasts[:, jj+mxlag] += kmats[cnt] @ forecasts[:, jj + mxlag - lag]
    return forecasts