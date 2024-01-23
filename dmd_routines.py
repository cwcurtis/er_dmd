import numpy as np
import knn_mi_comp as kmc
from copy import copy

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

def lagged_dmd_cmp(gm, gp, thrshhld, lags):
    pdim, nsteps = gm.shape
    mxlag = lags[-1]
    ypos = gp[:, mxlag:]
    yneg = np.zeros((len(lags)*pdim, nsteps-mxlag+1), dtype=np.float64)
    for cnt, lag in enumerate(lags[::-1]):
        yneg[cnt*pdim:(cnt+1)*pdim, :] = gm[:, mxlag-lag:nsteps-lag]
    u, s, vh = np.linalg.svd(yneg)
    si = np.diag(1./s)
    kmats = ypos @ vh.T @ si @ u.T
    return kmats

def it_dmd(max_lag, ydata):
    # ground expectation
    thrshhld = 15
    knghbr = 3
    kmatl = dmd_cmp(ydata[:, :-1], ydata[:, 1:], thrshhld)

    kmats = [kmatl]
    chosen_lags = [1]    

    # forward build
    significant = True
    remaining_lags = np.arange(2, max_lag)
    
    while significant:
        cmi_max = 0.01
        triggered = False
        for lcnt, lag in enumerate(remaining_lags):
            xvals = ydata[:, lag:]            
            zvals = 0.
            for cnt, kmat in enumerate(kmats):                
                zvals += kmat @ ydata[:, lag-cnt-1:-cnt-1]
                
            kmatl = dmd_cmp(ydata[:, :-lag], xvals-zvals, thrshhld)
            yvals = zvals + kmatl @ ydata[:, :-lag]
            cmi = kmc.cmiknn(xvals.T, yvals.T, zvals.T, knghbr)
            if cmi > cmi_max:
                cmi_max = cmi
                clag = lag
                ccnt = lcnt
                chosenkmat = kmatl
                xvalsmax = copy(xvals)
                yvalsmax = copy(yvals)                
                zvalsmax = copy(zvals)
                triggered = True

        if triggered:        
            dist = kmc.shuffle_test(xvalsmax.T, yvalsmax.T, zvalsmax.T, knghbr)
            if dist.cdf.evaluate(cmi_max) < .99:
                significant = False
            else:
                chosen_lags.append(clag)
                kmats.append(chosenkmat)
                if clag == max_lag:
                    significant = False
                else:
                    remaining_lags = np.delete(remaining_lags, np.arange(ccnt+1))
        else:
            significant = False

    print("Current choices for lags are:")
    print(chosen_lags)
    
    # backward prune
    if len(chosen_lags) > 1:
        yvalsmin = copy(yvalsmax)
        xvalsmin = ydata[:, max_lag:]
        while not(significant):
            cmi_min = 1e6
            for cnt, lag in enumerate(chosen_lags[1:]):
                zvals = yvalsmin - kmats[cnt+1] @ ydata[:,max_lag-lag:-lag]                
                cmi = kmc.cmiknn(xvalsmin.T, yvalsmin.T, zvals.T, knghbr)
                if cmi < cmi_min:
                    cmi_min = cmi
                    ccnt = cnt + 1
                    zvalsmin = copy(zvals)
            dist = kmc.shuffle_test(xvalsmin.T, yvalsmin.T, zvalsmin.T, knghbr)
            pval = dist.cdf.evaluate(cmi_min)
            
            if  pval < .99 and len(chosen_lags)>1:
                poppedcnt = chosen_lags.pop(ccnt)
                poppedkmat = kmats.pop(ccnt)            
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
        for ll in range(len(lags)):
            forecasts[:, jj+mxlag] += kmats[ll] @ forecasts[:, jj + mxlag - lags[ll]]
    return forecasts