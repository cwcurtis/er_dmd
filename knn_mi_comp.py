#
# * x : Mxd column wise matrix (source).
# * y : Mxr column wise matrix (distination).
# * z : Mxk column wise matrix of condition set.
# * options: Structure that has two fields: 
#            - options.K: the integer value K for the Kth 
#              nearest neighbor to be used in estimation.
#            - options.distFun: cell array that have name of the 
#              distance function to be used, which should match the
#              built-in distance function defined by matlab (see
#              pdist2)
#

import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import pairwise_distances
from scipy.special import psi
from scipy.spatial import distance
from scipy import stats

def miknn(x,y,knghbr):

    Npts = x.shape[0]
    
    allvar = np.concatenate((x,y),1)
    k_d, k_i = KDTree(allvar).query(allvar, k = knghbr+1, p=np.inf)
    evals = k_d[:,-1]
    
    dx = distance.cdist(x, x, metric='minkowski', p=np.inf)
    dy = distance.cdist(y, y, metric='minkowski', p=np.inf)
    
    nx = np.sum(dx < evals, 0)-1
    ny = np.sum(dy < evals, 0)-1
    
    I1 = psi(knghbr) + psi(Npts) - np.mean(psi(nx+1)+psi(ny+1)) 
    return I1

def cmiknn(x,y,z,knghbr):

    allvar = np.concatenate((x,y,z),1)
    xz = np.concatenate((x,z),1)
    yz = np.concatenate((y,z),1)
    
    k_d, k_i = KDTree(allvar).query(allvar, k = knghbr+1, p=np.inf)
    evals = k_d[:,-1]
    
    dxz = distance.cdist(xz, xz, metric='minkowski', p=np.inf)
    dyz = distance.cdist(yz, yz, metric='minkowski', p=np.inf)
    dzz = distance.cdist(z, z, metric='minkowski', p=np.inf)
    
    nxz = np.sum(dxz < evals, 0) - 1
    nyz = np.sum(dyz < evals, 0) - 1
    nzz = np.sum(dzz < evals, 0) - 1
    
    I1 = psi(knghbr) + np.mean(-psi(nxz+1)-psi(nyz+1)+psi(nzz+1)) 
    return I1

def shuffle_test(x,y,z,knghbr):
    nshuffle = 1600
    tstps = y.shape[0]
    shuffles = np.zeros(nshuffle)
    for jj in range(nshuffle):
        shuffles[jj] = cmiknn(x,y[np.random.permutation(tstps), :],z,knghbr)
    return stats.ecdf(shuffles)