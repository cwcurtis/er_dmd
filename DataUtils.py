"""
    Various functions and utilities for creating and plotting data sets.

    Created by:
        Opal Issan

    Modified:
        17 Nov 2020 - Jay Lago
        NOTE: Caveat emptor! These functions are not efficient nor consistent
"""
import numpy as np
from tqdm import tqdm

def rk4(lhs, dt, function):
    k1 = dt * function(lhs)
    k2 = dt * function(lhs + k1 / 2.0)
    k3 = dt * function(lhs + k2 / 2.0)
    k4 = dt * function(lhs + k3)
    rhs = lhs + 1.0 / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    return rhs

def rossler_lorenz63(lhs, Cpl):
    aval=0.2
    bval=0.2
    cval=7.
    rho=28.0
    sigma=10.0
    beta=8./3.
    
    rhs = np.zeros(6)
    rhs[0] = -(lhs[1] + lhs[2])
    rhs[1] = (lhs[0] + aval*lhs[1])
    rhs[2] = (bval + lhs[2] * (lhs[0] - cval))
    rhs[3] = sigma*(lhs[4] - lhs[3])
    rhs[4] = lhs[3]*(rho - lhs[5]) - lhs[4] + Cpl * lhs[1]**2.
    rhs[5] = lhs[3]*lhs[4] - beta*lhs[5]
    return rhs

def data_maker(pdim, num_ic, dt, tf, Cpl):
    x0 = 2.*(np.random.rand(num_ic, pdim) - .5)
    nsteps = int(tf / dt)
    data_mat = np.zeros((num_ic, pdim, nsteps + 1), dtype=np.float64)
    pfun = lambda lhs: rossler_lorenz63(lhs, Cpl)
    for ii in tqdm(range(num_ic), desc='Generating system data for coupled Lorenz/Rossler system', ncols=100):
        data_mat[ii, :, 0] = x0[ii, :]
        for jj in range(nsteps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, pfun)
    return data_mat
