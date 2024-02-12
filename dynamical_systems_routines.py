import numpy as np
from tqdm import tqdm

def rk4(lhs, dt, function):
    k1 = dt * function(lhs)
    k2 = dt * function(lhs + k1 / 2.0)
    k3 = dt * function(lhs + k2 / 2.0)
    k4 = dt * function(lhs + k3)
    rhs = lhs + 1.0 / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    return rhs

def trajectory(func, ic, start, stop, dt):
    num_dims = np.size(ic)
    num_steps = int((stop - start)/dt)
    traj = np.zeros((num_steps, num_dims))
    traj[0, :] = ic
    for ii in range(1, num_steps):
        traj[ii, :] = rk4(traj[ii-1, :], dt, func)
    return traj

def vanderpol(lhs,mu=1.0):
    x1, x2 = lhs[0], lhs[1]
    rhs = np.zeros(2, dtype=np.float64)
    rhs[0] = x2
    rhs[1] = mu*(1.-x1**2.)*x2 - x1
    return rhs

def lorenz63(lhs, rho=28.0, sigma=10.0, beta=8./3.):
    """ Lorenz63 example:
    ODE =>
    dx1/dt = sigma*(x2 - x1)
    dx2/dt = x1*(rho - x3) - x2
    dx3/dt = x1*x2 - beta*x3
    """
    rhs = np.zeros(3)
    rhs[0] = sigma*(lhs[1] - lhs[0])
    rhs[1] = lhs[0]*(rho - lhs[2]) - lhs[1]
    rhs[2] = lhs[0]*lhs[1] - beta*lhs[2]
    return rhs


def lorenz96(lhs):
    F = 8.0
    rhs = -lhs + F + ( np.roll(lhs,-1) - np.roll(lhs,2) ) * np.roll(lhs,1)
    return rhs

def rossler(lhs, alpha=0.1, beta=0.1, gamma=14):
    """ Rossler system:
    ODE =>
    dx1/dt = -x2 - x3
    dx2/dt = x1 + alpha*x2
    dx3/dt = beta + x3*(x1 - gamma)
    """
    rhs = np.zeros(3)
    rhs[0] = -lhs[1] - lhs[2]
    rhs[1] = lhs[0] + alpha*lhs[1]
    rhs[2] = beta + lhs[2] * (lhs[0] - gamma)
    return rhs

def generate_vanderpol(x1min, x1max, x2min, x2max, 
                      num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    data_mat = np.zeros((num_ic, 2, num_steps+1), dtype=np.float64)
    for ii in tqdm(range(num_ic), 
                   desc='Generating VanderPol system data...', ncols=100):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, vanderpol)
    return data_mat

def generate_rossler(x1min, x1max, x2min, x2max, 
                        x3min, x3max, num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    icond3 = np.random.uniform(x3min, x3max, num_ic)
    data_mat = np.zeros((num_ic, 3, num_steps+1), dtype=np.float64)
    for ii in tqdm(range(num_ic), 
                   desc="Generating Rossler system data...", ncols=100):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii], icond3[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, rossler)
    return data_mat

def generate_lorenz63(x1min, x1max, x2min, x2max, 
                      x3min, x3max, num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    icond3 = np.random.uniform(x3min, x3max, num_ic)
    data_mat = np.zeros((num_ic, 3, num_steps+1), dtype=np.float64)
    for ii in tqdm(range(num_ic), 
                   desc='Generating Lorenz63 system data...', ncols=100):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii], icond3[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, lorenz63)
    return data_mat

def generate_lorenz96(xbounds, num_ic=15000, dim=8, dt=0.05, tf=20.0, seed=None):
    np.random.seed(seed=seed)
    nsteps = int(tf / dt)
    num_ic = int(num_ic)
    iconds = np.zeros((num_ic, dim), dtype=np.float64)
    for ll in range(dim):
        iconds[:, ll] = np.linspace(xbounds[ll, 0], xbounds[ll, 1], num_ic) + .05*2.*(np.random.rand(1)-.5)
    data_mat = np.zeros((num_ic, dim, nsteps + 1), dtype=np.float64)
    for ii in tqdm(range(num_ic), desc='Generating Lorenz96 system data...', ncols=100):
        data_mat[ii, :, 0] = iconds[ii, :]
        for jj in range(nsteps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, lorenz96)
    return data_mat
