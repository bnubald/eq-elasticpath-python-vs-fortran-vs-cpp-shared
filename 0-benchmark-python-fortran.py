#!/usr/bin/env python
# coding: utf-8

# Making the imports
import time
import numpy as np
import matplotlib.pyplot as plt
from solver_python import _elastic_net_cd_py
from solver_fortran import elastic_net_cd_for, elastic_net_cd_purefor
from solver_cpp import elastic_net_cd_cpp, elastic_net_cd_cpp_nosets
import matplotlib

np.random.seed(0)

def get_lamdas(X,y,n_lambdas,lamda_eps,alpha):
    n,m = X.shape

    # Get list of lambda's
    Xy = (X.T@(y-np.mean(y))*n).reshape(-1) #*n as sum over n

    # From sec 2.5 (with normalisation factors added)
    Xy /= (np.sum(X**2, axis=0))
    lamda_max = np.max(np.abs(Xy[1:]))/(n*alpha) #1: in here as not applying regularisation to intercept
    if lamda_max <= np.finfo(float).resolution:
        lamdas = np.empty(n_lamdas)
        lamdas.fill(np.finfo(float).resolution)
        return lamdas
    return np.logspace(np.log10(lamda_max * lamda_eps), np.log10(lamda_max),
                           num=n_lamdas)[::-1]

A, B = 20, 20
xn = np.linspace(100, 10000, num=A).astype(int)
xm = np.linspace(5, 1000, num=B).astype(int)

alpha = 1.0
n_lamdas = 100
lamda_eps = 1e-6

from equadratures.datasets import gen_linear

Xo, Yo = np.meshgrid(xn, xm)
ftimes = np.zeros((A, B))
ptimes = np.zeros((A, B))
cpptimes = np.zeros((A, B))

for i, r in enumerate(xn):
    print(i, '--', A )
    for j, c in enumerate(xm):
        X, y = gen_linear(n_observations=r, n_dim=c, n_relevent=c, bias=0.0, noise=0.0, random_seed=0)

        n, m = X.shape

        alpha = 1.0
        n_lamdas = 100
        lamda_eps = 1e-6

        lamdas = get_lamdas(X,y,n_lamdas,lamda_eps,alpha)

        X  = np.asfortranarray(X, dtype=np.float64)
        y  = np.asfortranarray(y, dtype=np.float64)

        #Run lasso regression for each lambda (theta passed back in for warm-start)
        theta = np.zeros((m, 1), dtype=np.float64)
        thetas = np.empty([len(lamdas),m], dtype=np.float64)

        start = time.time()
        for l, lamda in enumerate(lamdas):
            elastic_net_cd_cpp(theta,X,y,lamda,alpha,100,1e-5,False)
            thetas[l,:] = theta.ravel()
        end = time.time()
        cpptime = end - start

        cpptimes[i, j] = cpptime

        #Run lasso regression for each lambda (theta passed back in for warm-start)
        theta = np.zeros(m, dtype=np.float64)
        thetas = np.empty([len(lamdas),m], dtype=np.float64)

        start = time.time()
        for l, lamda in enumerate(lamdas):
            elastic_net_cd_for(theta,X,y,lamda,alpha,100,1e-5,False)
            thetas[l,:] = theta
        end = time.time()
        ftime = end - start

        ftimes[i, j] = ftime

        #Run lasso regression for each lambda (theta passed back in for warm-start)
        theta = np.zeros(m, dtype=np.float64)
        thetas = np.empty([len(lamdas),m], dtype=np.float64)

        start = time.time()
        for l, lamda in enumerate(lamdas):
            theta = _elastic_net_cd_py(theta,X,y,lamda,alpha,100,1e-5,False)
            thetas[l,:] = theta
        end = time.time()
        ptime = end - start

        ptimes[i, j] = ptime

matplotlib.rcParams.update({'font.size': 16})
fig, (ax1) = plt.subplots(1, 1, figsize=(40, 10))

a = ax1.imshow(ptimes/ftimes, cmap=plt.cm.inferno, interpolation='bicubic', origin="lower")

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig.colorbar(a, ax=ax1, orientation='vertical', shrink=0.75, label='Speedup factor')

ax1.set_title(r'Coordinate Descent Python vs Fortran Speedup', fontdict={'fontsize': 20})
ax1.xaxis.set_label_text("Observations")
ax1.yaxis.set_label_text("Dimensions")

# plt.show()
