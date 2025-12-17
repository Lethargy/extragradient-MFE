import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------
# Set parameters
# ---------------------------------------------------------------------

N_samp = 100
lam = 0.1
r = 0.5

# initial X
np.random.seed(0)

X = norm.rvs(size = N_samp); X.sort()
Y = norm.rvs(size = N_samp); Y.sort()
alpha = norm.rvs(size = N_samp); alpha.sort()

tol = 1e-6   # stopping tolerance

for i in range(40):
    np.save(f'data/mu{i:02d}.npy', alpha)
    
    # first optimization
    dX = X - alpha + lam * (np.cos(X[:,None] - alpha[None,:]).mean(axis = 0) + 2*X)

    while dX @ dX > tol:
        #print(d_eta1)
        X = X - r * dX
        X.sort()
        dX = X - alpha + lam * (np.cos(X[:,None] - alpha[None,:]).mean(axis = 0) + 2*X)
    
    # second (extragradient) optimization
    beta = X.copy()
    dY = Y - alpha + lam * (np.cos(Y[:,None] - beta[None,:]).mean(axis = 0) + 2*Y)

    while dY @ dY > tol:
        #print(d_eta2)
        Y = Y - r * dY
        Y.sort()
        dY = Y - alpha + lam * (np.cos(Y[:,None] - beta[None,:]).mean(axis = 0) + 2*Y)
            
    alpha = Y.copy()