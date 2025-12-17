import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------
# Set parameters
# ---------------------------------------------------------------------

N_samp = 100
lam = 0.1
r = 0.5
n_iter = 70

def d_atan(x):
    return 1.0 / (1.0 + x**2)

# initial X
np.random.seed(0)

X = norm.rvs(size = N_samp); X.sort()
Y = norm.rvs(size = N_samp); Y.sort()
alpha = norm.rvs(size = N_samp); alpha.sort()

tol = 1e-6   # stopping tolerance

# history of mu
mu_hist = []

for i in range(n_iter):
    mu_hist.append(alpha.copy())
    
    # first optimization
    dX = X - alpha + lam * (d_atan(X[:,None] - alpha[None,:]).mean(axis = 0) + X-1.0)

    while dX @ dX > tol:
        #print(d_eta1)
        X = X - r * dX
        X.sort()
        dX = X - alpha + lam * (d_atan(X[:,None] - alpha[None,:]).mean(axis = 0) + X-1.0)
    
    # second (extragradient) optimization
    beta = X.copy()
    dY = Y - alpha + lam * (d_atan(Y[:,None] - beta[None,:]).mean(axis = 0) + Y-1.0)

    while dY @ dY > tol:
        #print(d_eta2)
        Y = Y - r * dY
        Y.sort()
        dY = Y - alpha + lam * (d_atan(Y[:,None] - beta[None,:]).mean(axis = 0) + Y-1.0)
            
    alpha = Y.copy()
    
np.savez(
    "data.npz",
    mu=mu_hist,
    lam=lam,
    N_samp = N_samp,
    n_iter = n_iter,
    r=r
)