import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------
# Set parameters
# ---------------------------------------------------------------------

N_samp = 100
lam = 0.1
alpha = 0.5

# initial X
np.random.seed(0)

mu = norm.rvs(size = N_samp); mu.sort()
eta1 = norm.rvs(size = N_samp); eta1.sort()
eta2 = norm.rvs(size = N_samp); eta2.sort()

tol = 1e-6   # stopping tolerance

for i in range(40):
    np.save(f'data/mu{i:02d}.npy', mu)
    
    # first optimization
    d_eta1 = eta1 - mu + lam * (np.cos(mu[:,None] - eta1[None,:]).mean(axis = 0) + 2*eta1)

    while d_eta1 @ d_eta1 > tol:
        #print(d_eta1)
        eta1 = eta1 - alpha * d_eta1
        eta1.sort()
        d_eta1 = eta1 - mu - lam * (np.cos(mu[:,None] - eta1[None,:]).mean(axis = 0) + 2*eta1)
    
    # second (extragradient) optimization
    nu = eta1.copy()
    d_eta2 = eta2 - mu + lam * (np.cos(nu[:,None] - eta1[None,:]).mean(axis = 0) + 2*eta2)

    while d_eta2 @ d_eta2 > tol:
        #print(d_eta2)
        eta2 = eta2 - alpha * d_eta2
        eta2.sort()
        d_eta2 = eta2 - mu + lam * (np.cos(nu[:,None] - eta1[None,:]).mean(axis = 0) + 2*eta2)
            
    mu = eta2.copy()