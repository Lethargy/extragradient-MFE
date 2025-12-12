import numpy as np
from scipy.special import roots_hermite
from scipy.stats import norm
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------
# Set parameters
# ---------------------------------------------------------------------

N_samp = 100
lam = 0.1
alpha = 0.5
sig = 0.5

s = 1
np.random.seed(s)

# initial V
V = norm.rvs(size = N_samp); V.sort()
np.save('data/V.npy', V)

# Gauss–Hermite nodes / weights for N(0, σ²)
n_gh = 20
z, p = roots_hermite(n_gh)
z = np.sqrt(2.0) * sig * z        # nodes for N(0, σ²)
p = p / np.sqrt(np.pi)            # weights for N(0, σ²)

def dC(x):
    """
    Vectorized gradient ∇C_μ(x).

    x : scalar or 1D array of evaluation points
    Uses global mu, V, z, p, sig.
    """
    x_arr = np.atleast_1d(x).astype(float)    # shape (M,)

    # Broadcasted shapes:
    #   X  : (M, 1, 1)
    #   Z  : (1, n_gh, 1)
    #   MU : (1, 1, N_samp)
    X  = x_arr[:, None, None]
    Z  = z[None, :, None]
    MU = mu[None, None, :]

    # Gaussian-like weights in m_i for each (x_j, z_k)
    diff = X + Z - MU                          # (M, n_gh, N_samp)
    w_unnorm = np.exp(-0.5 * (diff / sig)**2)
    w = w_unnorm / w_unnorm.sum(axis=2, keepdims=True)

    # E[V | x_j, z_k]
    Ev = np.sum(w * V[None, None, :], axis=2)  # (M, n_gh)

    # factor 1 + x z / σ²
    factor = 1.0 + (X[:, :, 0] * Z[:, :, 0]) / (sig**2)  # (M, n_gh)

    # Integrate over z with GH weights
    grad = (Ev * factor) @ p                   # (M,)

    return grad[0] if np.isscalar(x) else grad


# initial X
mu = norm.rvs(size = N_samp); mu.sort()
eta1 = norm.rvs(size = N_samp); eta1.sort()
eta2 = norm.rvs(size = N_samp); eta2.sort()

tol = 1e-6   # stopping tolerance

for i in range(50):
    np.save(f'data/mu{i}.npy', mu)
    
    # first optimization
    T = interp1d(mu, V, fill_value = 'extrapolate') # optimal map from mu to V
    d_eta1 = eta1 - mu - lam * (T(eta1) - dC(eta1))

    while d_eta1 @ d_eta1 > tol:
        #print(d_eta1)
        eta1 = eta1 - alpha * d_eta1
        eta1.sort()
        d_eta1 = eta1 - mu - lam * (T(eta1) - dC(eta1))
    
    # second (extragradient) optimization
    nu = eta1.copy()
    T = interp1d(nu,V, fill_value = 'extrapolate') # optimal map from nu to V
    d_eta2 = eta2 - mu - lam * (T(eta2) - dC(eta2))

    while d_eta2 @ d_eta2 > tol:
        #print(d_eta2)
        eta2 = eta2 - alpha * d_eta2
        eta2.sort()
        d_eta2 = eta2 - mu - lam * (T(eta2) - dC(eta2))
            
    mu = eta2.copy()