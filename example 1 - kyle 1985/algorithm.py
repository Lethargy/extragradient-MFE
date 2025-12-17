import numpy as np
from scipy.special import roots_hermite
from scipy.stats import norm
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------
# Helper: PAV (isotonic) projection onto x1 <= ... <= xn, O(n)
# ---------------------------------------------------------------------
def pav(y: np.ndarray) -> np.ndarray:
    """L2 projection of y onto the monotone nondecreasing cone."""
    y = np.asarray(y, dtype=float)
    lvl = []
    wts = []
    for val in y:
        lvl.append(val)
        wts.append(1)
        while len(lvl) >= 2 and lvl[-2] > lvl[-1]:
            w = wts[-2] + wts[-1]
            avg = (wts[-2] * lvl[-2] + wts[-1] * lvl[-1]) / w
            lvl[-2] = avg
            wts[-2] = w
            lvl.pop(); wts.pop()
    out = np.empty_like(y)
    i = 0
    for avg, w in zip(lvl, wts):
        out[i:i+w] = avg
        i += w
    return out

# ---------------------------------------------------------------------
# Set parameters
# ---------------------------------------------------------------------

N_samp = 100
N_iter = 50
lam = 0.1
rho = 0.5
sig = 0.5

s = 1
np.random.seed(s)

# initial V
V = norm.rvs(size = N_samp); V.sort()

# Gauss–Hermite nodes / weights for N(0, σ²)
n_gh = 20
z, p = roots_hermite(n_gh)
z = np.sqrt(2.0) * sig * z        # nodes for N(0, σ²)
p = p / np.sqrt(np.pi)            # weights for N(0, σ²)

def dC(x, mu):
    """
    Vectorized gradient ∇C_μ(x).

    x : scalar or 1D array of evaluation points
    Uses global V, z, p, sig.
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


# history of mu
mu_hist = []

# initial X
alpha = norm.rvs(size = N_samp); alpha.sort()
X = norm.rvs(size = N_samp); X.sort()
Y = norm.rvs(size = N_samp); Y.sort()

tol = 1e-6   # stopping tolerance

for i in range(N_iter):
    mu_hist.append(alpha.copy())
    
    # first optimization
    T = interp1d(alpha, V, fill_value = 'extrapolate') # optimal map from mu to V
    dX = X - alpha + lam * (dC(X, alpha) - T(X))

    while dX @ dX > tol:
        X = pav(X - rho * dX)
        dX = X - alpha + lam * (dC(X, alpha) - T(X))
    
    # second (extragradient) optimization
    beta = X.copy()
    T = interp1d(beta, V, fill_value = 'extrapolate') # optimal map from nu to V
    dY = Y - alpha + lam * (dC(Y, beta) - T(Y))

    while dY @ dY > tol:
        Y = pav(Y - rho * dY)
        dY = Y - alpha + lam * (dC(Y, beta) - T(Y))
            
    alpha = Y.copy()
    
np.savez(
    "data.npz",
    mu=np.stack(mu_hist),
    V=V,
    lam=lam,
    sig=sig,
    rho=rho,
    N_samp = N_samp,
    N_iter = N_iter
)